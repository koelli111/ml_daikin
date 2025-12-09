# pyscript/daikin_ml.py
# Online-RLS (oppiva) ohjain Daikin demand-selectille.
# Ominaisuudet:
# - min 30 %, max 100 %
# - 6h ulkoennusteen keskiarvo feed-forwardina
# - pakotettu suunta-askel (aina vähintään step_limit kohti setpointtia deadbandin ulkopuolella)
# - deadband, step-limit, monotonisuus ja selectin pykälöinti
# - oppiminen (RLS) jäädytetään defrostin aikana (sensor.faikin_liquid < 20)
# - parametrisäilö pysyvässä STORE_ENTITY-entiteetissä (vain theta tallennetaan, ei P)
# - PÖRSSISÄHKÖOHJAUS: sensor.day_ahead_price
# - UI-säädöt: pörssisähkön herkkyys, min/max-kerroin, max setpoint-drop

import json
import time
from math import isfinite
from datetime import datetime

# --------- ENTITYT (VAIHDA OMIKSI) ----------
INDOOR      = "sensor.living_room_lampotila"     # sisälämpötila
INDOOR_RATE = "sensor.sisalampotila_aula"        # derivaatta (°C/h)
OUTDOOR     = "sensor.iv_tulo_lampotila"         # ulkolämpötila
WEATHER     = "weather.koti"                     # weather.* josta forecast-attribuutti
SELECT      = "select.faikin_demand_control"     # Daikin demand-select
LIQUID      = "sensor.faikin_liquid"             # defrost-indikaattori

SP_HELPER         = "input_number.daikin_setpoint"
STEP_LIMIT_HELPER = "input_number.daikin_step_limit"
DEADBAND_HELPER   = "input_number.daikin_deadband"

# Icing cap helper
ICING_CAP_HELPER = "input_number.daikin_icing_cap"

# Pörssisähkö-ohjauksen helperit:
PRICE_SENS_HELPER = "input_number.daikin_price_sensitivity"
PRICE_FMIN_HELPER = "input_number.daikin_price_factor_min"
PRICE_FMAX_HELPER = "input_number.daikin_price_factor_max"
# Maksimi setpoint-alennus (°C) kalleimpana hetkenä
PRICE_SP_DROP_HELPER = "input_number.daikin_price_sp_drop_max"

# ML-oppien pysyvä storage
STORE_ENTITY = "pyscript.daikin_ml_params"

# Learned demand -sensori
LEARNED_SENSOR = "sensor.daikin_ml_learned_demand"

# Pörssisähkö-sensori (day-ahead, varttihinnoittelu)
PRICE_SENSOR = "sensor.day_ahead_price"

# ---------- SÄÄTÖNUPIT (oletukset, jos helper puuttuu) ----------
HORIZON_H  = 1.0
FORECAST_H = 6
LAMBDA     = 0.995
P0         = 1e4
MIN_DEM    = 30.0
MAX_DEM    = 100.0
KAPPA      = 1.0

GLOBAL_MILD_MAX = 95.0

COOLDOWN_MINUTES  = 15
COOLDOWN_STEP_UP  = 3.0

ICING_BAND_MIN = -2.0
ICING_BAND_MAX =  4.0
ICING_BAND_CAP_DEFAULT = 80.0

AUTO_STEP_MIN  = 3.0
AUTO_STEP_MAX  = 20.0
AUTO_STEP_BASE = 10.0

AUTO_DB_MIN    = 0.05
AUTO_DB_MAX    = 0.50
AUTO_DB_BASE   = 0.10

# Pörssisähkön oletukset (käytetään jos helper-arvo ei kelpaa)
PRICE_SENSITIVITY_DEFAULT = 0.4   # 0–1
PRICE_FACTOR_MIN_DEFAULT  = 0.5   # 0–1
PRICE_FACTOR_MAX_DEFAULT  = 1.0   # 0–1
PRICE_SP_DROP_MAX_DEFAULT = 1.5   # °C, maksimi SP-alennus kalleimpana hetkenä

# ---------- Apufunktio: turvallinen float-muunnos ----------
def _safe_float(val, default):
    """
    Palauttaa float(val) tai default, jos val on None / 'unknown' / 'unavailable' tms.
    """
    try:
        if val is None:
            return default
        s = str(val).strip().lower()
        if s in ("unknown", "unavailable", "none", "nan", ""):
            return default
        return float(val)
    except Exception:
        return default

# ---------- Konteksti per ulkolämpöaste 1.0°C ----------
def _context_key_for_outdoor(Tout: float) -> str:
    if not isfinite(Tout):
        return "nan"
    bucket = int(round(Tout))
    return str(bucket)

_theta_by_ctx   = {}
_P_by_ctx       = {}
_params_loaded  = False

_last_defrosting = None
_cooldown_until  = 0.0

# ---------- PYSYVÄ STORE ----------
state.persist(STORE_ENTITY)

def _load_params_from_store():
    global _theta_by_ctx
    attrs = state.getattr(STORE_ENTITY) or {}
    tb = attrs.get("theta_by_ctx")
    new = {}
    if isinstance(tb, dict):
        for key, val in tb.items():
            if isinstance(val, (list, tuple)) and len(val) == 4:
                try:
                    th0 = float(val[0])
                    th1 = float(val[1])
                    th2 = float(val[2])
                    th3 = float(val[3])
                    ok = True
                    for t in (th0, th1, th2, th3):
                        if not isfinite(t):
                            ok = False
                            break
                    if ok:
                        new[str(key)] = [th0, th1, th2, th3]
                    else:
                        log.warning(
                            "Daikin ML: dropping non-finite stored theta for ctx=%s: %s",
                            key, val,
                        )
                except Exception:
                    continue
    if new:
        _theta_by_ctx = new
        log.info("Daikin ML: loaded %d contexts from %s", len(_theta_by_ctx), STORE_ENTITY)
    else:
        log.info("Daikin ML: no stored thetas in %s, starting fresh", STORE_ENTITY)

def _save_params_to_store():
    global _theta_by_ctx
    clean = {}
    for key, th in _theta_by_ctx.items():
        if not isinstance(th, (list, tuple)) or len(th) != 4:
            continue
        clean[str(key)] = [
            round(float(th[0]), 4),
            round(float(th[1]), 4),
            round(float(th[2]), 4),
            round(float(th[3]), 4),
        ]
    try:
        state.set(
            STORE_ENTITY,
            value=time.time(),
            theta_by_ctx=clean,
        )
    except Exception as e:
        log.error("Daikin ML: error saving thetas to %s: %s", STORE_ENTITY, e)

def _init_context_params_if_needed():
    global _theta_by_ctx, _P_by_ctx, _params_loaded
    if _params_loaded:
        return
    _theta_by_ctx = {}
    _P_by_ctx = {}
    _load_params_from_store()
    for key in _theta_by_ctx.keys():
        _P_by_ctx[str(key)] = [[P0, 0, 0, 0],
                               [0, P0, 0, 0],
                               [0, 0, P0, 0],
                               [0, 0, 0, P0]]
    _params_loaded = True

# ---------- Matriisiapuja ----------
def _dot(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

def _matvec(M, v):
    out = [0.0, 0.0, 0.0, 0.0]
    for i in range(4):
        s = 0.0
        for j in range(4):
            s += M[i][j] * v[j]
        out[i] = s
    return out

def _matsub(A, B):
    C = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            C[i][j] = A[i][j] - B[i][j]
    return C

def _matscale(A, s):
    C = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            C[i][j] = s * A[i][j]
    return C

def _rls_update(theta, P, x, y, lam=LAMBDA):
    """
    RLS-päivitys 4-parametriselle thetalle ja 4x4 P:lle:

        Px    = P x
        denom = λ + x^T P x
        K     = Px / denom
        θ_new = θ_old + K (y - x^T θ_old)
        P_new = (P - Px Px^T / denom) / λ

    Jos päivityksen jälkeen theta/P eivät ole finiittejä, resetoidaan konteksti.
    Ei käytetä generaattoreita (Pyscriptin rajoite).
    """
    Px = _matvec(P, x)
    denom = lam + _dot(x, Px)

    if (not isfinite(denom)) or denom == 0.0:
        log.warning("Daikin ML: RLS denom not finite (%.6g), skipping update", denom)
        return theta, P

    gain = 1.0 / denom
    err_est = y - _dot(x, theta)

    theta = [
        theta[0] + Px[0] * gain * err_est,
        theta[1] + Px[1] * gain * err_est,
        theta[2] + Px[2] * gain * err_est,
        theta[3] + Px[3] * gain * err_est,
    ]

    Px_outer = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            Px_outer[i][j] = Px[i] * Px[j] * gain

    P = _matscale(_matsub(P, Px_outer), 1.0 / lam)

    # Tarkistetaan, että theta ja P ovat finiittejä
    all_ok = True
    for t in theta:
        if not isfinite(t):
            all_ok = False
            break
    if all_ok:
        for i in range(4):
            for j in range(4):
                if not isfinite(P[i][j]):
                    all_ok = False
                    break
            if not all_ok:
                break

    if not all_ok:
        log.warning("Daikin ML: RLS produced non-finite theta/P, resetting this context")
        theta = [0.0, 0.0, 5.0, 0.0]
        P = [[P0, 0, 0, 0],
             [0, P0, 0, 0],
             [0, 0, P0, 0],
             [0, 0, 0, P0]]

    return theta, P

def _clip(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v

# ---------- Sääennuste ----------
def _avg_future_outdoor():
    attrs = state.getattr(WEATHER) or {}
    fc = attrs.get("forecast") or []
    temps = []
    idx = 0
    for f in fc:
        if idx >= FORECAST_H:
            break
        t = f.get("temperature")
        if t is not None:
            try:
                temps.append(float(t))
                idx += 1
            except Exception:
                pass
    if len(temps) > 0:
        s = 0.0
        for v in temps:
            s += v
        return s / float(len(temps))
    try:
        return _safe_float(state.get(OUTDOOR), 0.0)
    except Exception:
        return 0.0

# ---------- Pörssisähkö: haetaan nykyinen hinta ja päivän min/max ----------
def _get_price_info():
    """
    Palauttaa (price_now, price_min, price_max) sensor.day_ahead_price -sensorista.
    Käyttää attributes['records'] -listaa, jossa on Time, End, Price.
    Jos jotain ei löydy, palauttaa (None, None, None).
    """
    try:
        attrs = state.getattr(PRICE_SENSOR) or {}
    except Exception:
        return None, None, None

    recs = attrs.get("records")
    if not isinstance(recs, list) or len(recs) == 0:
        return None, None, None

    price_min = None
    price_max = None
    price_now = None

    now_naive = datetime.now()

    for rec in recs:
        price_raw = rec.get("Price")
        try:
            p = float(price_raw)
        except Exception:
            continue

        if price_min is None or p < price_min:
            price_min = p
        if price_max is None or p > price_max:
            price_max = p

        # Yritetään päätellä nykyinen vartti
        t_str = rec.get("Time")
        e_str = rec.get("End")
        if t_str is not None:
            try:
                t0 = datetime.fromisoformat(str(t_str))
            except Exception:
                t0 = None
            if e_str is not None:
                try:
                    t1 = datetime.fromisoformat(str(e_str))
                except Exception:
                    t1 = None
            else:
                t1 = None

            if t0 is not None:
                if t0.tzinfo is not None:
                    now_cmp = datetime.now(t0.tzinfo)
                else:
                    now_cmp = now_naive

                in_interval = False
                if t1 is not None:
                    if (t0 <= now_cmp) and (now_cmp < t1):
                        in_interval = True
                else:
                    if t0 <= now_cmp:
                        in_interval = True

                if in_interval:
                    price_now = p

    return price_now, price_min, price_max

# ---------- Selectin optiot ----------
def _with_pct():
    attrs = state.getattr(SELECT) or {}
    opts = attrs.get("options") or []
    return ('%' in (opts[0] if opts else ''))

def _select_options_nums():
    attrs = state.getattr(SELECT) or {}
    opts = attrs.get("options") or []
    nums = []
    for o in opts:
        try:
            s = str(o).replace('%', '').strip()
            nums.append(float(s))
        except Exception:
            pass
    nums.sort()
    has_pct = False
    if len(opts) > 0 and ('%' in str(opts[0])):
        has_pct = True
    return nums, has_pct

def _snap_to_select(value, direction):
    nums, has_pct = _select_options_nums()
    picked = None
    if len(nums) == 0:
        picked = round(value)
        out = str(int(picked)) + ('%' if (has_pct or _with_pct()) else '')
        return out
    if direction > 0:
        chosen = None
        for n in nums:
            if n >= value:
                chosen = n
                break
        if chosen is None:
            chosen = nums[-1]
        picked = chosen
    elif direction < 0:
        chosen = None
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] <= value:
                chosen = nums[i]
                break
        if chosen is None:
            chosen = nums[0]
        picked = chosen
    else:
        best = nums[0]
        bestd = abs(nums[0] - value)
        for n in nums:
            d = abs(n - value)
            if d < bestd:
                bestd = d
                best = n
        picked = best
    if float(picked).is_integer():
        picked = int(picked)
    return str(picked) + ('%' if has_pct else '')

# ---------- AUTO-TUNING ----------
def _auto_tune_helpers(theta, ctx, step_limit_current, deadband_current):
    try:
        gain = abs(float(theta[2]))
    except Exception:
        gain = 1.0
    if gain < 0.1:
        gain = 0.1
    if gain > 10.0:
        gain = 10.0
    step_raw = AUTO_STEP_BASE / gain
    step_auto = _clip(step_raw, AUTO_STEP_MIN, AUTO_STEP_MAX)
    db_raw = AUTO_DB_BASE * (1.0 + 0.2 * gain)
    db_auto = _clip(db_raw, AUTO_DB_MIN, AUTO_DB_MAX)
    step_auto_rounded = round(step_auto, 1)
    db_auto_rounded   = round(db_auto, 2)
    if (abs(step_auto_rounded - step_limit_current) > 0.1 or
            abs(db_auto_rounded - deadband_current) > 0.01):
        log.info(
            "Daikin ML AUTO-TUNE suggestion (ctx=%s): gain=%.3f -> "
            "step_limit=%.1f (was %.1f), deadband=%.2f (was %.2f)",
            ctx, gain,
            step_auto_rounded, step_limit_current,
            db_auto_rounded,   deadband_current,
        )
    return step_auto_rounded, db_auto_rounded

# ---------- DIAG ----------
@time_trigger("startup")
def _ml_startup_ok():
    _init_context_params_if_needed()
    # alustetaan learned-demand sensori näkyviin
    try:
        state.set(
            LEARNED_SENSOR,
            value=0.0,
            ctx="init",
            note="ML+price demand (dem_clip) before select snapping",
        )
    except Exception as e:
        log.error("Daikin ML: failed to init learned demand sensor: %s", e)

    log.info(
        "Daikin ML: startup loaded (per-1.0°C + persistent store: %s, auto-tuning helpers, learned-demand sensor)",
        STORE_ENTITY,
    )

@time_trigger("cron(*/6 * * * *)")
def _ml_tick6():
    log.debug("Daikin ML: 6-minute cron tick")

# ---------- Varsinainen ohjain ----------
@state_trigger(f"{INDOOR}", f"{OUTDOOR}", f"{INDOOR_RATE}")
@time_trigger("cron(*/6 * * * *)")
def daikin_ml_controller(**kwargs):
    global _last_defrosting, _cooldown_until, _theta_by_ctx, _P_by_ctx

    _init_context_params_if_needed()

    # ---- LUE SENSORIT ----
    Tin      = _safe_float(state.get(INDOOR), float("nan"))
    Tout_raw = _safe_float(state.get(OUTDOOR), float("nan"))
    rate     = _safe_float(state.get(INDOOR_RATE), 0.0)

    sp_base = _safe_float(state.get(SP_HELPER), 22.5)

    step_limit_current = _safe_float(state.get(STEP_LIMIT_HELPER), 10.0)
    deadband_current   = _safe_float(state.get(DEADBAND_HELPER), 0.1)

    prev_str = state.get(SELECT) or ""
    try:
        prev = float(str(prev_str).replace('%', ''))
    except Exception:
        prev = 60.0

    liquid = _safe_float(state.get(LIQUID), 100.0)
    defrosting = (liquid < 10.0)

    # ---- DEFROST / COOLDOWN ----
    now = time.time()
    if _last_defrosting is None:
        _last_defrosting = defrosting
    if (_last_defrosting is True) and (defrosting is False):
        _cooldown_until = now + COOLDOWN_MINUTES * 60.0
        log.info("Daikin ML: defrost ended -> cooldown active for %d min", COOLDOWN_MINUTES)
    _last_defrosting = defrosting
    in_cooldown = (now < _cooldown_until)

    if not (isfinite(Tin) and isfinite(Tout_raw)):
        log.info("Daikin ML: sensors not ready; Tin=%s Tout=%s", state.get(INDOOR), state.get(OUTDOOR))
        return

    # ---- KONTEXTI ----
    Tout_bucket = int(round(Tout_raw))
    ctx = _context_key_for_outdoor(Tout_raw)

    if ctx not in _theta_by_ctx:
        _theta_by_ctx[ctx] = [0.0, 0.0, 5.0, 0.0]
    if ctx not in _P_by_ctx:
        _P_by_ctx[ctx] = [[P0, 0, 0, 0],
                          [0, P0, 0, 0],
                          [0, 0, P0, 0],
                          [0, 0, 0, P0]]

    theta = _theta_by_ctx[ctx]
    P     = _P_by_ctx[ctx]

    step_limit, deadband = _auto_tune_helpers(theta, ctx, step_limit_current, deadband_current)

    # ---- CAP-BUCKETIT / ICING ----
    if Tout_bucket <= -5:
        global_upper = MAX_DEM
    else:
        global_upper = GLOBAL_MILD_MAX

    icing_cap = _safe_float(state.get(ICING_CAP_HELPER), ICING_BAND_CAP_DEFAULT)
    icing_cap = _clip(icing_cap, MIN_DEM, MAX_DEM)

    in_icing_band = (Tout_bucket >= ICING_BAND_MIN and Tout_bucket <= ICING_BAND_MAX)

    if in_icing_band:
        band_upper = icing_cap if icing_cap < global_upper else global_upper
    else:
        band_upper = global_upper

    # jos edellinen yli band_upper, tiputetaan heti
    if prev > band_upper:
        option_cap = _snap_to_select(band_upper, -1)
        if option_cap and option_cap != prev_str:
            select.select_option(entity_id=SELECT, option=option_cap)
            log.info(
                "Daikin ML: CAP ENFORCED: prev=%s -> %s (upper=%.0f%%, ctx=%s, Tout_bucket=%d)",
                prev_str, option_cap, band_upper, ctx, Tout_bucket,
            )
        return

    prev_eff = prev if prev <= band_upper else band_upper

    # ---- PÖRSSISÄHKÖ: parametrit & hinta ----
    price_sens = _safe_float(state.get(PRICE_SENS_HELPER), PRICE_SENSITIVITY_DEFAULT)
    if price_sens < 0.0:
        price_sens = 0.0
    if price_sens > 1.0:
        price_sens = 1.0

    pf_min = _safe_float(state.get(PRICE_FMIN_HELPER), PRICE_FACTOR_MIN_DEFAULT)
    if pf_min < 0.0:
        pf_min = 0.0
    if pf_min > 1.0:
        pf_min = 1.0

    pf_max = _safe_float(state.get(PRICE_FMAX_HELPER), PRICE_FACTOR_MAX_DEFAULT)
    if pf_max < 0.0:
        pf_max = 0.0
    if pf_max > 1.0:
        pf_max = 1.0

    if pf_max < pf_min:
        pf_max = pf_min

    # max setpoint-drop
    sp_drop_max = _safe_float(state.get(PRICE_SP_DROP_HELPER), PRICE_SP_DROP_MAX_DEFAULT)
    if sp_drop_max < 0.0:
        sp_drop_max = 0.0
    if sp_drop_max > 5.0:
        sp_drop_max = 5.0

    price_now, price_min, price_max = _get_price_info()
    price_factor = 1.0
    price_norm = None

    if (price_now is not None) and (price_min is not None) and (price_max is not None):
        diff_range = price_max - price_min
        if diff_range > 0.0:
            price_norm = (price_now - price_min) / diff_range
            if price_norm < 0.0:
                price_norm = 0.0
            if price_norm > 1.0:
                price_norm = 1.0
            price_factor = 1.0 - price_sens * price_norm
            if price_factor < pf_min:
                price_factor = pf_min
            if price_factor > pf_max:
                price_factor = pf_max

    # ---- HINNAN MUKAINEN SETPOINTIN ALEMMAKS NOSTO ----
    price_sp_delta = 0.0
    if price_norm is not None:
        # mitä korkeampi hinta ja mitä suurempi price_sens, sitä enemmän pudotetaan SP:tä
        price_sp_delta = sp_drop_max * price_sens * price_norm

    sp_eff = sp_base - price_sp_delta

    # ---- RLS-OPPIMINEN ----
    err         = sp_eff - Tin
    demand_norm = _clip(prev_eff / 100.0, 0.0, 1.0)
    x = [1.0, err, demand_norm, (Tout_raw - Tin)]
    y = rate

    allow_learning = (not defrosting) and (not in_cooldown)

    if allow_learning:
        theta, P = _rls_update(theta, P, x, y)
        _theta_by_ctx[ctx] = theta
        _P_by_ctx[ctx] = P
        _save_params_to_store()
    else:
        _theta_by_ctx[ctx] = theta
        _P_by_ctx[ctx] = P
        log.debug(
            "Daikin ML: learning paused (defrost=%s, cooldown=%s, ctx=%s)",
            defrosting, in_cooldown, ctx,
        )

    # ---- ML-OPTIMI (ilman hintaa) ----
    Tout_future = _avg_future_outdoor()
    Tout_eff    = 0.5 * Tout_raw + 0.5 * Tout_future
    dTin_target = _clip(KAPPA * err / HORIZON_H, -2.0, 2.0)

    num        = dTin_target - (theta[0] + theta[1] * err + theta[3] * (Tout_eff - Tin))
    denom_raw  = theta[2]
    denom_sign = 1.0 if denom_raw >= 0 else -1.0
    denom_mag  = abs(denom_raw)
    if denom_mag < 0.5:
        denom_mag = 0.5

    dem_opt = _clip((num / (denom_mag * denom_sign)) * 100.0, 0.0, 100.0)

    # ---- PÖRSSI-SKAALAUS ML-OPTIMILLE ----
    dem_price_raw = dem_opt * price_factor

    # ---- DEAD BAND / HINTA → dem_target (ennen steppejä/cappeja) ----
    if abs(err) <= deadband:
        dem_target = prev_eff            # ei säätöä deadbandin sisällä
    else:
        dem_target = dem_price_raw       # ML-optimi * pörssikerroin

    dem_target_pre_clip = dem_target

    # Monotoninen ehto: jos ollaan kylmällä puolella, demand ei saa laskea
    if err > deadband and dem_target < prev_eff:
        dem_target = prev_eff

    # delta ja step/cooldown -rajat
    delta = dem_target - prev_eff
    if delta > 0:
        if in_cooldown:
            up_limit = COOLDOWN_STEP_UP
        else:
            up_limit = step_limit
        if delta > up_limit:
            dem_target = prev_eff + up_limit
    elif delta < 0:
        down_limit = step_limit
        if -delta > down_limit:
            dem_target = prev_eff - down_limit

    # ---- KLIPPI + ERILLINEN HINTA-CAP ----
    dem_clip = _clip(dem_target, MIN_DEM, band_upper)

    price_cap_demand = None
    if price_norm is not None:
        # price_sens 0 → ei rajoitusta, 1 → kalleimmillaan cap = MIN_DEM
        price_cap_demand = MIN_DEM + (band_upper - MIN_DEM) * (1.0 - price_sens * price_norm)
        if price_cap_demand < band_upper and dem_clip > price_cap_demand:
            dem_clip = price_cap_demand
        dem_clip = _clip(dem_clip, MIN_DEM, band_upper)

    option = _snap_to_select(dem_clip, 0)

    # ---- PÄIVITÄ ML LEARNED DEMAND -SENSORI ----
    learn_value = dem_clip   # mitä ML+hintalogiikka lopulta haluaa (ennen select-pykäliä)

    try:
        if price_now is None:
            price_now_val = None
        else:
            price_now_val = price_now

        state.set(
            LEARNED_SENSOR,
            value=round(learn_value, 1),
            ctx=ctx,
            outdoor_bucket=Tout_bucket,
            outdoor=round(Tout_raw, 1),
            setpoint_base=round(sp_base, 2),
            setpoint_effective=round(sp_eff, 2),
            setpoint_drop=round(price_sp_delta, 2),
            indoor=round(Tin, 2),
            defrosting=defrosting,
            cooldown=in_cooldown,
            price_now=price_now_val,
            price_min=price_min,
            price_max=price_max,
            price_factor=round(price_factor, 3),
            price_sensitivity=round(price_sens, 3),
            price_factor_min=round(pf_min, 3),
            price_factor_max=round(pf_max, 3),
            price_norm=price_norm,
            price_cap_demand=round(price_cap_demand, 1) if price_cap_demand is not None else None,
            sp_drop_max=round(sp_drop_max, 2),
            dem_opt_raw=round(dem_opt, 1),
            dem_price_raw=round(dem_price_raw, 1),
            dem_target_pre_clip=round(dem_target_pre_clip, 1),
        )
    except Exception as e:
        log.error("Daikin ML: failed to update learned demand sensor: %s", e)

    # Jos select-arvo jäisi cappen yli, clampataan vielä
    try:
        opt_num = float(str(option).replace('%', ''))
    except Exception:
        opt_num = 999.0
    if opt_num > band_upper:
        option = _snap_to_select(band_upper, -1)
        log.info(
            "Daikin ML: post-cap clamp -> %s (upper=%.0f%%, ctx=%s, Tout_bucket=%d)",
            option, band_upper, ctx, Tout_bucket,
        )

    if option and option != prev_str:
        select.select_option(entity_id=SELECT, option=option)

    theta_str = "[" + ", ".join([str(round(float(v), 4)) for v in theta]) + "]"
    cool_str  = "ACTIVE" if in_cooldown else "off"
    icing_str = "ON" if in_icing_band else "off"

    if price_now is None:
        price_now_log = -1.0
    else:
        price_now_log = price_now

    log.info(
        "Daikin ML: ctx=%s | Tin=%.2f°C, Tout=%.2f°C (bucket=%d, →%.1f°C), "
        "DEFROST=%s, cooldown=%s, icing_band=%s, band_upper=%.0f%%, theta=%s | "
        "SP_base=%.2f, SP_eff=%.2f, SP_drop=%.2f, step_limit=%.1f, deadband=%.2f | "
        "price=%.3f, pf=%.2f (sens=%.2f, fmin=%.2f, fmax=%.2f, spdrop_max=%.2f) | "
        "price_cap≈%s | "
        "prev=%s → opt≈%.0f%% → price≈%.0f%% → target≈%.0f%% → clip=%.0f%% → select=%s",
        ctx, Tin, Tout_raw, Tout_bucket, Tout_future,
        str(defrosting), cool_str, icing_str, band_upper, theta_str,
        sp_base, sp_eff, price_sp_delta, step_limit, deadband,
        price_now_log, price_factor, price_sens, pf_min, pf_max, sp_drop_max,
        str(round(price_cap_demand, 1)) if price_cap_demand is not None else "none",
        prev_str, dem_opt, dem_price_raw, dem_target, dem_clip, option,
    )

# ---------- CAP-VARTIJA ----------
@state_trigger(f"{SELECT}")
def daikin_ml_cap_guard(value=None, **kwargs):
    Tout_raw = _safe_float(state.get(OUTDOOR), None)
    if Tout_raw is None or not isfinite(Tout_raw):
        return
    Tout_bucket = int(round(Tout_raw))
    if Tout_bucket <= -5:
        global_upper = MAX_DEM
    else:
        global_upper = GLOBAL_MILD_MAX

    icing_cap = _safe_float(state.get(ICING_CAP_HELPER), ICING_BAND_CAP_DEFAULT)
    icing_cap = _clip(icing_cap, MIN_DEM, MAX_DEM)

    in_icing_band = (Tout_bucket >= ICING_BAND_MIN and Tout_bucket <= ICING_BAND_MAX)
    if in_icing_band:
        band_upper = icing_cap if icing_cap < global_upper else global_upper
    else:
        band_upper = global_upper

    curr_str = state.get(SELECT) or ""
    try:
        curr = float(str(curr_str).replace('%', ''))
    except Exception:
        return
    if curr > band_upper:
        option_cap = _snap_to_select(band_upper, -1)
        if option_cap and option_cap != curr_str:
            select.select_option(entity_id=SELECT, option=option_cap)
            log.info(
                "Daikin ML: CAP GUARD: %s -> %s (upper=%.0f%%, Tout_bucket=%d)",
                curr_str, option_cap, band_upper, Tout_bucket,
            )

# ---------- RESET-PALVELU ----------
@service
def daikin_ml_reset():
    """Nollaa kaikki ML-opit ja aloita alusta."""
    global _theta_by_ctx, _P_by_ctx, _params_loaded
    _theta_by_ctx = {}
    _P_by_ctx = {}
    _params_loaded = False
    try:
        state.set(
            STORE_ENTITY,
            value=time.time(),
            theta_by_ctx={},
        )
        log.warning("Daikin ML: ALL LEARNED PARAMETERS RESET (theta_by_ctx cleared)")
    except Exception as e:
        log.error("Daikin ML: error resetting params in %s: %s", STORE_ENTITY, e)

# ---------- Käsin ajettava askel ----------
@service
def daikin_ml_step():
    """Aja ohjain kerran (voi kutsua HA-automaatioista)."""
    try:
        daikin_ml_controller()
    except Exception as e:
        log.error("Daikin ML step error: %s", e)
