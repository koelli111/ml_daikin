# pyscript/daikin_ml.py
# Online-RLS (oppiva) ohjain Daikin demand-selectille.
# Ominaisuudet:
# - min 30 %, max 100 %
# - 6h ulkoennusteen keskiarvo feed-forwardina
# - pakotettu suunta-askel (aina vähintään step_limit kohti setpointtia deadbandin ulkopuolella)
# - deadband, step-limit, monotonisuus ja selectin pykälöinti
# - oppiminen (RLS) jäädytetään defrostin aikana (sensor.faikin_liquid < 20)
# - parametrisäilö input_text.daikin_rls_params (vain theta tallennetaan, ei P)
#
# VAIHDA NÄMÄ ENTITYT:
#   INDOOR  = "sensor.home_temp"                 # sisälämpötila
#   OUTDOOR = "sensor.outdoor_temp"              # ulkolämpötila
#   WEATHER = "weather.home"                     # weather.* jossa 'forecast'
#   SELECT  = "select.faikin_demand_control"     # Daikin demand-select
#   LIQUID  = "sensor.faikin_liquid"             # defrost-indikaattori (alle 20 = defrost)
#   INDOOR_RATE, SP_HELPER, STEP_LIMIT_HELPER, DEADBAND_HELPER, PARAMS_TXT: ks. alta

import json
import time
from math import isfinite

# --------- ENTITYT (VAIHDA OMIKSI) ----------
INDOOR = "sensor.living_room_lampotila"                 # <-- sisälämpötila
INDOOR_RATE = "sensor.sisalampotila_aula"  # derivative-sensori (°C/h)
OUTDOOR = "sensor.iv_tulo_lampotila"             # <-- ulkolämpötila
WEATHER = "weather.koti"                    # <-- weather.* josta forecast-attribuutti
SELECT = "select.faikin_demand_control"     # <-- Daikinin demand-select
LIQUID = "sensor.faikin_liquid"             # <-- defrost-indikaattori


SP_HELPER = "input_number.daikin_setpoint"
STEP_LIMIT_HELPER = "input_number.daikin_step_limit"
DEADBAND_HELPER = "input_number.daikin_deadband"

# UUSI: icing cap helper
ICING_CAP_HELPER = "input_number.daikin_icing_cap"

# ML-oppien pysyvä storage
STORE_ENTITY = "pyscript.daikin_ml_params"

# Learned demand -sensori
LEARNED_SENSOR = "sensor.daikin_ml_learned_demand"

# ---------- SÄÄTÖNUPIT ----------
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
                    th = [float(val[0]), float(val[1]), float(val[2]), float(val[3])]
                    new[str(key)] = th
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
    Px = _matvec(P, x)
    denom = lam + _dot(x, Px)
    if denom == 0:
        denom = 1e-6
    K = [v / denom for v in Px]
    err_est = y - _dot(x, theta)
    theta = [theta[0] + K[0] * err_est,
             theta[1] + K[1] * err_est,
             theta[2] + K[2] * err_est,
             theta[3] + K[3] * err_est]
    xTP = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            xTP[i][j] = x[i] * Px[j]
    P = _matscale(_matsub(P, xTP), 1.0 / lam)
    return theta, P

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
        return float(state.get(OUTDOOR) or 0.0)
    except Exception:
        return 0.0

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

def _clip(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v

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
            note="raw ML demand (dem_opt) before caps/step/cooldown"
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

    try:
        Tin = float(state.get(INDOOR))
    except Exception:
        Tin = float("nan")
    try:
        Tout_raw = float(state.get(OUTDOOR))
    except Exception:
        Tout_raw = float("nan")
    try:
        rate = float(state.get(INDOOR_RATE) or 0.0)
    except Exception:
        rate = 0.0

    sp = float(state.get(SP_HELPER) or 22.5)

    step_limit_current = float(state.get(STEP_LIMIT_HELPER) or 10.0)
    deadband_current   = float(state.get(DEADBAND_HELPER) or 0.1)

    prev_str   = state.get(SELECT) or ""
    try:
        prev = float(prev_str.replace('%', ''))
    except Exception:
        prev = 60.0

    try:
        liquid = float(state.get(LIQUID) or 100.0)
    except Exception:
        liquid = 100.0
    defrosting = (liquid < 20.0)

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

    # Globaali max bucketin perusteella
    if Tout_bucket <= -5:
        global_upper = MAX_DEM
    else:
        global_upper = GLOBAL_MILD_MAX

    # Icing band
    try:
        icing_cap = float(state.get(ICING_CAP_HELPER) or ICING_BAND_CAP_DEFAULT)
    except Exception:
        icing_cap = ICING_BAND_CAP_DEFAULT
    icing_cap = _clip(icing_cap, MIN_DEM, MAX_DEM)

    in_icing_band = (Tout_bucket >= ICING_BAND_MIN and Tout_bucket <= ICING_BAND_MAX)

    if in_icing_band:
        band_upper = min(global_upper, icing_cap)
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

    err         = sp - Tin
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

    # --- PÄIVITETÄÄN LEARNED DEMAND -SENSORI (raaka ML-optimi) ---
    try:
        state.set(
            LEARNED_SENSOR,
            value=round(dem_opt, 1),
            ctx=ctx,
            outdoor_bucket=Tout_bucket,
            outdoor=round(Tout_raw, 1),
            setpoint=round(sp, 2),
            indoor=round(Tin, 2),
            defrosting=defrosting,
            cooldown=in_cooldown,
        )
    except Exception as e:
        log.error("Daikin ML: failed to update learned demand sensor: %s", e)

    # Deadband: pidä ennallaan
    if abs(err) <= deadband:
        dem_target = prev_eff
    else:
        dem_target = dem_opt

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

    dem_clip = _clip(dem_target, MIN_DEM, band_upper)
    option = _snap_to_select(dem_clip, 0)

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
    log.info(
        "Daikin ML: ctx=%s | Tin=%.2f°C, Tout=%.2f°C (bucket=%d, →%.1f°C), "
        "DEFROST=%s, cooldown=%s, "
        "icing_band=%s, band_upper=%.0f%%, theta=%s | "
        "SP=%.2f, step_limit=%.1f, deadband=%.2f | "
        "prev=%s → opt≈%.0f%% → target≈%.0f%% → clip=%.0f%% → select=%s",
        ctx, Tin, Tout_raw, Tout_bucket, Tout_future,
        str(defrosting), cool_str,
        icing_str, band_upper, theta_str,
        sp, step_limit, deadband,
        prev_str, dem_opt, dem_target, dem_clip, option,
    )

# ---------- CAP-VARTIJA ----------
@state_trigger(f"{SELECT}")
def daikin_ml_cap_guard(value=None, **kwargs):
    try:
        Tout_raw = float(state.get(OUTDOOR))
    except Exception:
        return
    Tout_bucket = int(round(Tout_raw))
    if Tout_bucket <= -5:
        global_upper = MAX_DEM
    else:
        global_upper = GLOBAL_MILD_MAX
    try:
        icing_cap = float(state.get(ICING_CAP_HELPER) or ICING_BAND_CAP_DEFAULT)
    except Exception:
        icing_cap = ICING_BAND_CAP_DEFAULT
    icing_cap = _clip(icing_cap, MIN_DEM, MAX_DEM)
    in_icing_band = (Tout_bucket >= ICING_BAND_MIN and Tout_bucket <= ICING_BAND_MAX)
    if in_icing_band:
        band_upper = min(global_upper, icing_cap)
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
    old_cnt = len(_theta_by_ctx) if isinstance(_theta_by_ctx, dict) else 0
    _theta_by_ctx = {}
    _P_by_ctx = {}
    _params_loaded = False
    try:
        state.set(
            STORE_ENTITY,
            value=time.time(),
            theta_by_ctx={},
        )
        log.warning(
            "Daikin ML RESET: cleared %d learned contexts, theta_by_ctx now empty in %s",
            old_cnt,
            STORE_ENTITY,
        )
    except Exception as e:
        log.error("Daikin ML RESET: error resetting params in %s: %s", STORE_ENTITY, e)

# ---------- Käsin ajettava askel ----------
@service
def daikin_ml_step():
    """Aja ohjain kerran (voi kutsua HA-automaatioista)."""
    try:
        daikin_ml_controller()
    except Exception as e:
        log.error("Daikin ML step error: %s", e)

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