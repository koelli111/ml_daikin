# pyscript/daikin_ml_multi.py
# Online-RLS (oppiva) ohjain Daikin demand-selectille, MULTI-DAIKIN -tuella.
#
# ADDITIONS:
# - Helper to switch learning on/off:
#     * input_boolean.daikin_ml_learning_enabled (global)
#     * optional per-unit override: input_boolean.daikin1_ml_learning_enabled, input_boolean.daikin2_ml_learning_enabled, ...
#   Logic:
#     - If per-unit helper exists, it takes precedence for that unit.
#     - Otherwise global helper is used.
#
# FIXES:
# - NaN/inf-suojaukset: rate / x / y / theta / dem_opt ei voi enää korruptoitua tai mennä NaN:ksi
# - _clip on NaN-safe
# - _rls_update on NaN-safe (hylkää päivityksen jos jokin menee epäkelvoksi)
# - tallennus/lataus ohittaa epäkelvot (NaN/inf) theta-arvot
# - learned demand sensor ei koskaan saa NaN:ia (fallback prev_eff)
#
# MUOKKAA: DAIKINS-listaan omat entityt (yksi dict per Daikin)

import time
from math import isfinite

# ============================================================
# 0) LEARNING ENABLE HELPERS
# ============================================================
# Create these in Home Assistant:
# - input_boolean.daikin_ml_learning_enabled   (global master switch)
# Optionally per unit (takes precedence if exists):
# - input_boolean.daikin1_ml_learning_enabled
# - input_boolean.daikin2_ml_learning_enabled
LEARNING_ENABLED_HELPER_GLOBAL = "input_boolean.daikin_ml_learning_enabled"

# ============================================================
# 1) LAITEKONFIGURAATIOT (LISÄÄ TÄHÄN UUSIA DAIKINEita)
# ============================================================
DAIKINS = [
    {
        "name": "daikin1",

        "INDOOR": "sensor.living_room_lampotila",
        "INDOOR_RATE": "sensor.sisalampotila_aula",
        "OUTDOOR": "sensor.iv_tulo_lampotila",
        "WEATHER": "weather.koti",

        "SELECT": "select.faikin_demand_control",
        "LIQUID": "sensor.faikin_liquid",

        "SP_HELPER": "input_number.daikin_setpoint",
        "STEP_LIMIT_HELPER": "input_number.daikin_step_limit",
        "DEADBAND_HELPER": "input_number.daikin_deadband",
        "ICING_CAP_HELPER": "input_number.daikin_icing_cap",

        # Persistent store per laite (suositus: eri entity per daikin)
        "STORE_ENTITY": "pyscript.daikin1_ml_params",

        # Learned-demand sensori per laite
        "LEARNED_SENSOR": "sensor.daikin1_ml_learned_demand",

        # Optional per-unit learning switch (if you create it)
        "LEARNING_ENABLED_HELPER": "input_boolean.daikin1_ml_learning_enabled",
    },

    # Esimerkki toisesta laitteesta (muuta entityt):
    # {
    #     "name": "daikin2",
    #     "INDOOR": "sensor.toinen_sisalampo",
    #     "INDOOR_RATE": "sensor.toinen_sisalampo_rate",
    #     "OUTDOOR": "sensor.toinen_ulkolampo",
    #     "WEATHER": "weather.koti",
    #     "SELECT": "select.toinen_daikin_demand_control",
    #     "LIQUID": "sensor.toinen_daikin_liquid",
    #     "SP_HELPER": "input_number.daikin2_setpoint",
    #     "STEP_LIMIT_HELPER": "input_number.daikin2_step_limit",
    #     "DEADBAND_HELPER": "input_number.daikin2_deadband",
    #     "ICING_CAP_HELPER": "input_number.daikin2_icing_cap",
    #     "STORE_ENTITY": "pyscript.daikin2_ml_params",
    #     "LEARNED_SENSOR": "sensor.daikin2_ml_learned_demand",
    #     "LEARNING_ENABLED_HELPER": "input_boolean.daikin2_ml_learning_enabled",
    # },
]

# ============================================================
# 2) YHTEISET SÄÄTÖVAKIOT
# ============================================================
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


# ============================================================
# 3) APURIT
# ============================================================
def _context_key_for_outdoor(Tout: float) -> str:
    if not isfinite(Tout):
        return "nan"
    bucket = int(round(Tout))
    return str(bucket)

def _clip(v, lo, hi):
    # NaN-safe clip: if v is NaN/inf -> return lo (safe)
    try:
        if not isfinite(float(v)):
            return lo
    except Exception:
        return lo
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v

def _all_finite(seq):
    for v in seq:
        try:
            if not isfinite(float(v)):
                return False
        except Exception:
            return False
    return True

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
    # Guard: if bad inputs, do nothing
    if (not _all_finite(theta)) or (not _all_finite(x)):
        return theta, P
    try:
        y_f = float(y)
    except Exception:
        return theta, P
    if not isfinite(y_f):
        return theta, P

    Px = _matvec(P, x)
    denom = lam + _dot(x, Px)

    # Guard denom
    if (not isfinite(denom)) or denom <= 1e-12:
        return theta, P

    K = [v / denom for v in Px]

    err_est = y_f - _dot(x, theta)
    if not isfinite(err_est):
        return theta, P

    new_theta = [
        theta[0] + K[0] * err_est,
        theta[1] + K[1] * err_est,
        theta[2] + K[2] * err_est,
        theta[3] + K[3] * err_est,
    ]
    if not _all_finite(new_theta):
        return theta, P

    xTP = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            xTP[i][j] = x[i] * Px[j]

    newP = _matscale(_matsub(P, xTP), 1.0 / lam)

    # Guard P too
    for i in range(4):
        for j in range(4):
            if not isfinite(newP[i][j]):
                return theta, P

    return new_theta, newP

def _select_options_nums(select_entity):
    attrs = state.getattr(select_entity) or {}
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

def _with_pct(select_entity):
    attrs = state.getattr(select_entity) or {}
    opts = attrs.get("options") or []
    return ('%' in (opts[0] if opts else ''))

def _snap_to_select(select_entity, value, direction):
    nums, has_pct = _select_options_nums(select_entity)
    picked = None
    if len(nums) == 0:
        picked = round(value)
        out = str(int(picked)) + ('%' if (has_pct or _with_pct(select_entity)) else '')
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

def _auto_tune_helpers(theta, unit_name, ctx, step_limit_current, deadband_current):
    try:
        gain = abs(float(theta[2]))
    except Exception:
        gain = 1.0
    if not isfinite(gain):
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
            "Daikin ML AUTO-TUNE (%s, ctx=%s): gain=%.3f -> step_limit=%.1f (was %.1f), deadband=%.2f (was %.2f)",
            unit_name, ctx, gain,
            step_auto_rounded, step_limit_current,
            db_auto_rounded,   deadband_current,
        )
    return step_auto_rounded, db_auto_rounded

def _avg_future_outdoor(weather_entity, outdoor_entity):
    attrs = state.getattr(weather_entity) or {}
    fc = attrs.get("forecast") or []
    temps = []
    idx = 0
    for f in fc:
        if idx >= FORECAST_H:
            break
        t = f.get("temperature")
        if t is not None:
            try:
                tv = float(t)
                if isfinite(tv):
                    temps.append(tv)
                    idx += 1
            except Exception:
                pass
    if len(temps) > 0:
        s = 0.0
        for v in temps:
            s += v
        return s / float(len(temps))
    try:
        v = float(state.get(outdoor_entity) or 0.0)
        return v if isfinite(v) else 0.0
    except Exception:
        return 0.0

def _read_bool(entity_id, default=True):
    """Read HA boolean-ish state robustly."""
    try:
        s = state.get(entity_id)
    except Exception:
        return default
    if s is None:
        return default
    ss = str(s).strip().lower()
    if ss in ("on", "true", "1", "yes", "enabled"):
        return True
    if ss in ("off", "false", "0", "no", "disabled"):
        return False
    return default

def _learning_enabled_for_unit(unit):
    """
    Learning enabled logic:
    - if per-unit helper exists (state.get != None) -> use it
    - else use global helper (if exists), default True if not found
    """
    unit_helper = unit.get("LEARNING_ENABLED_HELPER")
    if unit_helper:
        try:
            v = state.get(unit_helper)
            if v is not None:
                return _read_bool(unit_helper, default=True)
        except Exception:
            pass
    # fallback global
    try:
        gv = state.get(LEARNING_ENABLED_HELPER_GLOBAL)
        if gv is None:
            return True
        return _read_bool(LEARNING_ENABLED_HELPER_GLOBAL, default=True)
    except Exception:
        return True


# ============================================================
# 4) PER-LAITE TILA (THETA/P + DEFROST-COOLDOWN)
# ============================================================
_theta_by_unit_ctx = {}   # unit -> ctx -> theta[4]
_P_by_unit_ctx     = {}   # unit -> ctx -> P[4x4]
_params_loaded     = {}   # unit -> bool

_last_defrosting   = {}   # unit -> bool/None
_cooldown_until    = {}   # unit -> epoch float


def _persist_all_stores():
    for u in DAIKINS:
        try:
            state.persist(u["STORE_ENTITY"])
        except Exception as e:
            log.error("Daikin ML (%s): state.persist failed for %s: %s", u["name"], u["STORE_ENTITY"], e)

def _init_unit_if_needed(unit_name):
    if unit_name not in _theta_by_unit_ctx:
        _theta_by_unit_ctx[unit_name] = {}
    if unit_name not in _P_by_unit_ctx:
        _P_by_unit_ctx[unit_name] = {}
    if unit_name not in _params_loaded:
        _params_loaded[unit_name] = False
    if unit_name not in _last_defrosting:
        _last_defrosting[unit_name] = None
    if unit_name not in _cooldown_until:
        _cooldown_until[unit_name] = 0.0

def _load_params_from_store(unit):
    unit_name = unit["name"]
    store = unit["STORE_ENTITY"]
    attrs = state.getattr(store) or {}
    tb = attrs.get("theta_by_ctx")
    new = {}
    if isinstance(tb, dict):
        for key, val in tb.items():
            if isinstance(val, (list, tuple)) and len(val) == 4:
                try:
                    th = [float(val[0]), float(val[1]), float(val[2]), float(val[3])]
                    if _all_finite(th):
                        new[str(key)] = th
                except Exception:
                    continue
    if new:
        _theta_by_unit_ctx[unit_name] = new
        log.info("Daikin ML (%s): loaded %d contexts from %s", unit_name, len(new), store)
    else:
        log.info("Daikin ML (%s): no stored thetas in %s, starting fresh", unit_name, store)

def _save_params_to_store(unit):
    unit_name = unit["name"]
    store = unit["STORE_ENTITY"]
    clean = {}
    for key, th in (_theta_by_unit_ctx.get(unit_name) or {}).items():
        if not isinstance(th, (list, tuple)) or len(th) != 4:
            continue
        if not _all_finite(th):
            continue
        clean[str(key)] = [
            round(float(th[0]), 4),
            round(float(th[1]), 4),
            round(float(th[2]), 4),
            round(float(th[3]), 4),
        ]
    try:
        state.set(
            store,
            value=time.time(),
            theta_by_ctx=clean,
        )
    except Exception as e:
        log.error("Daikin ML (%s): error saving thetas to %s: %s", unit_name, store, e)

def _init_context_params_if_needed(unit):
    unit_name = unit["name"]
    _init_unit_if_needed(unit_name)
    if _params_loaded.get(unit_name):
        return

    _theta_by_unit_ctx[unit_name] = {}
    _P_by_unit_ctx[unit_name] = {}

    _load_params_from_store(unit)

    # P-matriisit muistiin vain runtimeen (ei tallenneta storeen)
    for key in (_theta_by_unit_ctx.get(unit_name) or {}).keys():
        _P_by_unit_ctx[unit_name][str(key)] = [[P0, 0, 0, 0],
                                               [0, P0, 0, 0],
                                               [0, 0, P0, 0],
                                               [0, 0, 0, P0]]
    _params_loaded[unit_name] = True


# ============================================================
# 5) STARTUP + TRIGGERIT
# ============================================================
_persist_all_stores()

_TRIGGER_ENTITIES = []
for u in DAIKINS:
    for k in ("INDOOR", "OUTDOOR", "INDOOR_RATE"):
        ent = u.get(k)
        if ent:
            _TRIGGER_ENTITIES.append(ent)

@time_trigger("startup")
def _ml_startup_ok():
    for u in DAIKINS:
        _init_context_params_if_needed(u)
        try:
            state.set(
                u["LEARNED_SENSOR"],
                value=0.0,
                unit=u["name"],
                ctx="init",
                note="raw ML demand (dem_opt) before caps/step/cooldown"
            )
        except Exception as e:
            log.error("Daikin ML (%s): failed to init learned demand sensor: %s", u["name"], e)

    log.info("Daikin ML MULTI: startup loaded for %d unit(s)", len(DAIKINS))

@time_trigger("cron(*/6 * * * *)")
@state_trigger(*_TRIGGER_ENTITIES)
def daikin_ml_controller(**kwargs):
    for u in DAIKINS:
        try:
            _run_one_unit(u)
        except Exception as e:
            log.error("Daikin ML (%s): controller error: %s", u.get("name", "?"), e)


# ============================================================
# 6) VARSINAINEN LOGIIKKA PER LAITE
# ============================================================
def _run_one_unit(u):
    unit_name = u["name"]
    _init_context_params_if_needed(u)

    INDOOR = u["INDOOR"]
    INDOOR_RATE = u["INDOOR_RATE"]
    OUTDOOR = u["OUTDOOR"]
    WEATHER = u["WEATHER"]
    SELECT = u["SELECT"]
    LIQUID = u["LIQUID"]

    SP_HELPER = u["SP_HELPER"]
    STEP_LIMIT_HELPER = u["STEP_LIMIT_HELPER"]
    DEADBAND_HELPER = u["DEADBAND_HELPER"]
    ICING_CAP_HELPER = u["ICING_CAP_HELPER"]
    LEARNED_SENSOR = u["LEARNED_SENSOR"]

    # Learning helper state
    learning_enabled = _learning_enabled_for_unit(u)

    try:
        Tin = float(state.get(INDOOR))
    except Exception:
        Tin = float("nan")
    try:
        Tout_raw = float(state.get(OUTDOOR))
    except Exception:
        Tout_raw = float("nan")

    # RATE FIX: float("nan") does not raise -> must isfinite-check
    try:
        rate = float(state.get(INDOOR_RATE) or 0.0)
    except Exception:
        rate = 0.0
    if not isfinite(rate):
        log.warning(
            "Daikin ML (%s): INDOOR_RATE not finite (%s) -> forcing 0.0 and skipping learning this tick",
            unit_name, str(state.get(INDOOR_RATE))
        )
        rate = 0.0
        rate_bad = True
    else:
        rate_bad = False

    sp = float(state.get(SP_HELPER) or 22.5)

    step_limit_current = float(state.get(STEP_LIMIT_HELPER) or 10.0)
    deadband_current   = float(state.get(DEADBAND_HELPER) or 0.1)

    prev_str = state.get(SELECT) or ""
    try:
        prev = float(prev_str.replace('%', ''))
    except Exception:
        prev = 60.0

    try:
        liquid = float(state.get(LIQUID) or 100.0)
    except Exception:
        liquid = 100.0
    if not isfinite(liquid):
        liquid = 100.0

    defrosting = (liquid < 20.0)

    now = time.time()
    if _last_defrosting[unit_name] is None:
        _last_defrosting[unit_name] = defrosting
    if (_last_defrosting[unit_name] is True) and (defrosting is False):
        _cooldown_until[unit_name] = now + COOLDOWN_MINUTES * 60.0
        log.info("Daikin ML (%s): defrost ended -> cooldown active for %d min", unit_name, COOLDOWN_MINUTES)
    _last_defrosting[unit_name] = defrosting
    in_cooldown = (now < _cooldown_until[unit_name])

    if not (isfinite(Tin) and isfinite(Tout_raw)):
        log.info("Daikin ML (%s): sensors not ready; Tin=%s Tout=%s", unit_name, state.get(INDOOR), state.get(OUTDOOR))
        return

    Tout_bucket = int(round(Tout_raw))
    ctx = _context_key_for_outdoor(Tout_raw)

    if ctx not in _theta_by_unit_ctx[unit_name]:
        _theta_by_unit_ctx[unit_name][ctx] = [0.0, 0.0, 5.0, 0.0]
    if ctx not in _P_by_unit_ctx[unit_name]:
        _P_by_unit_ctx[unit_name][ctx] = [[P0, 0, 0, 0],
                                          [0, P0, 0, 0],
                                          [0, 0, P0, 0],
                                          [0, 0, 0, P0]]

    theta = _theta_by_unit_ctx[unit_name][ctx]
    P     = _P_by_unit_ctx[unit_name][ctx]

    # theta guard: if somehow corrupted, reset to defaults (per ctx)
    if not _all_finite(theta):
        log.error("Daikin ML (%s): theta not finite for ctx=%s -> resetting theta/P", unit_name, ctx)
        theta = [0.0, 0.0, 5.0, 0.0]
        P = [[P0, 0, 0, 0],
             [0, P0, 0, 0],
             [0, 0, P0, 0],
             [0, 0, 0, P0]]
        _theta_by_unit_ctx[unit_name][ctx] = theta
        _P_by_unit_ctx[unit_name][ctx] = P
        _save_params_to_store(u)

    step_limit, deadband = _auto_tune_helpers(theta, unit_name, ctx, step_limit_current, deadband_current)

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
    if not isfinite(icing_cap):
        icing_cap = ICING_BAND_CAP_DEFAULT
    icing_cap = _clip(icing_cap, MIN_DEM, MAX_DEM)

    in_icing_band = (Tout_bucket >= ICING_BAND_MIN and Tout_bucket <= ICING_BAND_MAX)
    band_upper = min(global_upper, icing_cap) if in_icing_band else global_upper

    # jos edellinen yli band_upper, tiputetaan heti
    if prev > band_upper:
        option_cap = _snap_to_select(SELECT, band_upper, -1)
        if option_cap and option_cap != prev_str:
            select.select_option(entity_id=SELECT, option=option_cap)
            log.info(
                "Daikin ML (%s): CAP ENFORCED: prev=%s -> %s (upper=%.0f%%, ctx=%s, Tout_bucket=%d)",
                unit_name, prev_str, option_cap, band_upper, ctx, Tout_bucket,
            )
        return

    prev_eff = prev if prev <= band_upper else band_upper

    err         = sp - Tin
    demand_norm = _clip(prev_eff / 100.0, 0.0, 1.0)
    x = [1.0, err, demand_norm, (Tout_raw - Tin)]
    y = rate

    # LEARNING ENABLE LOGIC INCLUDED HERE
    allow_learning = (
        learning_enabled
        and (not defrosting)
        and (not in_cooldown)
        and (not rate_bad)
    )

    if allow_learning:
        theta_prev = theta[:]
        P_prev = [row[:] for row in P]

        theta_new, P_new = _rls_update(theta, P, x, y)

        if _all_finite(theta_new):
            theta = theta_new
            P = P_new
            _theta_by_unit_ctx[unit_name][ctx] = theta
            _P_by_unit_ctx[unit_name][ctx] = P
            _save_params_to_store(u)
        else:
            log.error("Daikin ML (%s): theta update produced non-finite; reverting", unit_name)
            theta = theta_prev
            P = P_prev
            _theta_by_unit_ctx[unit_name][ctx] = theta
            _P_by_unit_ctx[unit_name][ctx] = P
    else:
        _theta_by_unit_ctx[unit_name][ctx] = theta
        _P_by_unit_ctx[unit_name][ctx] = P
        log.debug(
            "Daikin ML (%s): learning paused (learning_enabled=%s, defrost=%s, cooldown=%s, rate_bad=%s, ctx=%s)",
            unit_name, str(learning_enabled), defrosting, in_cooldown, rate_bad, ctx,
        )

    Tout_future = _avg_future_outdoor(WEATHER, OUTDOOR)
    Tout_eff    = 0.5 * Tout_raw + 0.5 * Tout_future
    dTin_target = _clip(KAPPA * err / HORIZON_H, -2.0, 2.0)

    num = dTin_target - (theta[0] + theta[1] * err + theta[3] * (Tout_eff - Tin))

    denom_raw = theta[2]
    if (not isfinite(denom_raw)) or abs(denom_raw) < 1e-6:
        denom_raw = 5.0

    denom_sign = 1.0 if denom_raw >= 0 else -1.0
    denom_mag  = abs(denom_raw)
    if denom_mag < 0.5:
        denom_mag = 0.5

    dem_opt = _clip((num / (denom_mag * denom_sign)) * 100.0, 0.0, 100.0)

    if not isfinite(dem_opt):
        log.error(
            "Daikin ML (%s): dem_opt became non-finite (ctx=%s). Falling back to prev_eff=%.1f",
            unit_name, ctx, prev_eff
        )
        dem_opt = prev_eff

    # Learned demand -sensori (raaka ML-optimi)
    try:
        state.set(
            LEARNED_SENSOR,
            value=round(float(dem_opt), 1),
            unit=unit_name,
            ctx=ctx,
            outdoor_bucket=Tout_bucket,
            outdoor=round(Tout_raw, 1),
            setpoint=round(sp, 2),
            indoor=round(Tin, 2),
            defrosting=defrosting,
            cooldown=in_cooldown,
            rate=round(rate, 4),
            learning_enabled=bool(learning_enabled),
            learning_allowed=bool(allow_learning),
        )
    except Exception as e:
        log.error("Daikin ML (%s): failed to update learned demand sensor: %s", unit_name, e)

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
        up_limit = COOLDOWN_STEP_UP if in_cooldown else step_limit
        if delta > up_limit:
            dem_target = prev_eff + up_limit
    elif delta < 0:
        down_limit = step_limit
        if -delta > down_limit:
            dem_target = prev_eff - down_limit

    dem_clip = _clip(dem_target, MIN_DEM, band_upper)
    option = _snap_to_select(SELECT, dem_clip, 0)

    try:
        opt_num = float(str(option).replace('%', ''))
    except Exception:
        opt_num = 999.0
    if opt_num > band_upper:
        option = _snap_to_select(SELECT, band_upper, -1)
        log.info(
            "Daikin ML (%s): post-cap clamp -> %s (upper=%.0f%%, ctx=%s, Tout_bucket=%d)",
            unit_name, option, band_upper, ctx, Tout_bucket,
        )

    if option and option != prev_str:
        select.select_option(entity_id=SELECT, option=option)

    theta_str = "[" + ", ".join([str(round(float(v), 4)) for v in theta]) + "]"
    cool_str  = "ACTIVE" if in_cooldown else "off"
    icing_str = "ON" if in_icing_band else "off"
    log.info(
        "Daikin ML (%s): ctx=%s | Tin=%.2f°C, Tout=%.2f°C (bucket=%d, →%.1f°C), "
        "DEFROST=%s, cooldown=%s, learning_enabled=%s, learning_allowed=%s, rate_bad=%s, "
        "icing_band=%s, band_upper=%.0f%%, theta=%s | "
        "SP=%.2f, step_limit=%.1f, deadband=%.2f | "
        "prev=%s → opt≈%.0f%% → target≈%.0f%% → clip=%.0f%% → select=%s",
        unit_name, ctx, Tin, Tout_raw, Tout_bucket, Tout_future,
        str(defrosting), cool_str, str(learning_enabled), str(allow_learning), str(rate_bad),
        icing_str, band_upper, theta_str,
        sp, step_limit, deadband,
        prev_str, dem_opt, dem_target, dem_clip, option,
    )


# ============================================================
# 7) PALVELUT
# ============================================================
@service
def daikin_ml_step():
    """Aja ohjain kerran (kaikille laitteille)."""
    daikin_ml_controller()

@service
def daikin_ml_reset():
    """Nollaa kaikkien laitteiden ML-opit ja aloita alusta."""
    global _theta_by_unit_ctx, _P_by_unit_ctx, _params_loaded, _last_defrosting, _cooldown_until

    for u in DAIKINS:
        unit_name = u["name"]
        _init_unit_if_needed(unit_name)

        old_cnt = len(_theta_by_unit_ctx.get(unit_name) or {})
        _theta_by_unit_ctx[unit_name] = {}
        _P_by_unit_ctx[unit_name] = {}
        _params_loaded[unit_name] = False
        _last_defrosting[unit_name] = None
        _cooldown_until[unit_name] = 0.0

        try:
            state.set(
                u["STORE_ENTITY"],
                value=time.time(),
                theta_by_ctx={},
            )
            log.warning(
                "Daikin ML RESET (%s): cleared %d learned contexts, theta_by_ctx now empty in %s",
                unit_name, old_cnt, u["STORE_ENTITY"],
            )
        except Exception as e:
            log.error("Daikin ML RESET (%s): error resetting params in %s: %s", unit_name, u["STORE_ENTITY"], e)

@service
def daikin_ml_persist():
    """Persistoi store-entiteetit (varmistus)."""
    _persist_all_stores()

@service
def daikin_ml_learning_enable():
    """Enable learning globally (input_boolean.daikin_ml_learning_enabled)."""
    try:
        input_boolean.turn_on(entity_id=LEARNING_ENABLED_HELPER_GLOBAL)
        log.info("Daikin ML: learning enabled (global) -> %s", LEARNING_ENABLED_HELPER_GLOBAL)
    except Exception as e:
        log.error("Daikin ML: failed to enable learning global: %s", e)

@service
def daikin_ml_learning_disable():
    """Disable learning globally (input_boolean.daikin_ml_learning_enabled)."""
    try:
        input_boolean.turn_off(entity_id=LEARNING_ENABLED_HELPER_GLOBAL)
        log.info("Daikin ML: learning disabled (global) -> %s", LEARNING_ENABLED_HELPER_GLOBAL)
    except Exception as e:
        log.error("Daikin ML: failed to disable learning global: %s", e)

@service
def daikin_ml_learning_toggle():
    """Toggle learning globally (input_boolean.daikin_ml_learning_enabled)."""
    try:
        cur = state.get(LEARNING_ENABLED_HELPER_GLOBAL)
        cur_on = (str(cur).strip().lower() == "on")
        if cur_on:
            input_boolean.turn_off(entity_id=LEARNING_ENABLED_HELPER_GLOBAL)
            log.info("Daikin ML: learning toggled OFF (global)")
        else:
            input_boolean.turn_on(entity_id=LEARNING_ENABLED_HELPER_GLOBAL)
            log.info("Daikin ML: learning toggled ON (global)")
    except Exception as e:
        log.error("Daikin ML: failed to toggle learning global: %s", e)


# ============================================================
# 8) STORE PERSIST (optional helper)
# ============================================================
_persist_all_stores()
