# pyscript/daikin_ml_multi.py
# Online-RLS (oppiva) ohjain Daikin demand-selectille, MULTI-DAIKIN -tuella.
#
# FEATURES INCLUDED:
# - Multi-daikin support (DAIKINS list)
# - NaN/inf guards (rate/x/y/theta/dem_opt, store load/save ignores non-finite)
# - Learning enable switch (global + per-unit override)
# - OPTION A (improved):
#     * learning enabled  -> learned-demand sensor value follows dem_opt
#     * learning disabled -> learned-demand sensor value frozen, attributes updated every tick
# - Nord Pool 15-min price bias integration (per-unit input_number bias points)
#
# QUIET OUTDOOR HANDLING (NEW BEHAVIOR):
# - If QUIET_SWITCH is missing OR entity does not exist in HA:
#     * Quiet logic is skipped entirely
#     * Demand will jump normally from 95% to 100% via select when needed
# - If QUIET_SWITCH exists:
#     * "QUIET 100" intermediate level works:
#         - <=95   -> quiet OFF, select snapped
#         - 95-100 -> quiet ON,  select 100
#         - >=100  -> quiet OFF, select 100 (REAL 100), BUT gated as last resort
#
# REAL 100% "LAST RESORT" GATING:
# - Applies ONLY when quiet switch exists (because then "quiet 100" is available as intermediate)
# - If quiet switch does NOT exist: no gating, normal 100% behavior.
#

import time
from math import isfinite

# ============================================================
# 0) LEARNING ENABLE HELPERS
# ============================================================
LEARNING_ENABLED_HELPER_GLOBAL = "input_boolean.daikin_ml_learning_enabled"

# ============================================================
# 1) UNIT CONFIGS
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

        "STORE_ENTITY": "pyscript.daikin1_ml_params",
        "LEARNED_SENSOR": "sensor.daikin1_ml_learned_demand",

        "LEARNING_ENABLED_HELPER": "input_boolean.daikin1_ml_learning_enabled",

        # Quiet switch for "QUIET 100" virtual level (optional; if missing, logic is skipped)
        "QUIET_SWITCH": "switch.faikin_quiet_outdoor",

        # Nordpool price bias helper (demand points), set by nordpool_15m_bias.py
        "PRICE_BIAS_HELPER": "input_number.daikin1_price_bias_points",
    },

    # Example second unit without quiet:
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
    #     # no QUIET_SWITCH -> jump 95->100 normally
    #     "PRICE_BIAS_HELPER": "input_number.daikin2_price_bias_points",
    # },
]

# ============================================================
# 2) TUNING CONSTANTS
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

# REAL 100% last resort gating (ONLY when quiet switch exists)
LAST_RESORT_MIN_ERR_C = 0.5
LAST_RESORT_ERR_C = 1.5
LAST_RESORT_MINUTES = 30


# ============================================================
# 3) HELPERS
# ============================================================
def _context_key_for_outdoor(Tout: float) -> str:
    if not isfinite(Tout):
        return "nan"
    return str(int(round(Tout)))

def _clip(v, lo, hi):
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
    if len(nums) == 0:
        picked = round(value)
        return str(int(picked)) + ('%' if (has_pct or _with_pct(select_entity)) else '')

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
    gain = _clip(gain, 0.1, 10.0)

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
    unit_helper = unit.get("LEARNING_ENABLED_HELPER")
    if unit_helper:
        try:
            v = state.get(unit_helper)
            if v is not None:
                return _read_bool(unit_helper, default=True)
        except Exception:
            pass

    try:
        gv = state.get(LEARNING_ENABLED_HELPER_GLOBAL)
        if gv is None:
            return True
        return _read_bool(LEARNING_ENABLED_HELPER_GLOBAL, default=True)
    except Exception:
        return True

def _entity_exists(entity_id):
    if not entity_id:
        return False
    try:
        s = state.get(entity_id)
    except Exception:
        return False
    return s is not None and str(s) != "unknown"

def _quiet_is_on(quiet_switch):
    if not quiet_switch:
        return False
    if not _entity_exists(quiet_switch):
        return False
    try:
        return str(state.get(quiet_switch)).strip().lower() == "on"
    except Exception:
        return False

def _set_quiet(quiet_switch, turn_on, unit_name):
    if not quiet_switch:
        return
    if not _entity_exists(quiet_switch):
        return
    try:
        cur = _quiet_is_on(quiet_switch)
        if turn_on and (not cur):
            switch.turn_on(entity_id=quiet_switch)
            log.info("Daikin ML (%s): QUIET_SWITCH ON -> %s", unit_name, quiet_switch)
        elif (not turn_on) and cur:
            switch.turn_off(entity_id=quiet_switch)
            log.info("Daikin ML (%s): QUIET_SWITCH OFF -> %s", unit_name, quiet_switch)
    except Exception as e:
        log.error("Daikin ML (%s): failed to set QUIET_SWITCH %s: %s", unit_name, quiet_switch, e)

def _apply_demand(unit_name, select_entity, target_percent, prev_str):
    """Plain demand apply (no quiet logic)."""
    option = _snap_to_select(select_entity, float(target_percent), 0)
    if option and option != prev_str:
        select.select_option(entity_id=select_entity, option=option)
    return option

def _apply_demand_with_quiet(unit_name, select_entity, quiet_switch, target_percent, prev_str, allow_real_100):
    """
    Quiet-aware apply with REAL 100 gating.
    NOTE: this function should only be used if quiet_switch exists.
    """
    t = float(target_percent)

    if t >= 100.0:
        if allow_real_100:
            _set_quiet(quiet_switch, False, unit_name)
            option = _snap_to_select(select_entity, 100.0, 0)
        else:
            _set_quiet(quiet_switch, True, unit_name)
            option = _snap_to_select(select_entity, 100.0, 0)
    elif t > GLOBAL_MILD_MAX:
        _set_quiet(quiet_switch, True, unit_name)
        option = _snap_to_select(select_entity, 100.0, 0)
    else:
        _set_quiet(quiet_switch, False, unit_name)
        option = _snap_to_select(select_entity, t, 0)

    if option and option != prev_str:
        select.select_option(entity_id=select_entity, option=option)

    return option

def _read_price_bias_points(unit):
    ent = unit.get("PRICE_BIAS_HELPER")
    if not ent:
        return 0.0
    try:
        v = float(state.get(ent) or 0.0)
        if not isfinite(v):
            return 0.0
        return v
    except Exception:
        return 0.0


# ============================================================
# 4) STATE (per unit)
# ============================================================
_theta_by_unit_ctx = {}
_P_by_unit_ctx     = {}
_params_loaded     = {}

_last_defrosting   = {}
_cooldown_until    = {}

_last_resort_since = {}

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
    if unit_name not in _last_resort_since:
        _last_resort_since[unit_name] = 0.0

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
        state.set(store, value=time.time(), theta_by_ctx=clean)
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

    for key in (_theta_by_unit_ctx.get(unit_name) or {}).keys():
        _P_by_unit_ctx[unit_name][str(key)] = [[P0, 0, 0, 0],
                                               [0, P0, 0, 0],
                                               [0, 0, P0, 0],
                                               [0, 0, 0, P0]]
    _params_loaded[unit_name] = True


# ============================================================
# 5) STARTUP + TRIGGERS
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
                note="init"
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
# 6) CONTROL LOGIC (per unit)
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

    QUIET_SWITCH = u.get("QUIET_SWITCH")
    quiet_available = _entity_exists(QUIET_SWITCH) if QUIET_SWITCH else False

    PRICE_BIAS_HELPER = u.get("PRICE_BIAS_HELPER")

    SP_HELPER = u["SP_HELPER"]
    STEP_LIMIT_HELPER = u["STEP_LIMIT_HELPER"]
    DEADBAND_HELPER = u["DEADBAND_HELPER"]
    ICING_CAP_HELPER = u["ICING_CAP_HELPER"]
    LEARNED_SENSOR = u["LEARNED_SENSOR"]

    learning_enabled = _learning_enabled_for_unit(u)

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

    quiet_prev = _quiet_is_on(QUIET_SWITCH) if quiet_available else False

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

    if Tout_bucket <= -5:
        global_upper = MAX_DEM
    else:
        global_upper = GLOBAL_MILD_MAX

    try:
        icing_cap = float(state.get(ICING_CAP_HELPER) or ICING_BAND_CAP_DEFAULT)
    except Exception:
        icing_cap = ICING_BAND_CAP_DEFAULT
    if not isfinite(icing_cap):
        icing_cap = ICING_BAND_CAP_DEFAULT
    icing_cap = _clip(icing_cap, MIN_DEM, MAX_DEM)

    in_icing_band = (Tout_bucket >= ICING_BAND_MIN and Tout_bucket <= ICING_BAND_MAX)
    band_upper = min(global_upper, icing_cap) if in_icing_band else global_upper

    if prev > band_upper:
        if quiet_available:
            _set_quiet(QUIET_SWITCH, False, unit_name)
        option_cap = _snap_to_select(SELECT, band_upper, -1)
        if option_cap and option_cap != prev_str:
            select.select_option(entity_id=SELECT, option=option_cap)
            log.info(
                "Daikin ML (%s): CAP ENFORCED: prev=%s -> %s (upper=%.0f%%, ctx=%s, Tout_bucket=%d)",
                unit_name, prev_str, option_cap, band_upper, ctx, Tout_bucket,
            )
        _last_resort_since[unit_name] = 0.0
        return

    prev_eff = prev if prev <= band_upper else band_upper

    err         = sp - Tin
    demand_norm = _clip(prev_eff / 100.0, 0.0, 1.0)
    x = [1.0, err, demand_norm, (Tout_raw - Tin)]
    y = rate

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
        dem_opt = prev_eff

    price_bias = _read_price_bias_points(u)

    # learned sensor (attrs always updated)
    try:
        if learning_enabled:
            learned_value = round(float(dem_opt), 1)
            note = "learning enabled: value follows dem_opt"
        else:
            prev_val = state.get(LEARNED_SENSOR)
            try:
                learned_value = float(prev_val)
                if not isfinite(learned_value):
                    learned_value = 0.0
            except Exception:
                learned_value = 0.0
            learned_value = round(float(learned_value), 1)
            note = "learning disabled: value frozen, attributes updated"

        state.set(
            LEARNED_SENSOR,
            value=learned_value,
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
            price_bias_points=round(float(price_bias), 2),
            price_bias_helper=str(PRICE_BIAS_HELPER),
            quiet_available=bool(quiet_available),
            note=note,
        )
    except Exception as e:
        log.error("Daikin ML (%s): failed to update learned demand sensor: %s", unit_name, e)

    if abs(err) <= deadband:
        dem_target = prev_eff
    else:
        dem_target = dem_opt

    if err > deadband and dem_target < prev_eff:
        dem_target = prev_eff

    if abs(price_bias) > 0.0001:
        dem_target = dem_target + price_bias

    delta = dem_target - prev_eff
    if delta > 0:
        up_limit = COOLDOWN_STEP_UP if in_cooldown else step_limit
        if delta > up_limit:
            dem_target = prev_eff + up_limit
    elif delta < 0:
        down_limit = step_limit
        if -delta > down_limit:
            dem_target = prev_eff - down_limit

    dem_clip = _clip(dem_target, MIN_DEM, min(band_upper, 100.0))

    # REAL 100 last-resort gating only if quiet is available (because otherwise no intermediate exists)
    allow_real_100 = True
    want_full = (dem_clip >= 99.95) and (band_upper >= 100.0)

    if quiet_available:
        allow_real_100 = False
        if want_full and (err > LAST_RESORT_MIN_ERR_C) and (not defrosting):
            if _last_resort_since[unit_name] <= 0.0:
                _last_resort_since[unit_name] = now
            elapsed = now - _last_resort_since[unit_name]
            if err >= LAST_RESORT_ERR_C:
                allow_real_100 = True
            elif elapsed >= (LAST_RESORT_MINUTES * 60.0):
                allow_real_100 = True
        else:
            _last_resort_since[unit_name] = 0.0

    # Apply final command
    if quiet_available:
        option = _apply_demand_with_quiet(unit_name, SELECT, QUIET_SWITCH, dem_clip, prev_str, allow_real_100)
    else:
        # No quiet: plain snap, jump 95->100 normally
        option = _apply_demand(unit_name, SELECT, dem_clip, prev_str)

    lr_elapsed = 0.0
    if _last_resort_since[unit_name] > 0.0:
        lr_elapsed = now - _last_resort_since[unit_name]

    log.info(
        "Daikin ML (%s): ctx=%s | Tin=%.2f°C, Tout=%.2f°C (bucket=%d, →%.1f°C), "
        "DEFROST=%s, cooldown=%s, learning_enabled=%s, learning_allowed=%s, rate_bad=%s, "
        "band_upper=%.0f%%, quiet_available=%s | "
        "SP=%.2f, step_limit=%.1f, deadband=%.2f | "
        "price_bias=%.2f pts | "
        "last_resort: want_full=%s allow_real_100=%s elapsed=%.0fs | "
        "prev=%s (quiet_prev=%s) → opt≈%.0f%% → target≈%.0f%% → clip=%.0f%% → select=%s (quiet_now=%s)",
        unit_name, ctx, Tin, Tout_raw, Tout_bucket, Tout_future,
        str(defrosting), ("ACTIVE" if in_cooldown else "off"),
        str(learning_enabled), str(allow_learning), str(rate_bad),
        band_upper, str(quiet_available),
        sp, step_limit, deadband,
        float(price_bias),
        str(want_full), str(allow_real_100), float(lr_elapsed),
        prev_str, str(quiet_prev), dem_opt, dem_target, dem_clip, option,
        str(_quiet_is_on(QUIET_SWITCH) if quiet_available else False),
    )


# ============================================================
# 7) SERVICES
# ============================================================
@service
def daikin_ml_step():
    """Run controller once (all units)."""
    daikin_ml_controller()

@service
def daikin_ml_reset():
    """Reset learning for all units."""
    global _theta_by_unit_ctx, _P_by_unit_ctx, _params_loaded, _last_defrosting, _cooldown_until, _last_resort_since

    for u in DAIKINS:
        unit_name = u["name"]
        _init_unit_if_needed(unit_name)

        old_cnt = len(_theta_by_unit_ctx.get(unit_name) or {})
        _theta_by_unit_ctx[unit_name] = {}
        _P_by_unit_ctx[unit_name] = {}
        _params_loaded[unit_name] = False
        _last_defrosting[unit_name] = None
        _cooldown_until[unit_name] = 0.0
        _last_resort_since[unit_name] = 0.0

        try:
            state.set(u["STORE_ENTITY"], value=time.time(), theta_by_ctx={})
            log.warning(
                "Daikin ML RESET (%s): cleared %d learned contexts, theta_by_ctx now empty in %s",
                unit_name, old_cnt, u["STORE_ENTITY"],
            )
        except Exception as e:
            log.error("Daikin ML RESET (%s): error resetting params in %s: %s", unit_name, u["STORE_ENTITY"], e)

@service
def daikin_ml_persist():
    """Persist store entities (safety)."""
    _persist_all_stores()

@service
def daikin_ml_learning_enable():
    """Enable learning globally."""
    try:
        input_boolean.turn_on(entity_id=LEARNING_ENABLED_HELPER_GLOBAL)
        log.info("Daikin ML: learning enabled (global) -> %s", LEARNING_ENABLED_HELPER_GLOBAL)
    except Exception as e:
        log.error("Daikin ML: failed to enable learning global: %s", e)

@service
def daikin_ml_learning_disable():
    """Disable learning globally."""
    try:
        input_boolean.turn_off(entity_id=LEARNING_ENABLED_HELPER_GLOBAL)
        log.info("Daikin ML: learning disabled (global) -> %s", LEARNING_ENABLED_HELPER_GLOBAL)
    except Exception as e:
        log.error("Daikin ML: failed to disable learning global: %s", e)

@service
def daikin_ml_learning_toggle():
    """Toggle learning globally."""
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
# 8) PERSIST STORES AT LOAD
# ============================================================
_persist_all_stores()
