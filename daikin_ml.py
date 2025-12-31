# pyscript/daikin_ml_multi.py
# Online-RLS Daikin demand controller with MULTI-DAIKIN support.
#
import time
from math import isfinite

# ============================================================
# 0) LEARNING ENABLE HELPERS
# ============================================================
LEARNING_ENABLED_HELPER_GLOBAL = "input_boolean.daikin_ml_learning_enabled"

# ============================================================
# Optional fireplace switch disables learning
# ============================================================
FIREPLACE_SWITCH = "switch.fireplace"

# ============================================================
# Global Nordpool avg window helper
# ============================================================
NORDPOOL_AVG_WINDOW_HELPER = "input_number.nordpool_avg_window_hours"
NORDPOOL_AVG_WINDOW_STEP_H = 0.25  # 15 min slots

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
        "FORECAST_HOURLY": "sensor.weather_forecast_hourly",

        "SELECT": "select.faikin_demand_control",
        "LIQUID": "sensor.faikin_liquid",

        "SP_HELPER": "input_number.daikin_setpoint",
        "SP_BASE_HELPER": "input_number.daikin_setpoint_base",

        "STEP_LIMIT_HELPER": "input_number.daikin_step_limit",
        "DEADBAND_HELPER": "input_number.daikin_deadband",
        "ICING_CAP_HELPER": "input_number.daikin_icing_cap",

        "STORE_ENTITY": "pyscript.daikin1_ml_params",
        "LEARNED_SENSOR": "sensor.daikin1_ml_learned_demand",

        "LEARNING_ENABLED_HELPER": "input_boolean.daikin1_ml_learning_enabled",

        "QUIET_SWITCH": "switch.faikin_quiet_outdoor",

        "PRICE_BIAS_HELPER": "input_number.daikin1_price_bias_points",

        "MIN_TEMP_HELPER": "input_number.daikin1_min_temp_guard",
        "MAX_TEMP_HELPER": "input_number.daikin1_max_temp_guard",

        # Optional sensors (ONLY used if exist)
        "POWER_SENSOR": "sensor.daikin_p40_power",
        "COMP_FREQ_SENSOR": "sensor.faikin_comp",

        # Optional price sensor for volatility-aware window (ONLY if exists)
        # Must be an entity that has list-like prices accessible via attributes or state JSON.
        "PRICE_SENSOR": "sensor.day_ahead_price",
    },
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

COOLDOWN_MINUTES  = 30
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

LAST_RESORT_MIN_ERR_C = 0.5
LAST_RESORT_ERR_C = 1.5
LAST_RESORT_MINUTES = 30

TEMP_GUARD_MIN_DEFAULT = -100.0
TEMP_GUARD_MAX_DEFAULT =  100.0

# ============================================================
# Confidence scoring & auto-freeze (blend)
# ============================================================
CONF_COUNT_SCALE = 10.0
CONF_P_NORM_SCALE = 1.0
CONF_FREEZE_THRESHOLD = 0.15
CONF_MAX = 1.0

# Confidence-driven step scaling
CONF_STEP_MIN_SCALE = 0.4   # at very low confidence, step_limit scaled down
CONF_STEP_MAX_SCALE = 1.0   # at high confidence, unchanged

# ============================================================
# Price bias -> setpoint delta mapping
# ============================================================
PRICE_BIAS_POINTS_AT_MAX = 20.0
PRICE_BIAS_MAX_SETPOINT_DELTA_C = 0.5
PRICE_BIAS_SETPOINT_MIN_C = 5.0
PRICE_BIAS_SETPOINT_MAX_C = 30.0

# Setpoint ramp limiting (per tick) - reduces oscillation/cycling.
# Controller runs every ~6 min + triggers; treat as "tick".
SETPOINT_RAMP_MAX_DELTA_C_PER_TICK = 0.10  # allow max 0.10°C per run

# ============================================================
# Humidity -> defrost/icing risk + learning feature injection
# ============================================================
HUMIDITY_LOW_PCT = 40.0
HUMIDITY_HIGH_PCT = 90.0

ICING_RISK_CENTER_C = 1.0
ICING_RISK_HALF_WIDTH_C = 3.0

DYNAMIC_CAP_MAX_REDUCTION_FRAC = 0.25

# Frost/defrost prediction
DEFROST_RISK_THRESHOLD = 0.60
FROST_INTEGRATOR_THRESHOLD = 1.0
FROST_INTEGRATOR_DECAY_PER_H = 0.10  # decay per hour
FROST_INTEGRATOR_GAIN = 0.50         # how fast integrator accumulates
FROST_INTEGRATOR_MIN_SECONDS = 10.0  # ignore too-frequent ticks

# Dynamic icing cap hysteresis (fast clamp, slow release)
CAP_HYST_RELEASE_PER_H = 0.20  # release towards base icing cap per hour
CAP_HYST_CLAMP_FAST_FACTOR = 1.0  # immediate clamp to dynamic cap

HUMIDITY_FEATURE_K = 0.05

# ============================================================
# Wind factor -> learning feature injection
# ============================================================
WIND_SPEED_LOW = 0.0
WIND_SPEED_HIGH = 20.0
WIND_COLD_START_C = 0.0
WIND_COLD_FULL_C = -10.0
WIND_FEATURE_K = 0.07

# ============================================================
# Dynamic Nordpool avg window (hours) based on conditions
# ============================================================
NORDPOOL_AVG_WINDOW_MIN_H = 1.0
NORDPOOL_AVG_WINDOW_MAX_H = 6.0

NORDPOOL_HARSH_TEMP_FULL_C = -10.0
NORDPOOL_SEV_TEMP_WEIGHT = 0.75
NORDPOOL_SEV_WIND_WEIGHT = 0.25

# Optional volatility influence (only if price list is available)
VOLATILITY_LOOKAHEAD_SLOTS = 16  # 16*15min = 4h
VOLATILITY_MIN_STD = 0.0
VOLATILITY_MAX_STD = 15.0  # cents/MWh-ish scaling; will be normalized; safe even if units differ
VOLATILITY_WINDOW_SHRINK_WEIGHT = 0.40  # reduce window when volatile (blended into severity)

# ============================================================
# Steady-state learning gate
# ============================================================
STEADY_AFTER_CHANGE_SECONDS = 12 * 60  # require 12 minutes after demand/setpoint changes
RATE_STEADY_MAX_ABS = 0.02            # if indoor rate too high => not steady (°C/min or unit-specific)
ERR_STEADY_MAX_ABS = 1.0              # if error too big => not steady for learning

# ============================================================
# Theta anomaly guard + freeze
# ============================================================
THETA_ABS_MAX = 50.0
THETA_MIN_DENOM_MAG = 0.1
ANOMALY_FREEZE_SECONDS = 3 * 60 * 60  # 3 hours freeze per ctx if anomaly detected

# ============================================================
# Gentle exploration micro-dither (only when safe)
# ============================================================
EXPLORATION_ENABLED = True
EXPLORATION_MAX_DELTA_DEM = 1.0     # +/- 1%
EXPLORATION_ERR_MAX = 0.4           # only when near setpoint
EXPLORATION_MIN_CONF = 0.25         # only when confidence low
EXPLORATION_MIN_SECONDS = 30 * 60   # at most every 30 minutes per ctx

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

def _round_to_step(v, step):
    try:
        fv = float(v)
        fs = float(step)
    except Exception:
        return v
    if not isfinite(fv):
        return v
    if (not isfinite(fs)) or fs <= 0:
        return fv
    return round(fv / fs) * fs

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

    return round(step_auto, 1), round(db_auto, 2)

def _read_forecast_list(forecast_sensor):
    if not forecast_sensor:
        return []
    try:
        attrs = state.getattr(forecast_sensor) or {}
    except Exception:
        return []
    fc = attrs.get("forecast") or []
    return fc if isinstance(fc, list) else []

def _avg_future_outdoor_from_hourly_forecast(forecast_sensor, outdoor_entity):
    fc = _read_forecast_list(forecast_sensor)
    temps = []
    idx = 0
    for f in fc:
        if idx >= FORECAST_H:
            break
        t = f.get("temperature")
        if t is None:
            continue
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

def _read_current_humidity_from_hourly_forecast(forecast_sensor, fallback_weather_entity=None):
    fc = _read_forecast_list(forecast_sensor)
    if len(fc) > 0:
        h = fc[0].get("humidity")
        if h is not None:
            try:
                hv = float(h)
                if isfinite(hv):
                    return _clip(hv, 0.0, 100.0)
            except Exception:
                pass

    if fallback_weather_entity:
        try:
            attrs = state.getattr(fallback_weather_entity) or {}
            h = attrs.get("humidity")
            if h is not None:
                hv = float(h)
                if isfinite(hv):
                    return _clip(hv, 0.0, 100.0)
        except Exception:
            pass
    return None

def _read_current_wind_speed_from_hourly_forecast(forecast_sensor):
    fc = _read_forecast_list(forecast_sensor)
    if len(fc) > 0:
        w = fc[0].get("wind_speed")
        if w is not None:
            try:
                wv = float(w)
                if isfinite(wv):
                    return max(0.0, wv)
            except Exception:
                pass
    return None

def _humidity_norm(h_pct):
    if h_pct is None:
        return 0.0
    return _clip((float(h_pct) - 65.0) / 50.0, -1.0, 1.0)

def _wind_norm(w_speed):
    if w_speed is None:
        return 0.0
    if WIND_SPEED_HIGH <= WIND_SPEED_LOW:
        return 0.0
    return _clip((float(w_speed) - WIND_SPEED_LOW) / (WIND_SPEED_HIGH - WIND_SPEED_LOW), 0.0, 1.0)

def _cold_factor(Tout_raw):
    if not isfinite(Tout_raw):
        return 0.0
    if WIND_COLD_FULL_C >= WIND_COLD_START_C:
        return 0.0
    if Tout_raw >= WIND_COLD_START_C:
        return 0.0
    if Tout_raw <= WIND_COLD_FULL_C:
        return 1.0
    return _clip(
        (float(WIND_COLD_START_C) - float(Tout_raw)) / (float(WIND_COLD_START_C) - float(WIND_COLD_FULL_C)),
        0.0, 1.0
    )

def _icing_defrost_risk(Tout_raw, humidity_pct, in_icing_band):
    if humidity_pct is None:
        hum_risk = 0.0
    else:
        hum_risk = _clip(
            (float(humidity_pct) - HUMIDITY_LOW_PCT) / (HUMIDITY_HIGH_PCT - HUMIDITY_LOW_PCT),
            0.0, 1.0
        )

    if not isfinite(Tout_raw):
        temp_risk = 0.0
    else:
        dist = abs(float(Tout_raw) - float(ICING_RISK_CENTER_C))
        temp_risk = 1.0 - _clip(dist / float(ICING_RISK_HALF_WIDTH_C), 0.0, 1.0)

    if not in_icing_band:
        return 0.0, float(hum_risk), float(temp_risk)

    risk = _clip(float(hum_risk) * float(temp_risk), 0.0, 1.0)
    return float(risk), float(hum_risk), float(temp_risk)

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

def _fireplace_is_on():
    if not FIREPLACE_SWITCH:
        return False
    if not _entity_exists(FIREPLACE_SWITCH):
        return False
    try:
        return str(state.get(FIREPLACE_SWITCH)).strip().lower() == "on"
    except Exception:
        return False

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
        elif (not turn_on) and cur:
            switch.turn_off(entity_id=quiet_switch)
    except Exception as e:
        log.error("Daikin ML (%s): failed to set QUIET_SWITCH %s: %s", unit_name, quiet_switch, e)

def _set_input_number_value(entity_id, value):
    if not entity_id:
        return
    try:
        input_number.set_value(entity_id=entity_id, value=float(value))
    except Exception as e:
        log.debug("Daikin ML: could not set %s. err=%s", entity_id, e)

def _apply_demand(unit_name, select_entity, target_percent, prev_str, direction=0):
    option = _snap_to_select(select_entity, float(target_percent), direction)
    if option and option != prev_str:
        select.select_option(entity_id=select_entity, option=option)
    return option

def _apply_demand_with_quiet(unit_name, select_entity, quiet_switch, target_percent, prev_str, allow_real_100, direction=0):
    t = float(target_percent)

    if t >= 100.0:
        if allow_real_100:
            _set_quiet(quiet_switch, False, unit_name)
            option = _snap_to_select(select_entity, 100.0, direction)
        else:
            _set_quiet(quiet_switch, True, unit_name)
            option = _snap_to_select(select_entity, 100.0, direction)
    elif t > GLOBAL_MILD_MAX:
        _set_quiet(quiet_switch, True, unit_name)
        option = _snap_to_select(select_entity, 100.0, direction)
    else:
        _set_quiet(quiet_switch, False, unit_name)
        option = _snap_to_select(select_entity, t, direction)

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

def _price_bias_to_setpoint_delta_c(points):
    try:
        p = float(points)
    except Exception:
        return 0.0
    if not isfinite(p):
        return 0.0
    if PRICE_BIAS_POINTS_AT_MAX <= 0:
        return 0.0
    frac = _clip(p / float(PRICE_BIAS_POINTS_AT_MAX), -1.0, 1.0)
    return float(frac) * float(PRICE_BIAS_MAX_SETPOINT_DELTA_C)

def _read_temp_guard(unit):
    min_ent = unit.get("MIN_TEMP_HELPER")
    max_ent = unit.get("MAX_TEMP_HELPER")

    min_v = TEMP_GUARD_MIN_DEFAULT
    max_v = TEMP_GUARD_MAX_DEFAULT

    if min_ent:
        try:
            mv = float(state.get(min_ent))
            if isfinite(mv):
                min_v = mv
        except Exception:
            pass

    if max_ent:
        try:
            mv = float(state.get(max_ent))
            if isfinite(mv):
                max_v = mv
        except Exception:
            pass

    if min_v > max_v:
        min_v, max_v = max_v, min_v

    return float(min_v), float(max_v), str(min_ent), str(max_ent)

def _p_trace(P):
    try:
        t = float(P[0][0]) + float(P[1][1]) + float(P[2][2]) + float(P[3][3])
        return t if isfinite(t) else float(4.0 * P0)
    except Exception:
        return float(4.0 * P0)

def _confidence_from(count, P):
    c_count = float(count) / float(count + CONF_COUNT_SCALE) if count >= 0 else 0.0

    tr = _p_trace(P)
    norm_trace = tr / float(4.0 * P0) if P0 > 0 else 1.0
    if not isfinite(norm_trace):
        norm_trace = 1.0
    norm_trace = _clip(norm_trace, 0.0, 10.0)

    c_p = 1.0 / (1.0 + CONF_P_NORM_SCALE * norm_trace)
    conf = _clip(c_count * (2.0 * c_p), 0.0, CONF_MAX)
    return float(conf), float(tr), float(norm_trace), float(c_count), float(c_p)

def _confidence_step_scale(conf):
    c = _clip(float(conf), 0.0, 1.0)
    # Low conf -> smaller steps, high conf -> normal
    return float(CONF_STEP_MIN_SCALE + (CONF_STEP_MAX_SCALE - CONF_STEP_MIN_SCALE) * c)

def _nordpool_severity_from_conditions(Tout_raw, wind_cold_factor):
    if not isfinite(Tout_raw):
        return 0.0

    if Tout_raw >= ICING_BAND_MAX:
        temp_sev = 0.0
    elif Tout_raw <= NORDPOOL_HARSH_TEMP_FULL_C:
        temp_sev = 1.0
    else:
        denom = float(ICING_BAND_MAX - NORDPOOL_HARSH_TEMP_FULL_C)
        if denom <= 0:
            temp_sev = 0.0
        else:
            temp_sev = _clip((float(ICING_BAND_MAX) - float(Tout_raw)) / denom, 0.0, 1.0)

    w = _clip(float(wind_cold_factor), 0.0, 1.0)
    sev = _clip(
        (NORDPOOL_SEV_TEMP_WEIGHT * temp_sev) + (NORDPOOL_SEV_WIND_WEIGHT * w),
        0.0, 1.0
    )
    return float(sev)

def _nordpool_window_hours_from_severity(sev):
    sev = _clip(float(sev), 0.0, 1.0)
    lo = float(NORDPOOL_AVG_WINDOW_MIN_H)
    hi = float(NORDPOOL_AVG_WINDOW_MAX_H)
    if hi < lo:
        lo, hi = hi, lo
    return float(hi - sev * (hi - lo))

def _read_float_optional(entity_id):
    if not entity_id:
        return None
    if not _entity_exists(entity_id):
        return None
    try:
        v = float(state.get(entity_id))
        return v if isfinite(v) else None
    except Exception:
        return None

def _theta_anomalous(theta):
    if not isinstance(theta, (list, tuple)) or len(theta) != 4:
        return True
    if not _all_finite(theta):
        return True
    for v in theta:
        if abs(float(v)) > THETA_ABS_MAX:
            return True
    # denom magnitude sanity
    try:
        if abs(float(theta[2])) < THETA_MIN_DENOM_MAG:
            return True
    except Exception:
        return True
    return False

def _extract_price_list(price_sensor):
    """Best-effort optional extraction of 15m-ish price list from an entity.
    Accepts:
      - attributes: prices / price / raw_today / raw_tomorrow / data / etc (list of numbers)
      - state as JSON-like string list -> ignored (we avoid json parsing to keep pyscript simple)
    Returns list[float] or [].
    """
    if not price_sensor or (not _entity_exists(price_sensor)):
        return []
    attrs = state.getattr(price_sensor) or {}
    candidates = []
    for k in ("prices", "price", "data", "raw_today", "raw_tomorrow", "values"):
        v = attrs.get(k)
        if isinstance(v, list) and len(v) > 0:
            candidates = v
            break
    out = []
    if isinstance(candidates, list):
        for it in candidates:
            try:
                fv = float(it)
                if isfinite(fv):
                    out.append(fv)
            except Exception:
                pass
    return out

def _volatility_norm_from_prices(price_list, lookahead_slots):
    if not price_list:
        return 0.0, 0.0
    n = int(lookahead_slots)
    if n <= 2:
        return 0.0, 0.0
    seg = price_list[:n] if len(price_list) >= n else price_list[:]
    if len(seg) < 3:
        return 0.0, 0.0
    m = sum(seg) / float(len(seg))
    var = 0.0
    for v in seg:
        dv = float(v) - float(m)
        var += dv * dv
    var /= float(len(seg) - 1)
    std = var ** 0.5
    std_norm = _clip((std - VOLATILITY_MIN_STD) / (VOLATILITY_MAX_STD - VOLATILITY_MIN_STD + 1e-9), 0.0, 1.0)
    return float(std_norm), float(std)

def _why_add(why_list, s):
    if s:
        why_list.append(str(s))


# ============================================================
# 4) STATE (per unit)
# ============================================================
_theta_by_unit_ctx = {}
_P_by_unit_ctx     = {}
_params_loaded     = {}

_last_defrosting   = {}
_cooldown_until    = {}
_last_resort_since = {}

_updates_by_unit_ctx = {}

# new states for "everything"
_frost_integrator = {}         # per unit
_last_tick_ts = {}             # per unit
_last_demand_change_ts = {}    # per unit (select change)
_last_setpoint_write_ts = {}   # per unit (SP_HELPER writes)
_last_sp_written = {}          # per unit
_cap_hyst_upper = {}           # per unit (smoothed band_upper in icing band)
_freeze_until_by_ctx = {}      # per unit -> dict ctx->timestamp
_exploration_last_ts = {}      # per unit -> dict ctx->timestamp
_exploration_phase = {}        # per unit -> +/-1

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
    if unit_name not in _updates_by_unit_ctx:
        _updates_by_unit_ctx[unit_name] = {}

    if unit_name not in _frost_integrator:
        _frost_integrator[unit_name] = 0.0
    if unit_name not in _last_tick_ts:
        _last_tick_ts[unit_name] = 0.0
    if unit_name not in _last_demand_change_ts:
        _last_demand_change_ts[unit_name] = 0.0
    if unit_name not in _last_setpoint_write_ts:
        _last_setpoint_write_ts[unit_name] = 0.0
    if unit_name not in _last_sp_written:
        _last_sp_written[unit_name] = None
    if unit_name not in _cap_hyst_upper:
        _cap_hyst_upper[unit_name] = None
    if unit_name not in _freeze_until_by_ctx:
        _freeze_until_by_ctx[unit_name] = {}
    if unit_name not in _exploration_last_ts:
        _exploration_last_ts[unit_name] = {}
    if unit_name not in _exploration_phase:
        _exploration_phase[unit_name] = 1

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

    ub = attrs.get("updates_by_ctx")
    upd_new = {}
    if isinstance(ub, dict):
        for key, val in ub.items():
            try:
                iv = int(val)
                if iv < 0:
                    iv = 0
                upd_new[str(key)] = iv
            except Exception:
                continue
    if upd_new:
        _updates_by_unit_ctx[unit_name] = upd_new

    fb = attrs.get("freeze_until_by_ctx")
    fr_new = {}
    if isinstance(fb, dict):
        for key, val in fb.items():
            try:
                tv = float(val)
                if isfinite(tv):
                    fr_new[str(key)] = tv
            except Exception:
                continue
    if fr_new:
        _freeze_until_by_ctx[unit_name] = fr_new

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

    upd_clean = {}
    for key, v in (_updates_by_unit_ctx.get(unit_name) or {}).items():
        try:
            iv = int(v)
            if iv < 0:
                iv = 0
            upd_clean[str(key)] = iv
        except Exception:
            continue

    fr_clean = {}
    for key, v in (_freeze_until_by_ctx.get(unit_name) or {}).items():
        try:
            fv = float(v)
            if isfinite(fv):
                fr_clean[str(key)] = fv
        except Exception:
            continue

    try:
        state.set(store, value=time.time(), theta_by_ctx=clean, updates_by_ctx=upd_clean, freeze_until_by_ctx=fr_clean)
    except Exception as e:
        log.error("Daikin ML (%s): error saving thetas to %s: %s", unit_name, store, e)

def _init_context_params_if_needed(unit):
    unit_name = unit["name"]
    _init_unit_if_needed(unit_name)
    if _params_loaded.get(unit_name):
        return

    _theta_by_unit_ctx[unit_name] = {}
    _P_by_unit_ctx[unit_name] = {}
    _updates_by_unit_ctx[unit_name] = {}
    _freeze_until_by_ctx[unit_name] = {}

    _load_params_from_store(unit)

    for key in (_theta_by_unit_ctx.get(unit_name) or {}).keys():
        k = str(key)
        _P_by_unit_ctx[unit_name][k] = [[P0, 0, 0, 0],
                                        [0, P0, 0, 0],
                                        [0, 0, P0, 0],
                                        [0, 0, 0, P0]]
        if k not in (_updates_by_unit_ctx.get(unit_name) or {}):
            _updates_by_unit_ctx[unit_name][k] = 0
        if k not in (_freeze_until_by_ctx.get(unit_name) or {}):
            _freeze_until_by_ctx[unit_name][k] = 0.0

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

    # optional: trigger on power/frequency if present
    for k in ("POWER_SENSOR", "COMP_FREQ_SENSOR"):
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
    FORECAST_HOURLY = u.get("FORECAST_HOURLY")

    SELECT = u["SELECT"]
    LIQUID = u["LIQUID"]

    QUIET_SWITCH = u.get("QUIET_SWITCH")
    quiet_available = _entity_exists(QUIET_SWITCH) if QUIET_SWITCH else False

    PRICE_BIAS_HELPER = u.get("PRICE_BIAS_HELPER")

    SP_HELPER = u["SP_HELPER"]
    SP_BASE_HELPER = u.get("SP_BASE_HELPER")

    STEP_LIMIT_HELPER = u["STEP_LIMIT_HELPER"]
    DEADBAND_HELPER = u["DEADBAND_HELPER"]
    ICING_CAP_HELPER = u["ICING_CAP_HELPER"]
    LEARNED_SENSOR = u["LEARNED_SENSOR"]

    POWER_SENSOR = u.get("POWER_SENSOR")
    COMP_FREQ_SENSOR = u.get("COMP_FREQ_SENSOR")
    PRICE_SENSOR = u.get("PRICE_SENSOR")

    now = time.time()

    # tick dt (for integrators/hysteresis)
    last_ts = float(_last_tick_ts.get(unit_name, 0.0) or 0.0)
    dt = max(0.0, now - last_ts) if last_ts > 0 else 0.0
    _last_tick_ts[unit_name] = now

    # base learning enable state from helpers
    learning_enabled = _learning_enabled_for_unit(u)

    # fireplace ON => disable learning
    fireplace_on = _fireplace_is_on()
    if fireplace_on:
        learning_enabled = False

    # read sensors
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
        rate = 0.0
        rate_bad = True
    else:
        rate_bad = False

    power_w = _read_float_optional(POWER_SENSOR)
    comp_hz = _read_float_optional(COMP_FREQ_SENSOR)

    humidity_pct = _read_current_humidity_from_hourly_forecast(FORECAST_HOURLY, fallback_weather_entity=WEATHER)
    wind_speed = _read_current_wind_speed_from_hourly_forecast(FORECAST_HOURLY)

    wnorm = _wind_norm(wind_speed)
    cold = _cold_factor(Tout_raw)
    wind_cold_factor = wnorm * cold

    # --- Dynamic Nordpool avg window + volatility (optional) ---
    nordpool_sev_base = _nordpool_severity_from_conditions(Tout_raw, wind_cold_factor)

    # Optional volatility component
    price_list = _extract_price_list(PRICE_SENSOR)
    vol_norm, vol_std = _volatility_norm_from_prices(price_list, VOLATILITY_LOOKAHEAD_SLOTS)
    nordpool_sev = _clip(nordpool_sev_base + VOLATILITY_WINDOW_SHRINK_WEIGHT * vol_norm, 0.0, 1.0)

    if NORDPOOL_AVG_WINDOW_HELPER and _entity_exists(NORDPOOL_AVG_WINDOW_HELPER):
        nordpool_avg_window_h_raw = _nordpool_window_hours_from_severity(nordpool_sev)
        nordpool_avg_window_h = _clip(
            _round_to_step(nordpool_avg_window_h_raw, NORDPOOL_AVG_WINDOW_STEP_H),
            NORDPOOL_AVG_WINDOW_MIN_H,
            NORDPOOL_AVG_WINDOW_MAX_H
        )
        nordpool_avg_window_h = _round_to_step(nordpool_avg_window_h, NORDPOOL_AVG_WINDOW_STEP_H)
        _set_input_number_value(NORDPOOL_AVG_WINDOW_HELPER, float(nordpool_avg_window_h))
    else:
        nordpool_avg_window_h = None

    # --- Nordpool bias points ---
    price_bias = _read_price_bias_points(u)

    # --- Base setpoint + biased setpoint target ---
    base_sp = None
    if SP_BASE_HELPER:
        try:
            v = state.get(SP_BASE_HELPER)
            if v is not None:
                fv = float(v)
                if isfinite(fv):
                    base_sp = fv
        except Exception:
            base_sp = None

    if base_sp is None:
        try:
            base_sp = float(state.get(SP_HELPER) or 22.5)
        except Exception:
            base_sp = 22.5
        if not isfinite(base_sp):
            base_sp = 22.5

    sp_delta = _price_bias_to_setpoint_delta_c(price_bias)
    sp_target = _clip(base_sp + sp_delta, PRICE_BIAS_SETPOINT_MIN_C, PRICE_BIAS_SETPOINT_MAX_C)

    # --- Setpoint ramp limiting (actual write) ---
    prev_sp_written = _last_sp_written.get(unit_name)
    if prev_sp_written is None:
        # try reading current helper to establish baseline
        try:
            cur_sp_val = float(state.get(SP_HELPER) or sp_target)
            prev_sp_written = cur_sp_val if isfinite(cur_sp_val) else sp_target
        except Exception:
            prev_sp_written = sp_target

    sp_write = sp_target
    max_step = float(SETPOINT_RAMP_MAX_DELTA_C_PER_TICK)
    try:
        delta_sp = float(sp_target) - float(prev_sp_written)
        if delta_sp > max_step:
            sp_write = float(prev_sp_written) + max_step
        elif delta_sp < -max_step:
            sp_write = float(prev_sp_written) - max_step
    except Exception:
        sp_write = sp_target

    sp_write = _clip(sp_write, PRICE_BIAS_SETPOINT_MIN_C, PRICE_BIAS_SETPOINT_MAX_C)
    sp_write = round(float(sp_write), 2)
    _set_input_number_value(SP_HELPER, sp_write)
    _last_sp_written[unit_name] = float(sp_write)
    _last_setpoint_write_ts[unit_name] = now

    # Use written setpoint for control
    sp = float(sp_write)

    # Read control helpers
    step_limit_current = float(state.get(STEP_LIMIT_HELPER) or 10.0)
    deadband_current   = float(state.get(DEADBAND_HELPER) or 0.1)

    prev_str = state.get(SELECT) or ""
    try:
        prev = float(prev_str.replace('%', ''))
    except Exception:
        prev = 60.0

    # Detect demand (select) change for steady-state gating
    # (We don’t know “who” changed it; any change restarts steady timer)
    # Track last applied option string as baseline
    last_seen_option = getattr(_run_one_unit, "__last_seen_option_" + unit_name, None)
    if last_seen_option is None:
        setattr(_run_one_unit, "__last_seen_option_" + unit_name, prev_str)
        last_seen_option = prev_str
    if prev_str != last_seen_option:
        _last_demand_change_ts[unit_name] = now
        setattr(_run_one_unit, "__last_seen_option_" + unit_name, prev_str)

    # Defrost detect
    try:
        liquid = float(state.get(LIQUID) or 100.0)
    except Exception:
        liquid = 100.0
    if not isfinite(liquid):
        liquid = 100.0
    defrosting = (liquid < 20.0)

    if _last_defrosting[unit_name] is None:
        _last_defrosting[unit_name] = defrosting

    # cooldown starts at beginning of defrost
    if (_last_defrosting[unit_name] is False) and (defrosting is True):
        _cooldown_until[unit_name] = now + COOLDOWN_MINUTES * 60.0
        # reset frost integrator on actual defrost start
        _frost_integrator[unit_name] = 0.0

    _last_defrosting[unit_name] = defrosting
    in_cooldown = (now < _cooldown_until[unit_name])

    if not (isfinite(Tin) and isfinite(Tout_raw)):
        return

    min_temp, max_temp, min_ent, max_ent = _read_temp_guard(u)

    Tout_bucket = int(round(Tout_raw))
    ctx = _context_key_for_outdoor(Tout_raw)

    if ctx not in _theta_by_unit_ctx[unit_name]:
        _theta_by_unit_ctx[unit_name][ctx] = [0.0, 0.0, 5.0, 0.0]
    if ctx not in _P_by_unit_ctx[unit_name]:
        _P_by_unit_ctx[unit_name][ctx] = [[P0, 0, 0, 0],
                                          [0, P0, 0, 0],
                                          [0, 0, P0, 0],
                                          [0, 0, 0, P0]]
    if ctx not in _updates_by_unit_ctx[unit_name]:
        _updates_by_unit_ctx[unit_name][ctx] = 0
    if ctx not in _freeze_until_by_ctx[unit_name]:
        _freeze_until_by_ctx[unit_name][ctx] = 0.0
    if ctx not in _exploration_last_ts[unit_name]:
        _exploration_last_ts[unit_name][ctx] = 0.0

    theta = _theta_by_unit_ctx[unit_name][ctx]
    P     = _P_by_unit_ctx[unit_name][ctx]

    # Theta anomaly guard: freeze learning for this ctx if abnormal
    if _theta_anomalous(theta):
        _freeze_until_by_ctx[unit_name][ctx] = max(float(_freeze_until_by_ctx[unit_name].get(ctx, 0.0) or 0.0), now + ANOMALY_FREEZE_SECONDS)
        theta = [0.0, 0.0, 5.0, 0.0]
        P = [[P0, 0, 0, 0],
             [0, P0, 0, 0],
             [0, 0, P0, 0],
             [0, 0, 0, P0]]
        _theta_by_unit_ctx[unit_name][ctx] = theta
        _P_by_unit_ctx[unit_name][ctx] = P
        _updates_by_unit_ctx[unit_name][ctx] = int(_updates_by_unit_ctx[unit_name].get(ctx, 0) or 0)
        _save_params_to_store(u)

    upd_count = int(_updates_by_unit_ctx[unit_name].get(ctx, 0) or 0)
    confidence, P_tr, P_norm, c_count, c_p = _confidence_from(upd_count, P)
    blend_factor = confidence
    if blend_factor < CONF_FREEZE_THRESHOLD:
        blend_factor = 0.0

    # confidence-driven step scaling
    step_scale = _confidence_step_scale(confidence)

    step_limit_auto, deadband_auto = _auto_tune_helpers(theta, unit_name, ctx, step_limit_current, deadband_current)
    step_limit = max(0.1, float(step_limit_auto) * float(step_scale))
    deadband = float(deadband_auto)

    global_upper = MAX_DEM if Tout_bucket <= -5 else GLOBAL_MILD_MAX

    try:
        icing_cap_base = float(state.get(ICING_CAP_HELPER) or ICING_BAND_CAP_DEFAULT)
    except Exception:
        icing_cap_base = ICING_BAND_CAP_DEFAULT
    if not isfinite(icing_cap_base):
        icing_cap_base = ICING_BAND_CAP_DEFAULT
    icing_cap_base = _clip(icing_cap_base, MIN_DEM, MAX_DEM)

    in_icing_band = (Tout_bucket >= ICING_BAND_MIN and Tout_bucket <= ICING_BAND_MAX)

    # Base icing risk
    risk, hum_risk, temp_risk = _icing_defrost_risk(Tout_raw, humidity_pct, in_icing_band)

    # Frost integrator (better defrost prediction)
    # Accumulate only if dt is reasonable (avoid trigger storms)
    frost_int = float(_frost_integrator.get(unit_name, 0.0) or 0.0)
    if dt >= FROST_INTEGRATOR_MIN_SECONDS:
        # decay
        decay = float(FROST_INTEGRATOR_DECAY_PER_H) * (dt / 3600.0)
        frost_int = max(0.0, frost_int - decay)

        # accumulate (only in icing band-ish)
        accum = float(FROST_INTEGRATOR_GAIN) * float(risk) * (dt / 3600.0)
        frost_int = _clip(frost_int + accum, 0.0, 10.0)

    _frost_integrator[unit_name] = frost_int

    defrost_predicted = (risk >= DEFROST_RISK_THRESHOLD) or (frost_int >= FROST_INTEGRATOR_THRESHOLD)

    # Dynamic icing cap (risk-based)
    if in_icing_band:
        cap_reduction = 1.0 - (DYNAMIC_CAP_MAX_REDUCTION_FRAC * risk)
        icing_cap_dynamic_raw = _clip(icing_cap_base * cap_reduction, MIN_DEM, icing_cap_base)
        band_upper_raw = min(global_upper, icing_cap_dynamic_raw)

        # Hysteresis: clamp quickly, release slowly back up
        prev_h = _cap_hyst_upper.get(unit_name)
        if prev_h is None or (not isfinite(float(prev_h))):
            band_upper = band_upper_raw
        else:
            # if new cap is lower => clamp immediately
            if band_upper_raw < float(prev_h):
                band_upper = band_upper_raw * CAP_HYST_CLAMP_FAST_FACTOR + float(prev_h) * (1.0 - CAP_HYST_CLAMP_FAST_FACTOR)
            else:
                # release slowly towards raw upper
                release = float(CAP_HYST_RELEASE_PER_H) * (dt / 3600.0) if dt > 0 else 0.0
                band_upper = float(prev_h) + release * (band_upper_raw - float(prev_h))
                band_upper = min(band_upper_raw, band_upper)  # never exceed raw target in one step
        _cap_hyst_upper[unit_name] = float(band_upper)
        icing_cap_dynamic = float(band_upper)  # for reporting
    else:
        icing_cap_dynamic = icing_cap_base
        band_upper = global_upper
        _cap_hyst_upper[unit_name] = None

    band_upper = _clip(float(band_upper), MIN_DEM, MAX_DEM)

    # If currently above cap, snap DOWN immediately (prevents "jump over cap")
    if prev > band_upper:
        if quiet_available:
            _set_quiet(QUIET_SWITCH, False, unit_name)
        option_cap = _snap_to_select(SELECT, band_upper, -1)
        if option_cap and option_cap != prev_str:
            select.select_option(entity_id=SELECT, option=option_cap)
        _last_resort_since[unit_name] = 0.0
        _last_demand_change_ts[unit_name] = now
        return

    prev_eff = prev if prev <= band_upper else band_upper

    # Control error uses ramped setpoint
    err = sp - Tin

    demand_norm = _clip(prev_eff / 100.0, 0.0, 1.0)
    hnorm = _humidity_norm(humidity_pct)
    x = [1.0, err, demand_norm, (Tout_raw - Tin) + (HUMIDITY_FEATURE_K * hnorm) + (WIND_FEATURE_K * wind_cold_factor)]
    y = rate

    # Steady-state gate
    since_dem_change = now - float(_last_demand_change_ts.get(unit_name, 0.0) or 0.0)
    since_sp_write = now - float(_last_setpoint_write_ts.get(unit_name, 0.0) or 0.0)
    steady_ok = (
        (since_dem_change >= STEADY_AFTER_CHANGE_SECONDS)
        and (since_sp_write >= STEADY_AFTER_CHANGE_SECONDS)
        and (abs(rate) <= RATE_STEADY_MAX_ABS)
        and (abs(err) <= ERR_STEADY_MAX_ABS)
    )

    # Per-ctx freeze timer (anomaly or auto-freeze event)
    freeze_until = float(_freeze_until_by_ctx[unit_name].get(ctx, 0.0) or 0.0)
    frozen_ctx = (now < freeze_until)

    allow_learning = (
        learning_enabled
        and (not defrosting)
        and (not in_cooldown)
        and (not rate_bad)
        and (not defrost_predicted)
        and steady_ok
        and (not frozen_ctx)
    )

    if allow_learning:
        theta_prev = theta[:]
        P_prev = [row[:] for row in P]
        theta_new, P_new = _rls_update(theta, P, x, y)
        if _all_finite(theta_new) and (not _theta_anomalous(theta_new)):
            theta = theta_new
            P = P_new
            _theta_by_unit_ctx[unit_name][ctx] = theta
            _P_by_unit_ctx[unit_name][ctx] = P
            _updates_by_unit_ctx[unit_name][ctx] = int(_updates_by_unit_ctx[unit_name].get(ctx, 0) or 0) + 1
            _save_params_to_store(u)
        else:
            # Freeze this ctx if learning produced garbage
            _theta_by_unit_ctx[unit_name][ctx] = theta_prev
            _P_by_unit_ctx[unit_name][ctx] = P_prev
            _freeze_until_by_ctx[unit_name][ctx] = max(float(_freeze_until_by_ctx[unit_name].get(ctx, 0.0) or 0.0), now + ANOMALY_FREEZE_SECONDS)
            _save_params_to_store(u)
    else:
        _theta_by_unit_ctx[unit_name][ctx] = theta
        _P_by_unit_ctx[unit_name][ctx] = P

    upd_count = int(_updates_by_unit_ctx[unit_name].get(ctx, 0) or 0)
    confidence, P_tr, P_norm, c_count, c_p = _confidence_from(upd_count, P)
    blend_factor = confidence
    if blend_factor < CONF_FREEZE_THRESHOLD:
        blend_factor = 0.0

    Tout_future = _avg_future_outdoor_from_hourly_forecast(FORECAST_HOURLY, OUTDOOR)
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

    # Efficiency proxy (optional): W per °C/h
    # If rate is °C/min -> convert to °C/h using *60; if it's already °C/h, still okay (just a metric).
    w_per_dth = None
    try:
        if power_w is not None and isfinite(power_w):
            dth = abs(float(rate)) * 60.0
            if dth > 0.05:
                w_per_dth = float(power_w) / float(dth)
    except Exception:
        w_per_dth = None

    # --- learned sensor update (attrs always updated) ---
    why = []
    _why_add(why, "fireplace_on" if fireplace_on else "")
    _why_add(why, "defrosting" if defrosting else "")
    _why_add(why, "cooldown" if in_cooldown else "")
    _why_add(why, "defrost_predicted" if defrost_predicted else "")
    _why_add(why, "steady_gate" if (not steady_ok) else "")
    _why_add(why, "ctx_frozen" if frozen_ctx else "")
    _why_add(why, "cap_limit" if (prev_eff >= band_upper - 1e-6) else "")
    _why_add(why, "conf_low" if (confidence < CONF_FREEZE_THRESHOLD) else "")

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
            outdoor=round(Tout_raw, 2),

            setpoint=round(sp, 2),
            setpoint_target=round(float(sp_target), 2),
            setpoint_base=round(float(base_sp), 2),
            setpoint_bias_delta=round(float(sp_delta), 3),
            setpoint_ramp_max_delta_per_tick=float(SETPOINT_RAMP_MAX_DELTA_C_PER_TICK),

            indoor=round(Tin, 2),
            err=round(float(err), 3),
            defrosting=bool(defrosting),
            cooldown=bool(in_cooldown),
            rate=round(float(rate), 5),

            learning_enabled=bool(learning_enabled),
            learning_allowed=bool(allow_learning),
            steady_ok=bool(steady_ok),

            fireplace_switch=str(FIREPLACE_SWITCH),
            fireplace_on=bool(fireplace_on),

            price_bias_points=round(float(price_bias), 2),
            price_bias_helper=str(PRICE_BIAS_HELPER),

            nordpool_avg_window_h=(round(float(nordpool_avg_window_h), 2) if nordpool_avg_window_h is not None else None),
            nordpool_avg_window_helper=str(NORDPOOL_AVG_WINDOW_HELPER),
            nordpool_avg_window_step_h=float(NORDPOOL_AVG_WINDOW_STEP_H),
            nordpool_severity_base=round(float(nordpool_sev_base), 4),
            nordpool_volatility_norm=round(float(vol_norm), 4),
            nordpool_volatility_std=(round(float(vol_std), 4) if isfinite(vol_std) else None),
            nordpool_severity=round(float(nordpool_sev), 4),
            price_sensor=str(PRICE_SENSOR),

            quiet_available=bool(quiet_available),

            temp_guard_min=min_temp,
            temp_guard_max=max_temp,
            temp_guard_min_helper=min_ent,
            temp_guard_max_helper=max_ent,

            confidence=round(float(confidence), 4),
            confidence_blend=round(float(blend_factor), 4),
            ctx_updates=int(upd_count),
            p_trace=round(float(P_tr), 2),
            p_norm=round(float(P_norm), 4),
            conf_count_factor=round(float(c_count), 4),
            conf_p_factor=round(float(c_p), 4),
            step_scale=round(float(step_scale), 4),

            humidity_pct=(round(float(humidity_pct), 1) if humidity_pct is not None else None),
            wind_speed=(round(float(wind_speed), 2) if wind_speed is not None else None),
            wind_norm=round(float(wnorm), 4),
            cold_factor=round(float(cold), 4),
            wind_cold_factor=round(float(wind_cold_factor), 4),

            icing_band=bool(in_icing_band),
            icing_risk=round(float(risk), 4),
            icing_risk_humidity=round(float(hum_risk), 4),
            icing_risk_temperature=round(float(temp_risk), 4),
            frost_integrator=round(float(frost_int), 4),
            defrost_predicted=bool(defrost_predicted),

            icing_cap_base=round(float(icing_cap_base), 2),
            icing_cap_dynamic=round(float(icing_cap_dynamic), 2),
            band_upper=round(float(band_upper), 2),

            outdoor_future=round(float(Tout_future), 2),
            forecast_hourly=str(FORECAST_HOURLY),

            power_sensor=str(POWER_SENSOR),
            power_w=(round(float(power_w), 2) if power_w is not None and isfinite(power_w) else None),
            comp_freq_sensor=str(COMP_FREQ_SENSOR),
            comp_hz=(round(float(comp_hz), 2) if comp_hz is not None and isfinite(comp_hz) else None),
            w_per_degC_per_h=(round(float(w_per_dth), 2) if w_per_dth is not None and isfinite(w_per_dth) else None),

            why=",".join(why),
            note=note,
        )
    except Exception as e:
        log.error("Daikin ML (%s): failed to update learned demand sensor: %s", unit_name, e)

    # ============================================================
    # HARD TEMPERATURE GUARD OVERRIDE ("no matter what")
    # ============================================================
    temp_override = None
    if isfinite(min_temp) and Tin < min_temp:
        temp_override = "below_min"
        dem_clip = min(100.0, float(band_upper))
    elif isfinite(max_temp) and Tin > max_temp:
        temp_override = "above_max"
        dem_clip = float(MIN_DEM)
    else:
        # normal blending
        if abs(err) <= deadband:
            dem_target = prev_eff
        else:
            dem_blend = prev_eff + float(blend_factor) * (float(dem_opt) - float(prev_eff))
            dem_target = dem_blend

        # never reduce demand when too cold
        if err > deadband and dem_target < prev_eff:
            dem_target = prev_eff

        # cooldown / step limiting
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

        # gentle exploration (only when safe, low confidence, near setpoint, not defrost/cooldown, not cap-limited)
        if EXPLORATION_ENABLED and (temp_override is None):
            if (confidence <= EXPLORATION_MIN_CONF) and (abs(err) <= EXPLORATION_ERR_MAX) and (not defrosting) and (not in_cooldown) and (not defrost_predicted) and (dem_clip < band_upper - 0.5):
                last_exp = float(_exploration_last_ts[unit_name].get(ctx, 0.0) or 0.0)
                if (now - last_exp) >= EXPLORATION_MIN_SECONDS:
                    phase = int(_exploration_phase.get(unit_name, 1) or 1)
                    d = float(EXPLORATION_MAX_DELTA_DEM) * (1.0 if phase >= 0 else -1.0)
                    dem_clip = _clip(dem_clip + d, MIN_DEM, min(band_upper, 100.0))
                    _exploration_last_ts[unit_name][ctx] = now
                    _exploration_phase[unit_name] = -phase

    # REAL 100% last resort gating (only with quiet)
    allow_real_100 = True
    want_full = (dem_clip >= 99.95) and (band_upper >= 100.0)

    if quiet_available and (temp_override is None):
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
    else:
        _last_resort_since[unit_name] = 0.0

    # snapping DOWN when cap-limited
    cap_limited = (dem_clip >= (float(band_upper) - 1e-9))
    snap_dir = -1 if cap_limited else 0

    # Apply demand
    if quiet_available:
        option = _apply_demand_with_quiet(unit_name, SELECT, QUIET_SWITCH, dem_clip, prev_str, allow_real_100, direction=snap_dir)
    else:
        option = _apply_demand(unit_name, SELECT, dem_clip, prev_str, direction=snap_dir)

    # If demand changed, update steady-state timer
    if option and option != prev_str:
        _last_demand_change_ts[unit_name] = now

    log.info(
        "Daikin ML (%s): ctx=%s Tin=%.2f Tout=%.2f Tout_f=%.2f hum=%s wind=%s wcf=%.3f sev=%.3f(vol=%.3f std=%s) avg_h=%s(step=%.2f) "
        "risk=%.3f frost=%.3f sp=%.2f target=%.2f base=%.2f d=%.3f err=%.2f | fireplace=%s steady=%s freeze=%s | conf=%.3f blend=%.3f step_scale=%.2f upd=%s | "
        "cap_base=%.1f cap_dyn=%.1f band_upper=%.1f def_pred=%s | power=%sW comp=%sHz w/degC/h=%s | dem_opt=%.1f -> dem_clip=%.1f -> %s",
        unit_name, ctx, Tin, Tout_raw, float(Tout_future),
        str(humidity_pct), str(wind_speed), float(wind_cold_factor),
        float(nordpool_sev), float(vol_norm), (str(round(vol_std, 4)) if isfinite(vol_std) else "None"),
        str(nordpool_avg_window_h), float(NORDPOOL_AVG_WINDOW_STEP_H),
        float(risk), float(frost_int),
        float(sp), float(sp_target), float(base_sp), float(sp_delta), err,
        str(fireplace_on), str(steady_ok), str(frozen_ctx),
        float(confidence), float(blend_factor), float(step_scale), str(upd_count),
        float(icing_cap_base), float(icing_cap_dynamic), float(band_upper), str(defrost_predicted),
        (str(round(power_w, 1)) if power_w is not None and isfinite(power_w) else "None"),
        (str(round(comp_hz, 2)) if comp_hz is not None and isfinite(comp_hz) else "None"),
        (str(round(w_per_dth, 2)) if w_per_dth is not None and isfinite(w_per_dth) else "None"),
        float(dem_opt), float(dem_clip), str(option),
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
    global _theta_by_unit_ctx, _P_by_unit_ctx, _params_loaded, _last_defrosting, _cooldown_until, _last_resort_since, _updates_by_unit_ctx
    global _frost_integrator, _last_tick_ts, _last_demand_change_ts, _last_setpoint_write_ts, _last_sp_written, _cap_hyst_upper
    global _freeze_until_by_ctx, _exploration_last_ts, _exploration_phase

    for u in DAIKINS:
        unit_name = u["name"]
        _init_unit_if_needed(unit_name)

        _theta_by_unit_ctx[unit_name] = {}
        _P_by_unit_ctx[unit_name] = {}
        _params_loaded[unit_name] = False
        _last_defrosting[unit_name] = None
        _cooldown_until[unit_name] = 0.0
        _last_resort_since[unit_name] = 0.0
        _updates_by_unit_ctx[unit_name] = {}

        _frost_integrator[unit_name] = 0.0
        _last_tick_ts[unit_name] = 0.0
        _last_demand_change_ts[unit_name] = 0.0
        _last_setpoint_write_ts[unit_name] = 0.0
        _last_sp_written[unit_name] = None
        _cap_hyst_upper[unit_name] = None
        _freeze_until_by_ctx[unit_name] = {}
        _exploration_last_ts[unit_name] = {}
        _exploration_phase[unit_name] = 1

        try:
            state.set(u["STORE_ENTITY"], value=time.time(), theta_by_ctx={}, updates_by_ctx={}, freeze_until_by_ctx={})
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
    except Exception as e:
        log.error("Daikin ML: failed to enable learning global: %s", e)

@service
def daikin_ml_learning_disable():
    """Disable learning globally."""
    try:
        input_boolean.turn_off(entity_id=LEARNING_ENABLED_HELPER_GLOBAL)
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
        else:
            input_boolean.turn_on(entity_id=LEARNING_ENABLED_HELPER_GLOBAL)
    except Exception as e:
        log.error("Daikin ML: failed to toggle learning global: %s", e)


# ============================================================
# 8) PERSIST STORES AT LOAD
# ============================================================
_persist_all_stores()
