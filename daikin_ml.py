# pyscript/daikin_ml_multi.py
# Online-RLS (oppiva) ohjain Daikin demand-selectille, MULTI-DAIKIN -tuella.

import time
from math import isfinite

# ============================================================
# 1) LAITEKONFIGURAATIOT (LISÄÄ TÄHÄN UUSIA DAIKINEITA)
# ============================================================
DAIKINS = [
    {
        "name": "daikin1",

        "INDOOR": "sensor.apollo_temp_1_96a244_board_temperature",
        "INDOOR_RATE": "sensor.apollo_temp_1_derivative",
        "OUTDOOR": "sensor.iv_tulo_lampotila",
        "WEATHER": "weather.koti",

        "SELECT": "select.faikin_demand_control",
        "LIQUID": "sensor.faikin_liquid",

        # Quiet outdoor mode switch (Faikin)
        "QUIET_OUTDOOR_SWITCH": "switch.faikin_quiet_outdoor",

        "SP_HELPER": "input_number.daikin_setpoint",
        "STEP_LIMIT_HELPER": "input_number.daikin_step_limit",
        "DEADBAND_HELPER": "input_number.daikin_deadband",
        "ICING_CAP_HELPER": "input_number.daikin_icing_cap",

        # Manual base setpoint (user controls this)
        "SP_BASE_HELPER": "input_number.daikin_setpoint_base",

        # Nordpool bias points (written by nordpool_15m_bias.py)
        "PRICE_BIAS_HELPER": "input_number.daikin1_price_bias_points",

        # Nordpool bias enable/disable
        "PRICE_BIAS_ENABLED": "input_boolean.nordpool_bias_enabled",

        # Low/high setpoint guards (per unit)
        "MIN_TEMP_GUARD_HELPER": "input_number.daikin1_min_temp_guard",
        "MAX_TEMP_GUARD_HELPER": "input_number.daikin1_max_temp_guard",

        # Persistent store per laite (suositus: eri entity per daikin)
        "STORE_ENTITY": "pyscript.daikin1_ml_params",

        # Learned-demand sensori per laite
        "LEARNED_SENSOR": "sensor.daikin1_ml_learned_demand",

        # ------------------------------------------------------------
        # Minimum demand floors by outdoor temperature band
        # (Hard floor enforced regardless of setpoint vs indoor temp)
        # Bands are in whole-degree buckets (int(round(Tout_raw))).
        # ------------------------------------------------------------
        "MIN_DEM_FLOOR_M05_M10": "input_number.daikin1_min_dem_m05_m10",  # -5 .. -10
        "MIN_DEM_FLOOR_M11_M15": "input_number.daikin1_min_dem_m11_m15",  # -11 .. -15
        "MIN_DEM_FLOOR_LE_M16":  "input_number.daikin1_min_dem_le_m16",   # <= -16

        # ------------------------------------------------------------
        # NEW: Maximum demand caps by outdoor temperature band
        # (Hard cap enforced regardless of setpoint vs indoor temp)
        # ------------------------------------------------------------
        "MAX_DEM_CAP_M05_M10": "input_number.daikin1_max_dem_m05_m10",  # -5 .. -10
        "MAX_DEM_CAP_M11_M15": "input_number.daikin1_max_dem_m11_m15",  # -11 .. -15
        "MAX_DEM_CAP_LE_M16":  "input_number.daikin1_max_dem_le_m16",   # <= -16

        # ------------------------------------------------------------
        # NEW: Per-unit minimum interval between *applied* demand changes (seconds)
        # ------------------------------------------------------------
        "DEMAND_CHANGE_MIN_INTERVAL_HELPER": "input_number.daikin1_demand_change_min_interval_s",
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
    #     "QUIET_OUTDOOR_SWITCH": "switch.toinen_daikin_quiet_outdoor",
    #     "SP_HELPER": "input_number.daikin2_setpoint",
    #     "STEP_LIMIT_HELPER": "input_number.daikin2_step_limit",
    #     "DEADBAND_HELPER": "input_number.daikin2_deadband",
    #     "ICING_CAP_HELPER": "input_number.daikin2_icing_cap",
    #     "SP_BASE_HELPER": "input_number.daikin2_setpoint_base",
    #     "PRICE_BIAS_HELPER": "input_number.daikin2_price_bias_points",
    #     "PRICE_BIAS_ENABLED": "input_boolean.nordpool_bias_enabled",
    #     "MIN_TEMP_GUARD_HELPER": "input_number.daikin2_min_temp_guard",
    #     "MAX_TEMP_GUARD_HELPER": "input_number.daikin2_max_temp_guard",
    #     "STORE_ENTITY": "pyscript.daikin2_ml_params",
    #     "LEARNED_SENSOR": "sensor.daikin2_ml_learned_demand",
    #
    #     "MIN_DEM_FLOOR_M05_M10": "input_number.daikin2_min_dem_m05_m10",
    #     "MIN_DEM_FLOOR_M11_M15": "input_number.daikin2_min_dem_m11_m15",
    #     "MIN_DEM_FLOOR_LE_M16":  "input_number.daikin2_min_dem_le_m16",
    #
    #     "MAX_DEM_CAP_M05_M10": "input_number.daikin2_max_dem_m05_m10",
    #     "MAX_DEM_CAP_M11_M15": "input_number.daikin2_max_dem_m11_m15",
    #     "MAX_DEM_CAP_LE_M16":  "input_number.daikin2_max_dem_le_m16",
    #
    #     "DEMAND_CHANGE_MIN_INTERVAL_HELPER": "input_number.daikin2_demand_change_min_interval_s",
    # },
]

# ============================================================
# 2) YHTEISET SÄÄTÖVAKIOT (KUTEN ENNEN)
# ============================================================
HORIZON_H  = 1.0
FORECAST_H = 6
LAMBDA     = 0.995
P0         = 1e4
MIN_DEM    = 30.0
MAX_DEM    = 100.0

# Extra "layer" for max demand using quiet outdoor switch:
#  - Demand 100 with quiet_outdoor ON  => effective demand 100
#  - Demand 100 with quiet_outdoor OFF => effective demand 105
MAX_DEM_LAYER = 105.0
QUIET_LAYER_EXTRA = 5.0
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

# Soft landing near setpoint: when indoor temperature approaches setpoint,
# slow down demand changes to avoid harsh overshoot/oscillation.
SOFT_ERR_START = 0.2   # °C: start softening when |err| <= this AND moving toward setpoint
SOFT_ERR_END   = 0.0   # °C: strongest softening when |err| <= this
SOFT_STEP_MIN  = 1.0   # %: minimum step change allowed near setpoint
SOFT_APPROACH_EPS = 0.01  # °C: require this much progress toward setpoint to count as 'approaching'

# ------------------------------------------------------------
# Efficiency behavior inside deadband
# Trim down demand while inside ±deadband to find lowest sustaining demand.
# ------------------------------------------------------------
EFF_TRIM_STEP = 2.0         # demand-% per tick inside deadband
EFF_TRIM_STEP_FAST = 5.0    # demand-% per tick when above setpoint (err < 0)

# ------------------------------------------------------------
# NEW: Default minimum interval between applied demand changes (seconds)
# This is overridden per-unit by DEMAND_CHANGE_MIN_INTERVAL_HELPER if present.
# ------------------------------------------------------------
DEMAND_CHANGE_MIN_INTERVAL_DEFAULT_S = 60.0

# ------------------------------------------------------------
# NEW: Learning uses smoothed 5-minute slope of indoor temperature
# - Keeps a per-unit history of (timestamp, Tin)
# - If not enough history/span, falls back to the derivative sensor
# ------------------------------------------------------------
TIN_SLOPE_WINDOW_S = 5 * 60.0
TIN_SLOPE_MIN_SPAN_S = 4 * 60.0  # require at least ~4 minutes span for "true 5-minute" slope
TIN_SLOPE_MIN_SAMPLES = 3        # require a few points for smoothing/robustness

# ------------------------------------------------------------
# Defrost hold behavior
# - Freeze to pre-defrost demand during defrost
# - Keep same demand for 5 minutes after defrost ends
# - Do not collect/use Tin measurements until hold period ends
# ------------------------------------------------------------
DEFROST_LIQUID_THRESHOLD = 20.0
POST_DEFROST_HOLD_S = 5 * 60.0

# ============================================================
# Nordpool -> Setpoint integration
# ============================================================
SP_BASE_DEFAULT = 22.5

# Convert "bias points" (typically -20..+20) into °C setpoint shift.
SP_BIAS_DEGC_PER_POINT = 0.05

# Clamp applied setpoint shift (°C)
SP_BIAS_CLAMP_MIN = -0.5
SP_BIAS_CLAMP_MAX = +0.5

# Defaults for guards if helpers missing/unset
MIN_GUARD_DEFAULT = 16.0
MAX_GUARD_DEFAULT = 28.0

# Don’t spam writes for tiny changes
SP_WRITE_EPS = 0.01

# Nordpool avg window hours (dynamic) -> bias strength scaling
NORDPOOL_AVG_WINDOW_HELPER = "input_number.nordpool_avg_window_hours"
# Reference window used for scaling. Shorter window => stronger bias, longer => weaker.
AVG_WINDOW_REF_H = 12.0
AVG_WINDOW_FACTOR_MIN = 0.5
AVG_WINDOW_FACTOR_MAX = 2.0

# Hourly outdoor forecast sensor (attributes: forecast: [{temperature: ...}, ...])
WEATHER_FORECAST_HOURLY_SENSOR = "sensor.weather_forecast_hourly"

# Dynamic mapping: colder -> shorter Nordpool averaging window (hours)
AVG_WINDOW_MIN_H = 1.0
AVG_WINDOW_MAX_H = 4.0
AVG_WINDOW_TEMP_COLD = -10.0   # at / below this => AVG_WINDOW_MIN_H
AVG_WINDOW_TEMP_WARM = 5.0     # at / above this => AVG_WINDOW_MAX_H
AVG_WINDOW_WRITE_EPS = 0.1     # don't spam helper writes


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

# ------------------------------------------------------------
# Read float helper + enforce min/max demand limits by outdoor band
# ------------------------------------------------------------
def _read_float_entity(entity_id, default):
    try:
        v = float(state.get(entity_id))
        return v if isfinite(v) else default
    except Exception:
        return default

def _min_demand_floor_for_outdoor(u, Tout_bucket):
    """
    Returns the enforced minimum demand floor based on outdoor temperature bucket.

    Bands (in whole-degree buckets, using int(round(Tout_raw))):
      -10..-5   -> MIN_DEM_FLOOR_M05_M10
      -15..-11  -> MIN_DEM_FLOOR_M11_M15
      <= -16    -> MIN_DEM_FLOOR_LE_M16
    """
    floor = MIN_DEM

    ent_05_10 = u.get("MIN_DEM_FLOOR_M05_M10")
    ent_11_15 = u.get("MIN_DEM_FLOOR_M11_M15")
    ent_le_16 = u.get("MIN_DEM_FLOOR_LE_M16")

    if -10 <= Tout_bucket <= -5 and ent_05_10:
        floor = _read_float_entity(ent_05_10, floor)
    elif -15 <= Tout_bucket <= -11 and ent_11_15:
        floor = _read_float_entity(ent_11_15, floor)
    elif Tout_bucket <= -16 and ent_le_16:
        floor = _read_float_entity(ent_le_16, floor)

    floor = _clip(floor, 0.0, 100.0)
    floor = max(MIN_DEM, floor)
    return floor

def _max_demand_cap_for_outdoor(u, Tout_bucket, default_cap):
    """
    Returns the enforced maximum demand cap based on outdoor temperature bucket.

    Bands (in whole-degree buckets, using int(round(Tout_raw))):
      -10..-5   -> MAX_DEM_CAP_M05_M10
      -15..-11  -> MAX_DEM_CAP_M11_M15
      <= -16    -> MAX_DEM_CAP_LE_M16

    default_cap should already reflect other caps (global/icing/etc) for the current conditions.
    """
    cap = float(default_cap)

    ent_05_10 = u.get("MAX_DEM_CAP_M05_M10")
    ent_11_15 = u.get("MAX_DEM_CAP_M11_M15")
    ent_le_16 = u.get("MAX_DEM_CAP_LE_M16")

    if -10 <= Tout_bucket <= -5 and ent_05_10:
        cap = _read_float_entity(ent_05_10, cap)
    elif -15 <= Tout_bucket <= -11 and ent_11_15:
        cap = _read_float_entity(ent_11_15, cap)
    elif Tout_bucket <= -16 and ent_le_16:
        cap = _read_float_entity(ent_le_16, cap)

    cap = _clip(cap, 0.0, 100.0)
    cap = max(MIN_DEM, cap)
    return cap

def _demand_change_min_interval_s(u):
    ent = u.get("DEMAND_CHANGE_MIN_INTERVAL_HELPER")
    if ent:
        v = _read_float_entity(ent, DEMAND_CHANGE_MIN_INTERVAL_DEFAULT_S)
    else:
        v = DEMAND_CHANGE_MIN_INTERVAL_DEFAULT_S
    # clamp to something sane: 0..600s
    return _clip(v, 0.0, 600.0)


# ============================================================
# Nordpool avg window dynamic update (based on hourly forecast)
# ============================================================
def _read_hourly_forecast_temps(forecast_sensor, hours=FORECAST_H):
    """
    Reads hourly forecast temperatures from an entity like sensor.weather_forecast_hourly
    whose attributes include: forecast: [{temperature: ...}, ...]
    Returns list[float] length <= hours.
    """
    attrs = state.getattr(forecast_sensor) or {}
    fc = attrs.get("forecast") or []
    temps = []
    for i in range(min(hours, len(fc))):
        t = fc[i].get("temperature")
        if t is None:
            continue
        try:
            tv = float(t)
        except Exception:
            continue
        if isfinite(tv):
            temps.append(tv)
    return temps

def _compute_outdoor_effective_temp_for_window(outdoor_entity):
    """
    Conservative effective outdoor temperature for window sizing.
    Uses min(current_outdoor, avg_forecast_next_hours) when possible so that
    if it's going to get colder soon, window shortens in advance.
    """
    Tout_cur = float("nan")
    try:
        Tout_cur = float(state.get(outdoor_entity))
    except Exception:
        Tout_cur = float("nan")

    temps = _read_hourly_forecast_temps(WEATHER_FORECAST_HOURLY_SENSOR, hours=FORECAST_H)
    Tout_fc = float("nan")
    if temps:
        Tout_fc = sum(temps) / float(len(temps))

    candidates = []
    if isfinite(Tout_cur):
        candidates.append(Tout_cur)
    if isfinite(Tout_fc):
        candidates.append(Tout_fc)

    if not candidates:
        return 0.0

    return min(candidates)

def _temp_to_avg_window_hours(temp_c):
    """
    Linear map: AVG_WINDOW_TEMP_COLD..AVG_WINDOW_TEMP_WARM -> AVG_WINDOW_MIN_H..AVG_WINDOW_MAX_H
    Colder -> shorter window.
    """
    t = _clip(temp_c, AVG_WINDOW_TEMP_COLD, AVG_WINDOW_TEMP_WARM)
    span_t = (AVG_WINDOW_TEMP_WARM - AVG_WINDOW_TEMP_COLD)
    if span_t <= 0:
        return AVG_WINDOW_REF_H
    frac = (t - AVG_WINDOW_TEMP_COLD) / span_t  # 0..1
    hours = AVG_WINDOW_MIN_H + frac * (AVG_WINDOW_MAX_H - AVG_WINDOW_MIN_H)
    return _clip(hours, AVG_WINDOW_MIN_H, AVG_WINDOW_MAX_H)

def _update_nordpool_avg_window_hours(outdoor_entity):
    """
    Compute and write input_number.nordpool_avg_window_hours based on forecast+outdoor.
    """
    try:
        temp_eff = _compute_outdoor_effective_temp_for_window(outdoor_entity)
        new_h = float(_temp_to_avg_window_hours(temp_eff))
        prev_h = float(state.get(NORDPOOL_AVG_WINDOW_HELPER) or new_h)
        if (not isfinite(prev_h)) or abs(prev_h - new_h) > AVG_WINDOW_WRITE_EPS:
            input_number.set_value(entity_id=NORDPOOL_AVG_WINDOW_HELPER, value=round(new_h, 2))
    except Exception as e:
        try:
            log.debug("Daikin ML: failed to update nordpool avg window: %s", e)
        except Exception:
            pass


# ============================================================
# 4) PER-LAITE TILA (THETA/P + DEFROST-COOLDOWN + RATE LIMIT + 5min slope)
# ============================================================
_theta_by_unit_ctx = {}   # unit -> ctx -> theta[4]
_P_by_unit_ctx     = {}   # unit -> ctx -> P[4x4]
_params_loaded     = {}   # unit -> bool

_last_defrosting   = {}   # unit -> bool/None
_cooldown_until    = {}   # unit -> epoch float

# Track previous control error per unit (for soft-landing approach detection)
_prev_err          = {}   # unit -> last err (float)

# per-unit demand-change rate limiter state
_last_demand_change_ts = {}  # unit -> epoch float of last *applied* change
_last_demand_sig       = {}  # unit -> signature tuple

# per-unit Tin history for 5-minute smoothed slope
_tin_hist = {}  # unit -> list[(ts, Tin)]

# Hold demand through defrost and for 5 minutes after defrost ends
_hold_until = {}          # unit -> epoch float (0 if inactive)
_held_select_option = {}  # unit -> string option, e.g. "70%" (or "70")
_held_quiet_on = {}       # unit -> bool/None


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
    if unit_name not in _prev_err:
        _prev_err[unit_name] = None
    if unit_name not in _last_demand_change_ts:
        _last_demand_change_ts[unit_name] = 0.0
    if unit_name not in _last_demand_sig:
        _last_demand_sig[unit_name] = None
    if unit_name not in _tin_hist:
        _tin_hist[unit_name] = []
    if unit_name not in _hold_until:
        _hold_until[unit_name] = 0.0
    if unit_name not in _held_select_option:
        _held_select_option[unit_name] = None
    if unit_name not in _held_quiet_on:
        _held_quiet_on[unit_name] = None

def _tin_hist_add(unit_name, ts, tin):
    if not (isfinite(ts) and isfinite(tin)):
        return
    hist = _tin_hist.get(unit_name)
    if hist is None:
        hist = []
        _tin_hist[unit_name] = hist
    hist.append((float(ts), float(tin)))

    # prune window + a small extra margin
    cutoff = float(ts) - (TIN_SLOPE_WINDOW_S + 30.0)
    new_hist = []
    for t, v in hist:
        if isfinite(t) and isfinite(v) and t >= cutoff:
            new_hist.append((t, v))
    if len(new_hist) > 200:
        new_hist = new_hist[-200:]
    _tin_hist[unit_name] = new_hist

def _tin_smoothed_rate_5min(unit_name, now_ts):
    """
    Returns (rate_degC_per_hour, used_smoothed, span_s, n_samples)
    Uses linear regression slope over last ~5 minutes for smoothing.
    """
    hist = _tin_hist.get(unit_name) or []
    if len(hist) < TIN_SLOPE_MIN_SAMPLES:
        return 0.0, False, 0.0, len(hist)

    cutoff = float(now_ts) - TIN_SLOPE_WINDOW_S
    pts = [(t, v) for (t, v) in hist if t >= cutoff]
    if len(pts) < TIN_SLOPE_MIN_SAMPLES:
        return 0.0, False, 0.0, len(pts)

    t0 = pts[0][0]
    t1 = pts[-1][0]
    span = float(t1 - t0)
    if span < TIN_SLOPE_MIN_SPAN_S:
        return 0.0, False, span, len(pts)

    ts = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    t_mean = sum(ts) / float(len(ts))
    v_mean = sum(vs) / float(len(vs))

    num = 0.0
    den = 0.0
    for t, v in pts:
        dt = (t - t_mean)
        dv = (v - v_mean)
        num += dt * dv
        den += dt * dt

    if (not isfinite(den)) or den <= 1e-9 or (not isfinite(num)):
        return 0.0, False, span, len(pts)

    slope_degC_per_s = num / den
    if not isfinite(slope_degC_per_s):
        return 0.0, False, span, len(pts)

    rate_h = slope_degC_per_s * 3600.0
    if not isfinite(rate_h):
        return 0.0, False, span, len(pts)

    return float(rate_h), True, span, len(pts)

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


def _read_defrosting(u):
    """Return (defrosting_bool, liquid_float)."""
    LIQUID = u["LIQUID"]
    try:
        liquid = float(state.get(LIQUID) or 100.0)
    except Exception:
        liquid = 100.0
    if not isfinite(liquid):
        liquid = 100.0
    return (liquid < DEFROST_LIQUID_THRESHOLD), liquid

def _update_tin_history_only(u):
    """
    Called from state_trigger runs:
    - only update Tin history (for 5-minute smoothed slope)
    - do NOT publish learned demand or apply control
    - during defrost or post-defrost hold: do not collect Tin at all
    """
    unit_name = u["name"]
    _init_unit_if_needed(unit_name)

    now = time.time()
    defrosting, _liq = _read_defrosting(u)
    hold_active = (now < float(_hold_until.get(unit_name) or 0.0))

    if defrosting or hold_active:
        return

    INDOOR = u["INDOOR"]
    try:
        Tin = float(state.get(INDOOR))
    except Exception:
        Tin = float("nan")
    if isfinite(Tin):
        _tin_hist_add(unit_name, now, Tin)

# ============================================================
# 5) STARTUP + TRIGGERIT
# ============================================================
_persist_all_stores()

# Build a single pyscript trigger expression string: "sensor.a or sensor.b"
_INDOOR_TRIGGERS = []
for u in DAIKINS:
    ent = u.get("INDOOR")
    if ent and isinstance(ent, str):
        _INDOOR_TRIGGERS.append(ent)

_seen = set()
_INDOOR_TRIGGERS = [x for x in _INDOOR_TRIGGERS if not (x in _seen or _seen.add(x))]
_INDOOR_TRIGGER_EXPR = " or ".join(_INDOOR_TRIGGERS) if _INDOOR_TRIGGERS else None

# ------------------------------------------------------------
# FIX (multi-unit cross-talk):
# identify which INDOOR entity triggered this run (if state_trigger)
# ------------------------------------------------------------
def _get_trigger_entity_from_kwargs(kwargs):
    """
    Pyscript versions differ; try common keys and return an entity_id like 'sensor.xxx'.
    If not found, return None (e.g., cron run).
    """
    for k in ("entity_id", "var_name", "trigger_entity", "trigger_var", "trigger"):
        v = kwargs.get(k)
        if isinstance(v, str) and "." in v:
            return v
    return None

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

# Run on every INDOOR change (collect Tin history only) + periodic cron (full control + publish learned demand)
if _INDOOR_TRIGGER_EXPR:
    @time_trigger("cron(*/5 * * * *)")  # TRUE 5-minute cadence
    @state_trigger(_INDOOR_TRIGGER_EXPR)
    def daikin_ml_controller(**kwargs):

        # Update Nordpool averaging window hours dynamically based on outdoor forecast (colder -> shorter)
        try:
            temps_eff = []
            for uu in DAIKINS:
                oe = uu.get("OUTDOOR")
                if oe:
                    try:
                        temps_eff.append(_compute_outdoor_effective_temp_for_window(oe))
                    except Exception:
                        pass
            if temps_eff:
                temp_eff = min(temps_eff)
                new_h = float(_temp_to_avg_window_hours(temp_eff))
                prev_h = float(state.get(NORDPOOL_AVG_WINDOW_HELPER) or new_h)
                if (not isfinite(prev_h)) or abs(prev_h - new_h) > AVG_WINDOW_WRITE_EPS:
                    input_number.set_value(entity_id=NORDPOOL_AVG_WINDOW_HELPER, value=round(new_h, 2))
        except Exception as e:
            try:
                log.debug("Daikin ML: avg window update failed: %s", e)
            except Exception:
                pass


        trig_ent = _get_trigger_entity_from_kwargs(kwargs)
        is_cron = (trig_ent is None)

        # If this was triggered by an indoor sensor update, ONLY update Tin history and exit.
        if not is_cron and trig_ent:
            for uu in DAIKINS:
                if uu.get("INDOOR") == trig_ent:
                    try:
                        _update_tin_history_only(uu)
                    except Exception as e:
                        log.debug("Daikin ML (%s): Tin history update failed: %s", uu.get("name", "?"), e)
                    break
            return

        # Cron run: run full controller and publish learned demand (and apply control) for all units.
        for u in DAIKINS:
            try:
                _run_one_unit(u)
            except Exception as e:
                log.error("Daikin ML (%s): controller error: %s", u.get("name", "?"), e)
else:
    @time_trigger("cron(*/5 * * * *)")  # TRUE 5-minute cadence
    def daikin_ml_controller(**kwargs):

        # Update Nordpool averaging window hours dynamically based on outdoor forecast (colder -> shorter)
        try:
            temps_eff = []
            for uu in DAIKINS:
                oe = uu.get("OUTDOOR")
                if oe:
                    try:
                        temps_eff.append(_compute_outdoor_effective_temp_for_window(oe))
                    except Exception:
                        pass
            if temps_eff:
                temp_eff = min(temps_eff)
                new_h = float(_temp_to_avg_window_hours(temp_eff))
                prev_h = float(state.get(NORDPOOL_AVG_WINDOW_HELPER) or new_h)
                if (not isfinite(prev_h)) or abs(prev_h - new_h) > AVG_WINDOW_WRITE_EPS:
                    input_number.set_value(entity_id=NORDPOOL_AVG_WINDOW_HELPER, value=round(new_h, 2))
        except Exception as e:
            try:
                log.debug("Daikin ML: avg window update failed: %s", e)
            except Exception:
                pass

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
    _init_unit_if_needed(unit_name)

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

    try:
        Tin = float(state.get(INDOOR))
    except Exception:
        Tin = float("nan")
    try:
        Tout_raw = float(state.get(OUTDOOR))
    except Exception:
        Tout_raw = float("nan")

    now = time.time()


    # Tin history is appended only when not defrosting and not in post-defrost hold.

    # ------------------------------------------------------------
    # NEW: Learning rate uses smoothed 5-minute Tin slope.
    # If not enough history yet, fall back to the derivative sensor.
    # ------------------------------------------------------------
    rate = 0.0
    rate_bad = False
    rate_source = "unknown"
    rate_span_s = 0.0
    rate_n = 0

    # ----------------------------
    # Effective setpoint = base + (optional Nordpool bias), clamped by guards
    # ----------------------------
    SP_BASE_HELPER = u.get("SP_BASE_HELPER", "input_number.daikin_setpoint_base")
    PRICE_BIAS_HELPER = u.get("PRICE_BIAS_HELPER")
    PRICE_BIAS_ENABLED = u.get("PRICE_BIAS_ENABLED", "input_boolean.nordpool_bias_enabled")
    MIN_TEMP_GUARD_HELPER = u.get("MIN_TEMP_GUARD_HELPER")
    MAX_TEMP_GUARD_HELPER = u.get("MAX_TEMP_GUARD_HELPER")

    # 1) Read manual base setpoint (user-controlled)
    try:
        sp_base = float(state.get(SP_BASE_HELPER) or SP_BASE_DEFAULT)
    except Exception:
        sp_base = SP_BASE_DEFAULT
    if not isfinite(sp_base):
        sp_base = SP_BASE_DEFAULT

    # 2) Check Nordpool bias enable
    bias_enabled = (state.get(PRICE_BIAS_ENABLED) == "on")

    # 3) Read bias points (if enabled)
    if bias_enabled and PRICE_BIAS_HELPER:
        try:
            bias_points = float(state.get(PRICE_BIAS_HELPER) or 0.0)
        except Exception:
            bias_points = 0.0
        if not isfinite(bias_points):
            bias_points = 0.0
    else:
        bias_points = 0.0

    # 4) Read dynamic averaging window hours and convert to a scaling factor
    try:
        avg_window_h = float(state.get(NORDPOOL_AVG_WINDOW_HELPER) or AVG_WINDOW_REF_H)
    except Exception:
        avg_window_h = AVG_WINDOW_REF_H
    if (not isfinite(avg_window_h)) or (avg_window_h <= 0.0):
        avg_window_h = AVG_WINDOW_REF_H

    # Shorter averaging window (colder) => stronger response; longer => weaker
    avg_window_factor = _clip(AVG_WINDOW_REF_H / avg_window_h, AVG_WINDOW_FACTOR_MIN, AVG_WINDOW_FACTOR_MAX)

    # 5) Convert bias points -> °C shift, apply window scaling, then clamp
    sp_bias_degC = _clip(bias_points * SP_BIAS_DEGC_PER_POINT * avg_window_factor, SP_BIAS_CLAMP_MIN, SP_BIAS_CLAMP_MAX)

    # Effective setpoint
    sp = sp_base + sp_bias_degC

    # 6) Apply min/max guard clamps (per unit)
    try:
        sp_min_guard = float(state.get(MIN_TEMP_GUARD_HELPER) or MIN_GUARD_DEFAULT) if MIN_TEMP_GUARD_HELPER else MIN_GUARD_DEFAULT
    except Exception:
        sp_min_guard = MIN_GUARD_DEFAULT
    try:
        sp_max_guard = float(state.get(MAX_TEMP_GUARD_HELPER) or MAX_GUARD_DEFAULT) if MAX_TEMP_GUARD_HELPER else MAX_GUARD_DEFAULT
    except Exception:
        sp_max_guard = MAX_GUARD_DEFAULT

    if not isfinite(sp_min_guard):
        sp_min_guard = MIN_GUARD_DEFAULT
    if not isfinite(sp_max_guard):
        sp_max_guard = MAX_GUARD_DEFAULT

    if sp_min_guard > sp_max_guard:
        sp_min_guard, sp_max_guard = sp_max_guard, sp_min_guard

    sp = _clip(sp, sp_min_guard, sp_max_guard)

    # 7) Write effective setpoint into SP_HELPER (input_number.daikin_setpoint)
    try:
        sp_prev = float(state.get(SP_HELPER) or sp)
        if (not isfinite(sp_prev)) or abs(sp_prev - sp) > SP_WRITE_EPS:
            input_number.set_value(entity_id=SP_HELPER, value=round(float(sp), 2))
    except Exception as e:
        log.debug("Could not write effective setpoint to %s err=%s", SP_HELPER, e)

    step_limit_current = float(state.get(STEP_LIMIT_HELPER) or 10.0)
    deadband_current   = float(state.get(DEADBAND_HELPER) or 0.1)

    prev_str = state.get(SELECT) or ""
    try:
        prev_base = float(prev_str.replace('%', ''))
    except Exception:
        prev_base = 60.0

    # Quiet outdoor switch adds an extra "layer" at max demand:
    #  - select=100 + quiet_outdoor ON  => effective demand 100
    #  - select=100 + quiet_outdoor OFF => effective demand 105
    QUIET_SW = u.get("QUIET_OUTDOOR_SWITCH")
    quiet_state = None
    quiet_on = None
    if QUIET_SW:
        quiet_state = state.get(QUIET_SW)
        quiet_on = (quiet_state == "on")

    unit_has_quiet = bool(QUIET_SW)
    unit_max_dem = MAX_DEM_LAYER if unit_has_quiet else MAX_DEM

    prev = prev_base
    if isfinite(prev_base) and prev_base >= 100.0 - 1e-6 and (quiet_on is False):
        prev = unit_max_dem


    # ------------------------------------------------------------
    # Defrost + post-defrost hold behavior:
    # - Freeze to the pre-defrost demand during defrost
    # - Keep same demand for 5 minutes after defrost ends
    # - Do not collect/use Tin measurements until hold period ends
    # ------------------------------------------------------------
    try:
        liquid = float(state.get(LIQUID) or 100.0)
    except Exception:
        liquid = 100.0
    if not isfinite(liquid):
        liquid = 100.0

    defrosting = (liquid < DEFROST_LIQUID_THRESHOLD)

    if _last_defrosting[unit_name] is None:
        _last_defrosting[unit_name] = defrosting

    # Entering defrost: capture current demand and clear Tin history
    if (_last_defrosting[unit_name] is False) and (defrosting is True):
        _held_select_option[unit_name] = prev_str
        _held_quiet_on[unit_name] = quiet_on if unit_has_quiet else None
        _hold_until[unit_name] = 0.0
        _tin_hist[unit_name] = []
        log.info(
            "Daikin ML (%s): entering defrost -> holding demand select=%s quiet=%s",
            unit_name, str(_held_select_option[unit_name]), str(_held_quiet_on[unit_name])
        )

    # Exiting defrost: start cooldown as before + start 5-minute hold
    if (_last_defrosting[unit_name] is True) and (defrosting is False):
        _cooldown_until[unit_name] = now + COOLDOWN_MINUTES * 60.0
        _hold_until[unit_name] = now + POST_DEFROST_HOLD_S
        _tin_hist[unit_name] = []
        log.info(
            "Daikin ML (%s): defrost ended -> cooldown %d min, holding pre-defrost demand for %.0f s",
            unit_name, COOLDOWN_MINUTES, float(POST_DEFROST_HOLD_S)
        )

    _last_defrosting[unit_name] = defrosting
    hold_active = (now < float(_hold_until.get(unit_name) or 0.0))
    in_cooldown = (now < _cooldown_until[unit_name])

    # During defrost or hold: enforce held demand and skip using Tin measurements
    if defrosting or hold_active:
        held_opt = _held_select_option.get(unit_name) or prev_str
        held_quiet = _held_quiet_on.get(unit_name) if unit_has_quiet else None

        # Enforce select option
        if held_opt and held_opt != prev_str:
            try:
                select.select_option(entity_id=SELECT, option=held_opt)
                prev_str = held_opt
            except Exception as e:
                log.error("Daikin ML (%s): failed to enforce held select %s: %s", unit_name, held_opt, e)

        # Enforce quiet state if available
        if unit_has_quiet and held_quiet is not None and QUIET_SW:
            try:
                if held_quiet and (quiet_on is False):
                    switch.turn_on(entity_id=QUIET_SW)
                    quiet_on = True
                elif (not held_quiet) and (quiet_on is True):
                    switch.turn_off(entity_id=QUIET_SW)
                    quiet_on = False
            except Exception:
                pass

        # Publish learned demand as held (cron-only publishing already)
        try:
            held_num = float(str(held_opt).replace('%', '')) if held_opt else float(prev_base)
        except Exception:
            held_num = float(prev_base) if isfinite(prev_base) else 0.0

        try:
            state.set(
                LEARNED_SENSOR,
                value=round(float(held_num), 1),
                unit=unit_name,
                ctx="hold",
                defrosting=defrosting,
                post_defrost_hold=hold_active,
                hold_until=round(float(_hold_until.get(unit_name) or 0.0), 1),
                held_select=str(held_opt),
                held_quiet=bool(held_quiet) if unit_has_quiet and held_quiet is not None else None,
                liquid=round(float(liquid), 1),
            )
        except Exception as e:
            log.error("Daikin ML (%s): failed to publish held learned demand: %s", unit_name, e)

        return

    # Normal operation resumes here (defrost false, hold ended).
    # Tin measurements are allowed again; start building history from scratch.
    if isfinite(Tin):
        _tin_hist_add(unit_name, now, Tin)

    if not (isfinite(Tin) and isfinite(Tout_raw)):
        log.info("Daikin ML (%s): sensors not ready; Tin=%s Tout=%s", unit_name, state.get(INDOOR), state.get(OUTDOOR))
        return

    Tout_bucket = int(round(Tout_raw))
    ctx = _context_key_for_outdoor(Tout_raw)

    # Minimum demand floor for this outdoor band (hard constraint)
    min_floor = _min_demand_floor_for_outdoor(u, Tout_bucket)

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

    # apply max cap by outdoor band
    band_upper = min(band_upper, _max_demand_cap_for_outdoor(u, Tout_bucket, band_upper))

    # If min_floor is higher than computed band_upper, lift the band_upper:
    if min_floor > band_upper:
        log.warning(
            "Daikin ML (%s): min_floor %.1f > band_upper %.1f (conflict: min vs max). Lifting upper to min_floor.",
            unit_name, float(min_floor), float(band_upper)
        )
        band_upper = min_floor

    # cooldown rule: during cooldown, upper must be 100 with quiet ON (no 105 layer)
    unit_max_dem_eff = unit_max_dem
    if in_cooldown and unit_has_quiet:
        band_upper = min(band_upper, 100.0)
        unit_max_dem_eff = 100.0

    band_upper_layer = band_upper if band_upper < 100.0 else unit_max_dem_eff

    # jos edellinen yli band_upper, tiputetaan heti
    if prev > band_upper_layer:
        option_cap = _snap_to_select(SELECT, band_upper, -1)
        if option_cap and option_cap != prev_str:
            select.select_option(entity_id=SELECT, option=option_cap)

        # Ensure quiet outdoor is ON when capping / cooldown (per-unit)
        if QUIET_SW and (quiet_on is False):
            if in_cooldown or band_upper < 100.0 or abs(prev - unit_max_dem) < 1e-6:
                try:
                    switch.turn_on(entity_id=QUIET_SW)
                    quiet_on = True
                except Exception:
                    pass

        log.info(
            "Daikin ML (%s): CAP ENFORCED: prev=%s -> %s (upper=%.0f%%, ctx=%s, Tout_bucket=%d)",
            unit_name, prev_str, option_cap, band_upper, ctx, Tout_bucket,
        )
        return

    prev_eff = prev if prev <= band_upper_layer else band_upper_layer

    # ------------------------------------------------------------
    # Learning rate uses smoothed 5-minute Tin slope.
    # If not enough history yet, fall back to the derivative sensor.
    # ------------------------------------------------------------
    sm_rate, used_smoothed, span_s, n_pts = _tin_smoothed_rate_5min(unit_name, now)
    if used_smoothed and isfinite(sm_rate):
        rate = float(sm_rate)
        rate_bad = False
        rate_source = "smoothed_5min"
        rate_span_s = float(span_s)
        rate_n = int(n_pts)
    else:
        # Fallback: derivative sensor (existing behavior)
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
        rate_source = "derivative_fallback"
        rate_span_s = float(span_s) if isfinite(span_s) else 0.0
        rate_n = int(n_pts) if isinstance(n_pts, int) else 0

    err         = sp - Tin
    demand_norm = _clip(prev_eff / 100.0, 0.0, 1.0)
    x = [1.0, err, demand_norm, (Tout_raw - Tin)]
    y = rate

    allow_learning = (not defrosting) and (not hold_active) and (not in_cooldown) and (not rate_bad)

    if allow_learning:
        theta_prev = theta[:]
        P_prev = [row[:] for row in P]

        theta_new, P_new = _rls_update(theta, P, x, y)

        if _all_finite(theta_new):
            theta = theta_new
            try:
                tr = float(P_new[0][0] + P_new[1][1] + P_new[2][2] + P_new[3][3])
            except Exception:
                tr = float("nan")
            if (not isfinite(tr)) or tr <= 1e-9:
                log.warning("RLS reset for %s/%s: P_update_trace_nonpositive", unit_name, ctx)
                P = [[P0, 0, 0, 0],
                     [0, P0, 0, 0],
                     [0, 0, P0, 0],
                     [0, 0, 0, P0]]
            else:
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
            "Daikin ML (%s): learning paused (defrost=%s, hold=%s, cooldown=%s, rate_bad=%s, ctx=%s, rate_source=%s)",
            unit_name, defrosting, hold_active, in_cooldown, rate_bad, ctx, rate_source,
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

    dem_opt = _clip((num / (denom_mag * denom_sign)) * 100.0, 0.0, unit_max_dem_eff)

    if not isfinite(dem_opt):
        log.error(
            "Daikin ML (%s): dem_opt became non-finite (ctx=%s). Falling back to prev_eff=%.1f",
            unit_name, ctx, prev_eff
        )
        dem_opt = prev_eff

    # Quiet outdoor extra layer (OPTIONAL): only if unit has quiet switch.
    # never allow 105 layer during cooldown
    if (not in_cooldown) and unit_has_quiet and dem_opt >= 100.0 - 1e-6 and err > deadband:
        over = _clip((err - deadband) * 2.0, 0.0, QUIET_LAYER_EXTRA)
        dem_opt = _clip(100.0 + over, 100.0, unit_max_dem_eff)
    elif in_cooldown and unit_has_quiet and dem_opt >= 100.0 - 1e-6:
        dem_opt = 100.0

    try:
        state.set(
            LEARNED_SENSOR,
            value=round(float(dem_opt), 1),
            unit=unit_name,
            ctx=ctx,
            outdoor_bucket=Tout_bucket,
            outdoor=round(Tout_raw, 1),
            setpoint=round(sp, 2),
            sp_base=round(float(sp_base), 2),
            sp_bias_degC=round(float(sp_bias_degC), 3),
            bias_points=round(float(bias_points), 3),
            bias_enabled=bias_enabled,
            avg_window_h=round(float(avg_window_h), 2),
            avg_window_factor=round(float(avg_window_factor), 3),
            sp_min_guard=round(float(sp_min_guard), 2),
            sp_max_guard=round(float(sp_max_guard), 2),
            sp_effective=round(float(sp), 2),
            indoor=round(Tin, 2),
            defrosting=defrosting,
            post_defrost_hold=hold_active,
            hold_until=round(float(_hold_until.get(unit_name) or 0.0), 1),
            cooldown=in_cooldown,
            rate=round(float(rate), 4),
            rate_source=rate_source,
            rate_span_s=round(float(rate_span_s), 1),
            rate_n=int(rate_n),
            quiet_outdoor=quiet_state,
            demand_layer_max=(unit_max_dem_eff if (unit_has_quiet and band_upper >= 100.0) else band_upper),
            min_floor=round(float(min_floor), 1),
            max_cap=round(float(band_upper), 1),
        )
    except Exception as e:
        log.error("Daikin ML (%s): failed to update learned demand sensor: %s", unit_name, e)

    # Deadband / efficiency:
    if abs(err) <= deadband:
        trim = EFF_TRIM_STEP_FAST if err < 0 else EFF_TRIM_STEP
        if trim > step_limit:
            trim = step_limit
        dem_target = max(min_floor, prev_eff - trim)
    else:
        dem_target = dem_opt

    # 105-layer stick fix: if stable or above, force back to 100
    if unit_has_quiet and prev_eff > 100.0 and (abs(err) <= deadband or err <= 0.0):
        dem_target = min(dem_target, 100.0)

    # cooldown rule: force max 100
    if in_cooldown and unit_has_quiet:
        dem_target = min(dem_target, 100.0)

    # Monotoninen ehto: jos ollaan kylmällä puolella, demand ei saa laskea
    if err > deadband and dem_target < prev_eff:
        dem_target = prev_eff

    # Soft landing
    abs_err = abs(err)
    prev_err = _prev_err.get(unit_name)
    approaching = False
    if prev_err is not None and isfinite(float(prev_err)):
        if err > 0 and prev_err > 0 and (err < (prev_err - SOFT_APPROACH_EPS)):
            approaching = True
        elif err < 0 and prev_err < 0 and (err > (prev_err + SOFT_APPROACH_EPS)):
            approaching = True

    if approaching and abs_err <= SOFT_ERR_START:
        if SOFT_ERR_START > SOFT_ERR_END:
            soft_factor = _clip((abs_err - SOFT_ERR_END) / (SOFT_ERR_START - SOFT_ERR_END), 0.0, 1.0)
        else:
            soft_factor = 1.0

        dem_target = prev_eff + soft_factor * (dem_target - prev_eff)
        step_limit_soft = max(SOFT_STEP_MIN, step_limit * soft_factor)
    else:
        step_limit_soft = step_limit

    # delta ja step/cooldown -rajat
    delta = dem_target - prev_eff
    if delta > 0:
        up_limit = COOLDOWN_STEP_UP if in_cooldown else step_limit_soft
        if delta > up_limit:
            dem_target = prev_eff + up_limit
    elif delta < 0:
        down_limit = step_limit_soft
        if -delta > down_limit:
            dem_target = prev_eff - down_limit

    # Final clamp now respects the outdoor-band minimum floor and maximum cap
    dem_clip = _clip(dem_target, min_floor, band_upper_layer)

    stable_or_above = (abs(err) <= deadband) or (err <= 0.0)
    if unit_has_quiet and stable_or_above and dem_clip >= 100.0 - 1e-6:
        dem_clip = 100.0
    if in_cooldown and unit_has_quiet and dem_clip >= 100.0 - 1e-6:
        dem_clip = 100.0

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

    # Apply demand command:
    desired_layer = float(dem_clip)
    desired_base = desired_layer if desired_layer < 100.0 else 100.0
    desired_option = _snap_to_select(SELECT, desired_base, 0)

    # Determine what quiet state we want (per-unit)
    desired_quiet = None
    if QUIET_SW and desired_base >= 100.0 - 1e-6:
        if in_cooldown:
            desired_quiet = True
        elif stable_or_above:
            desired_quiet = True
        else:
            desired_quiet = (desired_layer <= 100.0 + 1e-6)
    elif QUIET_SW:
        desired_quiet = True if (prev >= 100.0 - 1e-6 and (quiet_on is False)) else quiet_on

    # per-unit minimum interval between applied demand changes
    min_interval_s = float(_demand_change_min_interval_s(u))
    current_sig = (str(prev_str), bool(quiet_on) if unit_has_quiet else None)
    desired_sig = (str(desired_option), bool(desired_quiet) if unit_has_quiet else None)

    if desired_sig != current_sig and min_interval_s > 0.0:
        last_ts = float(_last_demand_change_ts.get(unit_name) or 0.0)
        dt = now - last_ts
        if dt < min_interval_s:
            log.info(
                "Daikin ML (%s): demand change rate-limited (%.1fs < %.1fs). Keeping select=%s quiet=%s, wanted select=%s quiet=%s",
                unit_name, dt, min_interval_s,
                str(prev_str), str(quiet_on),
                str(desired_option), str(desired_quiet),
            )
            _prev_err[unit_name] = err
            return

    # Actually apply changes (if any)
    applied_any = False

    if desired_option and desired_option != prev_str:
        try:
            select.select_option(entity_id=SELECT, option=desired_option)
            applied_any = True
        except Exception as e:
            log.error("Daikin ML (%s): failed to set select %s -> %s: %s", unit_name, prev_str, desired_option, e)

    if QUIET_SW and desired_base >= 100.0 - 1e-6:
        want_quiet_on = True if desired_quiet is True else False
        if want_quiet_on and (quiet_on is False):
            try:
                switch.turn_on(entity_id=QUIET_SW)
                quiet_on = True
                applied_any = True
            except Exception:
                pass
        elif (not want_quiet_on) and (quiet_on is True):
            try:
                switch.turn_off(entity_id=QUIET_SW)
                quiet_on = False
                applied_any = True
            except Exception:
                pass
    elif QUIET_SW and desired_base < 100.0 - 1e-6:
        if prev >= 100.0 - 1e-6 and (quiet_on is False):
            try:
                switch.turn_on(entity_id=QUIET_SW)
                quiet_on = True
                applied_any = True
            except Exception:
                pass

    if applied_any:
        _last_demand_change_ts[unit_name] = now
        _last_demand_sig[unit_name] = desired_sig

    _prev_err[unit_name] = err

    theta_str = "[" + ", ".join([str(round(float(v), 4)) for v in theta]) + "]"
    cool_str  = "ACTIVE" if in_cooldown else "off"
    icing_str = "ON" if in_icing_band else "off"
    log.info(
        "Daikin ML (%s): ctx=%s | Tin=%.2f°C, Tout=%.2f°C (bucket=%d, →%.1f°C), "
        "DEFROST=%s, hold=%s, cooldown=%s, rate_bad=%s, rate_source=%s, "
        "icing_band=%s, band_upper=%.0f%%, min_floor=%.0f%%, theta=%s | "
        "SP=%.2f, step_limit=%.1f, deadband=%.2f | "
        "prev=%s → opt≈%.0f%% → target≈%.0f%% → clip=%.0f%% → select=%s",
        unit_name, ctx, Tin, Tout_raw, Tout_bucket, Tout_future,
        str(defrosting), str(hold_active), cool_str, str(rate_bad), str(rate_source),
        icing_str, band_upper, min_floor, theta_str,
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
    global _prev_err, _last_demand_change_ts, _last_demand_sig, _tin_hist
    global _hold_until, _held_select_option, _held_quiet_on

    for u in DAIKINS:
        unit_name = u["name"]
        _init_unit_if_needed(unit_name)

        old_cnt = len(_theta_by_unit_ctx.get(unit_name) or {})
        _theta_by_unit_ctx[unit_name] = {}
        _P_by_unit_ctx[unit_name] = {}
        _params_loaded[unit_name] = False
        _last_defrosting[unit_name] = None
        _cooldown_until[unit_name] = 0.0
        _prev_err[unit_name] = None

        _last_demand_change_ts[unit_name] = 0.0
        _last_demand_sig[unit_name] = None

        _tin_hist[unit_name] = []

        _hold_until[unit_name] = 0.0
        _held_select_option[unit_name] = None
        _held_quiet_on[unit_name] = None

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


# ============================================================
# 8) STORE PERSIST (optional helper)
# ============================================================
@service
def daikin_ml_persist():
    """Persistoi store-entiteetit (varmistus)."""
    _persist_all_stores()
