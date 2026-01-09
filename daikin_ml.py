
# pyscript/daikin_ml_multi.py
# Daikin demand-select controller with online learning (RLS) + discrete MPC action selection.
# MULTI-DAIKIN support.
#
# Version: 2026-01-08 (ML v2)
#
# Key upgrades vs previous version:
#  - FIXED: Correct RLS covariance update (previous version used an incorrect P update).
#  - More robust online learning: outlier gating, skip learning immediately after demand changes.
#  - Coarser configurable outdoor-context bucketing + nearest-context initialization (faster learning transfer).
#  - Persistent store now includes model version, feature names, theta_by_ctx, Pdiag_by_ctx, residual stats.
#  - Discrete MPC-style action selection over actual select options (+ optional 105 "layer" via quiet switch).
#  - Optional setpoint source mode: manual setpoint vs (base + Nordpool bias).
#  - Optional extra sensors as additional exogenous ML features (per unit).
#
# NOTE:
#  - This script is designed for Home Assistant Pyscript.
#  - It intentionally avoids heavy dependencies; everything is incremental and O(n^2) per 5 minutes.

import time
from math import isfinite, sqrt

# ============================================================
# 1) DEVICE CONFIGURATIONS (ADD MORE DAIKINS HERE)
# ============================================================
DAIKINS = [
    {
        "name": "daikin1",

        "INDOOR": "sensor.apollo_temp_1_96a244_board_temperature",
        "INDOOR_RATE": "sensor.apollo_temp_1_derivative",  # fallback only
        "OUTDOOR": "sensor.iv_tulo_lampotila",
        "WEATHER": "weather.koti",

        "SELECT": "select.faikin_demand_control",
        "LIQUID": "sensor.faikin_liquid",

        # Quiet outdoor mode switch (Faikin)
        "QUIET_OUTDOOR_SWITCH": "switch.faikin_quiet_outdoor",

        # Effective setpoint helper (this is what the heat pump targets)
        "SP_HELPER": "input_number.daikin_setpoint",

        # Optional setpoint mode selector (create in HA if you want manual override):
        #  - input_select with states: "auto", "manual"
        #  - OR input_boolean: on=auto, off=manual
        # If missing => "auto".
        "SP_MODE_HELPER": "input_select.daikin1_setpoint_mode",

        "STEP_LIMIT_HELPER": "input_number.daikin_step_limit",
        "DEADBAND_HELPER": "input_number.daikin_deadband",
        "ICING_CAP_HELPER": "input_number.daikin_icing_cap",

        # Manual base setpoint (user controls this; used in AUTO mode)
        "SP_BASE_HELPER": "input_number.daikin_setpoint_base",

        # Nordpool bias points (written by nordpool_15m_bias.py)
        "PRICE_BIAS_HELPER": "input_number.daikin1_price_bias_points",

        # Nordpool bias enable/disable
        "PRICE_BIAS_ENABLED": "input_boolean.nordpool_bias_enabled",

        # Low/high setpoint guards (per unit)
        "MIN_TEMP_GUARD_HELPER": "input_number.daikin1_min_temp_guard",
        "MAX_TEMP_GUARD_HELPER": "input_number.daikin1_max_temp_guard",

        # Persistent store per device (recommend separate entity per Daikin)
        "STORE_ENTITY": "pyscript.daikin1_ml_params",

        # Learned-demand sensor per device
        "LEARNED_SENSOR": "sensor.daikin1_ml_learned_demand",

        # ------------------------------------------------------------
        # Optional: outdoor-context bucket width in °C (coarser => more transfer, fewer contexts)
        #   e.g. 1.0=per-degree, 2.0=every 2°C, 3.0=every 3°C
        # ------------------------------------------------------------
        "CTX_BIN_C": 1.0,

        # ------------------------------------------------------------
        # Optional: additional ML features (exogenous sensors)
        # Each item is mapped to a normalized feature ~[-1..+1] by (value-mid)/half_range.
        # Example:
        # "EXTRA_FEATURES": [
        #   {"name": "solar", "entity": "sensor.solar_power", "min": 0, "max": 6000},
        #   {"name": "wind", "entity": "sensor.wind_speed", "min": 0, "max": 20},
        # ]
        # ------------------------------------------------------------
        "EXTRA_FEATURES": [],

        # ------------------------------------------------------------
        # Minimum demand floors by outdoor temperature band
        # ------------------------------------------------------------
        "MIN_DEM_FLOOR_M05_M10": "input_number.daikin1_min_dem_m05_m10",  # -5 .. -10
        "MIN_DEM_FLOOR_M11_M15": "input_number.daikin1_min_dem_m11_m15",  # -11 .. -15
        "MIN_DEM_FLOOR_LE_M16":  "input_number.daikin1_min_dem_le_m16",   # <= -16

        # ------------------------------------------------------------
        # Maximum demand caps by outdoor temperature band
        # ------------------------------------------------------------
        "MAX_DEM_CAP_M05_M10": "input_number.daikin1_max_dem_m05_m10",  # -5 .. -10
        "MAX_DEM_CAP_M11_M15": "input_number.daikin1_max_dem_m11_m15",  # -11 .. -15
        "MAX_DEM_CAP_LE_M16":  "input_number.daikin1_max_dem_le_m16",   # <= -16

        # ------------------------------------------------------------
        # Per-unit minimum interval between *applied* demand changes (seconds)
        # ------------------------------------------------------------
        "DEMAND_CHANGE_MIN_INTERVAL_HELPER": "input_number.daikin1_demand_change_min_interval_s",

        # ------------------------------------------------------------
        # Post-defrost behavior helpers (per unit)
        # ------------------------------------------------------------
        "POST_DEFROST_DEMAND_HELPER": "input_number.daikin1_post_defrost_demand_pct",
        "POST_DEFROST_HOLD_MINUTES_HELPER": "input_number.daikin1_post_defrost_hold_minutes",
    },
]

# ============================================================
# 2) SHARED CONTROL CONSTANTS
# ============================================================

# Controller cadence (cron): TRUE 5-minute cadence
CONTROL_DT_S = 5 * 60.0
CONTROL_DT_H = CONTROL_DT_S / 3600.0

# Online learning (RLS)
MODEL_VERSION = 2

LAMBDA = 0.995
P0 = 1e4

# Hard demand clamps
MIN_DEM = 30.0
MAX_DEM = 100.0

# Quiet-switch extra "layer" at max demand:
#  - Demand 100 with quiet_outdoor ON  => effective demand 100
#  - Demand 100 with quiet_outdoor OFF => effective demand 105
MAX_DEM_LAYER = 105.0
QUIET_LAYER_EXTRA = 5.0  # fixed layer (binary), kept for backwards naming

# Global "mild weather" max
GLOBAL_MILD_MAX = 95.0

# Cooldown after defrost
COOLDOWN_MINUTES = 15
COOLDOWN_STEP_UP = 3.0

# Icing band cap
ICING_BAND_MIN = -2.0
ICING_BAND_MAX = 4.0
ICING_BAND_CAP_DEFAULT = 80.0

# Auto step/deadband tuning (internal only, does NOT overwrite helpers)
AUTO_STEP_MIN = 3.0
AUTO_STEP_MAX = 20.0
AUTO_STEP_BASE = 10.0

AUTO_DB_MIN = 0.05
AUTO_DB_MAX = 0.50
AUTO_DB_BASE = 0.10

# Soft landing near setpoint
SOFT_ERR_START = 0.2
SOFT_ERR_END = 0.0
SOFT_STEP_MIN = 1.0
SOFT_APPROACH_EPS = 0.01

# Efficiency behavior inside deadband
EFF_TRIM_STEP = 2.0
EFF_TRIM_STEP_FAST = 5.0

# Default min interval between applied demand changes (seconds)
DEMAND_CHANGE_MIN_INTERVAL_DEFAULT_S = 60.0

# Indoor temperature slope estimation
TIN_SLOPE_WINDOW_S = 5 * 60.0
TIN_SLOPE_MIN_SPAN_S = 4 * 60.0
TIN_SLOPE_MIN_SAMPLES = 3

# Defrost detection
DEFROST_LIQUID_THRESHOLD = 20.0
POST_DEFROST_HOLD_DEFAULT_MIN = 5.0
POST_DEFROST_DEMAND_DEFAULT_PCT = 60.0

# ------------------------------------------------------------
# ML feature scaling (keep features in similar numeric ranges)
# ------------------------------------------------------------
ERR_SCALE = 2.0
DELTA_SCALE = 10.0
RATE_SCALE = 2.0
TOUT_SCALE = 10.0

# Rate prediction clamp (degC/h) to avoid insane model outputs destabilizing MPC
PRED_RATE_CLIP = 3.0

# Learning gating:
LEARN_SKIP_AFTER_DEMAND_CHANGE_S = 120.0  # don't update immediately after changing demand
LEARN_MIN_UPDATES_FOR_OUTLIER_GATE = 10   # start outlier gating after some updates
RESID_EWMA_ALPHA = 0.10
RESID_GATE_K = 4.0
RESID_GATE_ABS_MIN = 0.60  # degC/h (always gate truly crazy points)

# Parameter projection constraints (physics-informed):
THETA_DEMAND_MIN = 0.10   # demand coefficient lower bound
THETA_DEMAND_MAX = 20.0   # upper bound (scaled units)
THETA_LOSS_MIN = 0.00     # heat loss coefficient should be >= 0
THETA_LOSS_MAX = 20.0
THETA_INT_MIN = 0.00      # demand*Tout interaction coefficient >= 0
THETA_INT_MAX = 20.0
THETA_RATEPREV_MIN = -1.0
THETA_RATEPREV_MAX = 1.5

# MPC parameters (discrete action selection)
HORIZON_H = 1.0
MPC_STEPS = max(1, int(round(HORIZON_H / CONTROL_DT_H)))  # ~12 steps for 1 hour
MPC_W_ERR = 1.0
MPC_W_OVERSHOOT = 2.0
MPC_W_ENERGY = 0.10
MPC_W_CHANGE = 0.02

# If model has too few updates, fall back to a safe PI-like heuristic.
MIN_UPDATES_FOR_MPC = 20
PI_KP_PCT_PER_C = 15.0   # % demand per °C error (step limit still applies)

# ============================================================
# 3) NORDPOOL -> SETPOINT INTEGRATION
# ============================================================
SP_BASE_DEFAULT = 22.5
SP_BIAS_DEGC_PER_POINT = 0.05
SP_BIAS_CLAMP_MIN = -0.5
SP_BIAS_CLAMP_MAX = +0.5
MIN_GUARD_DEFAULT = 16.0
MAX_GUARD_DEFAULT = 28.0
SP_WRITE_EPS = 0.01

NORDPOOL_AVG_WINDOW_HELPER = "input_number.nordpool_avg_window_hours"
AVG_WINDOW_REF_H = 12.0
AVG_WINDOW_FACTOR_MIN = 0.5
AVG_WINDOW_FACTOR_MAX = 2.0

WEATHER_FORECAST_HOURLY_SENSOR = "sensor.weather_forecast_hourly"
FORECAST_H = 6

AVG_WINDOW_MIN_H = 1.0
AVG_WINDOW_MAX_H = 4.0
AVG_WINDOW_TEMP_COLD = -10.0
AVG_WINDOW_TEMP_WARM = 5.0
AVG_WINDOW_WRITE_EPS = 0.1

# ============================================================
# 4) HELPERS
# ============================================================

def _clip(v, lo, hi):
    try:
        vf = float(v)
    except Exception:
        return lo
    if not isfinite(vf):
        return lo
    if vf < lo:
        return lo
    if vf > hi:
        return hi
    return vf

def _read_float_entity(entity_id, default):
    try:
        v = float(state.get(entity_id))
        return v if isfinite(v) else default
    except Exception:
        return default

def _all_finite(seq):
    for v in seq:
        try:
            if not isfinite(float(v)):
                return False
        except Exception:
            return False
    return True

def _dot(a, b):
    s = 0.0
    n = min(len(a), len(b))
    for i in range(n):
        s += float(a[i]) * float(b[i])
    return s

def _matvec(M, v):
    n = len(v)
    out = [0.0] * n
    for i in range(n):
        s = 0.0
        Mi = M[i]
        for j in range(n):
            s += Mi[j] * v[j]
        out[i] = s
    return out

def _make_eye(n, diag):
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        M[i][i] = float(diag)
    return M

def _symmetrize(P):
    n = len(P)
    for i in range(n):
        for j in range(i + 1, n):
            v = 0.5 * (P[i][j] + P[j][i])
            P[i][j] = v
            P[j][i] = v
    return P

def _rls_update(theta, P, x, y, lam=LAMBDA):
    """
    Correct RLS update:
      K = P x / (lam + x^T P x)
      theta <- theta + K * (y - x^T theta)
      P <- (P - K (x^T P)) / lam
    With symmetric P, x^T P is (P x)^T, so:
      P <- (P - outer(K, Px)) / lam
    Returns (theta_new, P_new, did_update_bool, residual)
    """
    if not (_all_finite(theta) and _all_finite(x)):
        return theta, P, False, 0.0
    try:
        y_f = float(y)
    except Exception:
        return theta, P, False, 0.0
    if not isfinite(y_f):
        return theta, P, False, 0.0

    n = len(theta)
    if len(x) != n or len(P) != n:
        return theta, P, False, 0.0

    Px = _matvec(P, x)
    denom = lam + _dot(x, Px)
    if (not isfinite(denom)) or denom <= 1e-12:
        return theta, P, False, 0.0

    K = [Px[i] / denom for i in range(n)]
    y_hat = _dot(theta, x)
    resid = y_f - y_hat
    if not isfinite(resid):
        return theta, P, False, 0.0

    theta_new = [theta[i] + K[i] * resid for i in range(n)]
    if not _all_finite(theta_new):
        return theta, P, False, resid

    inv_lam = 1.0 / lam
    P_new = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Ki = K[i]
        Pi = P[i]
        for j in range(n):
            P_new[i][j] = (Pi[j] - Ki * Px[j]) * inv_lam

    _symmetrize(P_new)

    # Basic sanity on diagonals
    for i in range(n):
        if (not isfinite(P_new[i][i])) or (P_new[i][i] <= 1e-9) or (P_new[i][i] > 1e12):
            return theta, P, False, resid

    return theta_new, P_new, True, resid

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
    nums = sorted(set(nums))
    has_pct = bool(opts and ('%' in str(opts[0])))
    return nums, has_pct

def _with_pct(select_entity):
    attrs = state.getattr(select_entity) or {}
    opts = attrs.get("options") or []
    return bool(opts and ('%' in str(opts[0])))

def _snap_to_select(select_entity, value, direction):
    nums, has_pct = _select_options_nums(select_entity)
    picked = None

    if not nums:
        picked = round(float(value))
        out = str(int(picked)) + ('%' if (has_pct or _with_pct(select_entity)) else '')
        return out

    v = float(value)

    if direction > 0:
        chosen = None
        for n in nums:
            if n >= v:
                chosen = n
                break
        if chosen is None:
            chosen = nums[-1]
        picked = chosen
    elif direction < 0:
        chosen = None
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] <= v:
                chosen = nums[i]
                break
        if chosen is None:
            chosen = nums[0]
        picked = chosen
    else:
        best = nums[0]
        bestd = abs(nums[0] - v)
        for n in nums:
            d = abs(n - v)
            if d < bestd:
                bestd = d
                best = n
        picked = best

    if float(picked).is_integer():
        picked = int(picked)
    return str(picked) + ('%' if has_pct else '')

def _auto_tune_helpers(theta, unit_name, ctx, step_limit_current, deadband_current):
    # Use demand coefficient magnitude as "gain proxy".
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
    db_auto_rounded = round(db_auto, 2)

    if (abs(step_auto_rounded - step_limit_current) > 0.1 or
            abs(db_auto_rounded - deadband_current) > 0.01):
        log.info(
            "Daikin ML AUTO-TUNE (%s, ctx=%s): gain=%.3f -> step_limit=%.1f (was %.1f), deadband=%.2f (was %.2f)",
            unit_name, ctx, gain,
            step_auto_rounded, step_limit_current,
            db_auto_rounded, deadband_current,
        )

    return step_auto_rounded, db_auto_rounded

def _ctx_key_for_outdoor(Tout: float, bin_c: float) -> str:
    if not isfinite(Tout):
        return "nan"
    try:
        bc = float(bin_c)
    except Exception:
        bc = 0.0
    if (not isfinite(bc)) or bc <= 0.0:
        return "global"
    bucket = int(round(Tout / bc))
    key = bucket * bc
    # make nice string keys, avoid "2.0" if integer
    if abs(key - round(key)) < 1e-6:
        return str(int(round(key)))
    return str(round(key, 2))

def _nearest_existing_ctx(theta_by_ctx, ctx_key, bin_c):
    """
    When entering a new context (outdoor bucket), initialize theta from nearest existing context
    to speed up learning transfer.
    """
    try:
        target = float(ctx_key)
    except Exception:
        return None
    best_key = None
    best_dist = None
    for k in theta_by_ctx.keys():
        try:
            kv = float(k)
        except Exception:
            continue
        d = abs(kv - target)
        if best_dist is None or d < best_dist:
            best_dist = d
            best_key = k
    # accept only reasonably close contexts
    if best_key is None:
        return None
    try:
        bc = float(bin_c)
    except Exception:
        bc = 0.0
    if bc <= 0:
        return best_key
    if best_dist is not None and best_dist <= (2.0 * bc + 1e-6):
        return best_key
    return None

# ------------------------------------------------------------
# Outdoor forecast / Nordpool averaging window dynamic update
# ------------------------------------------------------------
def _read_hourly_forecast_temps(forecast_sensor, hours=FORECAST_H):
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
    t = _clip(temp_c, AVG_WINDOW_TEMP_COLD, AVG_WINDOW_TEMP_WARM)
    span_t = (AVG_WINDOW_TEMP_WARM - AVG_WINDOW_TEMP_COLD)
    if span_t <= 0:
        return AVG_WINDOW_REF_H
    frac = (t - AVG_WINDOW_TEMP_COLD) / span_t
    hours = AVG_WINDOW_MIN_H + frac * (AVG_WINDOW_MAX_H - AVG_WINDOW_MIN_H)
    return _clip(hours, AVG_WINDOW_MIN_H, AVG_WINDOW_MAX_H)

def _update_nordpool_avg_window_hours_all():
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

def _avg_future_outdoor(weather_entity, outdoor_entity):
    attrs = state.getattr(weather_entity) or {}
    fc = attrs.get("forecast") or []
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
        except Exception:
            continue
        if isfinite(tv):
            temps.append(tv)
            idx += 1

    if temps:
        return sum(temps) / float(len(temps))

    try:
        v = float(state.get(outdoor_entity) or 0.0)
        return v if isfinite(v) else 0.0
    except Exception:
        return 0.0

# ------------------------------------------------------------
# Demand floors / caps by outdoor band
# ------------------------------------------------------------
def _min_demand_floor_for_outdoor(u, Tout_bucket):
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
    return max(MIN_DEM, floor)

def _max_demand_cap_for_outdoor(u, Tout_bucket, default_cap):
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
    return max(MIN_DEM, cap)

def _demand_change_min_interval_s(u):
    ent = u.get("DEMAND_CHANGE_MIN_INTERVAL_HELPER")
    if ent:
        v = _read_float_entity(ent, DEMAND_CHANGE_MIN_INTERVAL_DEFAULT_S)
    else:
        v = DEMAND_CHANGE_MIN_INTERVAL_DEFAULT_S
    return _clip(v, 0.0, 600.0)

def _post_defrost_hold_s(u):
    ent = u.get("POST_DEFROST_HOLD_MINUTES_HELPER")
    if ent:
        minutes = _read_float_entity(ent, POST_DEFROST_HOLD_DEFAULT_MIN)
    else:
        minutes = POST_DEFROST_HOLD_DEFAULT_MIN
    minutes = _clip(minutes, 0.0, 120.0)
    return float(minutes) * 60.0

def _post_defrost_demand_pct(u):
    ent = u.get("POST_DEFROST_DEMAND_HELPER")
    if ent:
        pct = _read_float_entity(ent, POST_DEFROST_DEMAND_DEFAULT_PCT)
    else:
        pct = POST_DEFROST_DEMAND_DEFAULT_PCT
    return _clip(pct, 0.0, 100.0)

def _compute_post_defrost_hold_option_and_quiet(u, select_entity, unit_has_quiet):
    pct = float(_post_defrost_demand_pct(u))
    opt = _snap_to_select(select_entity, pct, 0)
    if unit_has_quiet:
        return opt, True
    return opt, None

# ------------------------------------------------------------
# Setpoint computation (manual vs auto base+bias)
# ------------------------------------------------------------
def _read_setpoint_mode(u):
    ent = u.get("SP_MODE_HELPER")
    if not ent:
        return "auto"
    s = state.get(ent)
    if s is None:
        return "auto"
    # input_boolean: on=auto, off=manual
    if s == "on":
        return "auto"
    if s == "off":
        return "manual"
    # input_select: expect "auto"/"manual"
    s2 = str(s).strip().lower()
    if s2 in ("manual", "user"):
        return "manual"
    return "auto"

def _compute_effective_setpoint(u):
    """
    Returns dict:
      sp, sp_mode, sp_base, bias_points, sp_bias_degC, bias_enabled,
      avg_window_h, avg_window_factor, sp_min_guard, sp_max_guard, wrote_sp_helper(bool)
    """
    SP_HELPER = u["SP_HELPER"]
    SP_BASE_HELPER = u.get("SP_BASE_HELPER")
    PRICE_BIAS_HELPER = u.get("PRICE_BIAS_HELPER")
    PRICE_BIAS_ENABLED = u.get("PRICE_BIAS_ENABLED", "input_boolean.nordpool_bias_enabled")
    MIN_TEMP_GUARD_HELPER = u.get("MIN_TEMP_GUARD_HELPER")
    MAX_TEMP_GUARD_HELPER = u.get("MAX_TEMP_GUARD_HELPER")

    sp_mode = _read_setpoint_mode(u)

    # Guards
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

    bias_enabled = (state.get(PRICE_BIAS_ENABLED) == "on")
    bias_points = 0.0
    sp_base = SP_BASE_DEFAULT

    # Read base setpoint (auto only)
    if SP_BASE_HELPER:
        try:
            sp_base = float(state.get(SP_BASE_HELPER) or SP_BASE_DEFAULT)
        except Exception:
            sp_base = SP_BASE_DEFAULT
    if not isfinite(sp_base):
        sp_base = SP_BASE_DEFAULT

    # Bias points (auto only)
    if bias_enabled and PRICE_BIAS_HELPER:
        try:
            bias_points = float(state.get(PRICE_BIAS_HELPER) or 0.0)
        except Exception:
            bias_points = 0.0
        if not isfinite(bias_points):
            bias_points = 0.0

    # Dynamic avg window factor (auto only)
    try:
        avg_window_h = float(state.get(NORDPOOL_AVG_WINDOW_HELPER) or AVG_WINDOW_REF_H)
    except Exception:
        avg_window_h = AVG_WINDOW_REF_H
    if (not isfinite(avg_window_h)) or avg_window_h <= 0.0:
        avg_window_h = AVG_WINDOW_REF_H
    avg_window_factor = _clip(AVG_WINDOW_REF_H / avg_window_h, AVG_WINDOW_FACTOR_MIN, AVG_WINDOW_FACTOR_MAX)

    sp_bias_degC = _clip(bias_points * SP_BIAS_DEGC_PER_POINT * avg_window_factor, SP_BIAS_CLAMP_MIN, SP_BIAS_CLAMP_MAX)

    wrote = False

    if sp_mode == "manual":
        # Manual mode: use SP_HELPER as user-set setpoint; do NOT overwrite it.
        try:
            sp = float(state.get(SP_HELPER) or sp_base)
        except Exception:
            sp = sp_base
        if not isfinite(sp):
            sp = sp_base
        sp = _clip(sp, sp_min_guard, sp_max_guard)
        return {
            "sp": sp,
            "sp_mode": "manual",
            "sp_base": sp_base,
            "bias_points": bias_points,
            "sp_bias_degC": sp_bias_degC,
            "bias_enabled": bias_enabled,
            "avg_window_h": avg_window_h,
            "avg_window_factor": avg_window_factor,
            "sp_min_guard": sp_min_guard,
            "sp_max_guard": sp_max_guard,
            "wrote_sp_helper": False,
        }

    # Auto mode: base + optional bias, then write effective into SP_HELPER
    sp = sp_base + (sp_bias_degC if bias_enabled else 0.0)
    sp = _clip(sp, sp_min_guard, sp_max_guard)

    try:
        sp_prev = float(state.get(SP_HELPER) or sp)
        if (not isfinite(sp_prev)) or abs(sp_prev - sp) > SP_WRITE_EPS:
            input_number.set_value(entity_id=SP_HELPER, value=round(float(sp), 2))
            wrote = True
    except Exception:
        wrote = False

    return {
        "sp": sp,
        "sp_mode": "auto",
        "sp_base": sp_base,
        "bias_points": bias_points,
        "sp_bias_degC": sp_bias_degC if bias_enabled else 0.0,
        "bias_enabled": bias_enabled,
        "avg_window_h": avg_window_h,
        "avg_window_factor": avg_window_factor,
        "sp_min_guard": sp_min_guard,
        "sp_max_guard": sp_max_guard,
        "wrote_sp_helper": wrote,
    }

# ------------------------------------------------------------
# Feature vector builder (base + optional extra sensors)
# ------------------------------------------------------------
def _normalized_extra_feature(val, vmin, vmax):
    try:
        v = float(val)
    except Exception:
        return 0.0
    if not isfinite(v):
        return 0.0
    try:
        lo = float(vmin)
        hi = float(vmax)
    except Exception:
        return 0.0
    if (not isfinite(lo)) or (not isfinite(hi)) or hi <= lo:
        return 0.0
    v = _clip(v, lo, hi)
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    if half <= 1e-9:
        return 0.0
    z = (v - mid) / half  # ~[-1..1]
    return _clip(z, -3.0, 3.0)

def _feature_names_for_unit(u):
    names = [
        "bias",
        "err",
        "demand",
        "delta",
        "rate_prev",
        "demand_tout",
    ]
    extras = u.get("EXTRA_FEATURES") or []
    if isinstance(extras, list):
        for item in extras:
            try:
                nm = str(item.get("name") or "").strip()
                if nm:
                    names.append("x_" + nm)
            except Exception:
                pass
    return names

def _build_feature_vector(u, Tin, sp, Tout, demand_eff, rate_prev):
    """
    Build feature vector x for the model:
      y = dTin/dt (degC/h) ~= dot(theta, x)

    demand_eff is "effective demand layer" (0..105)
    """
    err = float(sp - Tin)
    demand_norm = _clip(float(demand_eff) / 100.0, 0.0, 1.2)
    delta = float(Tout - Tin)

    x = [
        1.0,
        err / ERR_SCALE,
        demand_norm,
        delta / DELTA_SCALE,
        float(rate_prev) / RATE_SCALE,
        demand_norm * (float(Tout) / TOUT_SCALE),
    ]

    extras = u.get("EXTRA_FEATURES") or []
    if isinstance(extras, list) and extras:
        for item in extras:
            try:
                ent = item.get("entity")
                vmin = item.get("min")
                vmax = item.get("max")
                if not ent:
                    x.append(0.0)
                    continue
                raw = state.get(ent)
                x.append(_normalized_extra_feature(raw, vmin, vmax))
            except Exception:
                x.append(0.0)

    return x

def _project_theta(theta, feature_names):
    """
    Apply simple physics-informed projection constraints.
    This prevents the model from learning nonsensical signs that make control unstable.
    """
    n = len(theta)
    out = theta[:]

    # Indices for known base features:
    # 0 bias, 1 err, 2 demand, 3 delta(loss), 4 rate_prev, 5 demand_tout
    if n >= 3:
        out[2] = _clip(out[2], THETA_DEMAND_MIN, THETA_DEMAND_MAX)
    if n >= 4:
        out[3] = _clip(out[3], THETA_LOSS_MIN, THETA_LOSS_MAX)
    if n >= 5:
        out[4] = _clip(out[4], THETA_RATEPREV_MIN, THETA_RATEPREV_MAX)
    if n >= 6:
        out[5] = _clip(out[5], THETA_INT_MIN, THETA_INT_MAX)

    # No constraints for extras by default
    return out

def _predict_rate(theta, x):
    y_hat = _dot(theta, x)
    if not isfinite(y_hat):
        return 0.0
    return _clip(y_hat, -PRED_RATE_CLIP, PRED_RATE_CLIP)

# ============================================================
# 5) PER-UNIT STATE (theta/P + defrost cooldown + learning meta + Tin slope)
# ============================================================

_theta_by_unit_ctx = {}  # unit -> ctx -> theta[list]
_P_by_unit_ctx = {}      # unit -> ctx -> P[list[list]]
_meta_by_unit_ctx = {}   # unit -> ctx -> dict(resid_var, n_updates)
_feature_names_by_unit = {}  # unit -> list[str]
_params_loaded = {}      # unit -> bool

_last_defrosting = {}    # unit -> bool/None
_cooldown_until = {}     # unit -> epoch float

_prev_err = {}           # unit -> float/None
_last_rate = {}          # unit -> last measured smoothed rate (degC/h)

_last_demand_change_ts = {}  # unit -> epoch float of last applied change
_last_demand_sig = {}        # unit -> signature tuple

_tin_hist = {}           # unit -> list[(ts, Tin)]

_hold_until = {}         # unit -> epoch float
_held_select_option = {} # unit -> option str
_held_quiet_on = {}      # unit -> bool/None

# ------------------------------------------------------------
# Persistent store utilities
# ------------------------------------------------------------
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
    if unit_name not in _meta_by_unit_ctx:
        _meta_by_unit_ctx[unit_name] = {}
    if unit_name not in _feature_names_by_unit:
        _feature_names_by_unit[unit_name] = []
    if unit_name not in _params_loaded:
        _params_loaded[unit_name] = False
    if unit_name not in _last_defrosting:
        _last_defrosting[unit_name] = None
    if unit_name not in _cooldown_until:
        _cooldown_until[unit_name] = 0.0
    if unit_name not in _prev_err:
        _prev_err[unit_name] = None
    if unit_name not in _last_rate:
        _last_rate[unit_name] = 0.0
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

def _load_params_from_store(unit, feature_names_new):
    unit_name = unit["name"]
    store = unit["STORE_ENTITY"]
    attrs = state.getattr(store) or {}

    ver = attrs.get("model_version") or 0
    stored_names = attrs.get("feature_names") or []
    theta_by_ctx = attrs.get("theta_by_ctx") or {}
    Pdiag_by_ctx = attrs.get("Pdiag_by_ctx") or {}
    resid_var_by_ctx = attrs.get("resid_var_by_ctx") or {}
    n_updates_by_ctx = attrs.get("n_updates_by_ctx") or {}

    # Backwards compatibility: previous script stored only theta_by_ctx (len=4 vectors)
    # Attempt a migration to the new feature space (base 6 features, plus extras => variable).
    migrated = False

    if not isinstance(theta_by_ctx, dict):
        theta_by_ctx = {}

    # Determine mapping from stored feature names to new names
    map_idx = {}
    if isinstance(stored_names, list) and stored_names and isinstance(feature_names_new, list):
        for i, nm in enumerate(stored_names):
            try:
                map_idx[str(nm)] = int(i)
            except Exception:
                pass

    out_theta = {}
    out_Pdiag = {}
    out_meta = {}

    for ctx, th in theta_by_ctx.items():
        ctx_k = str(ctx)
        if not isinstance(th, (list, tuple)):
            continue

        th_list = list(th)

        # If old 4-feature model, expand into new base features:
        if (ver != MODEL_VERSION) and (len(th_list) == 4) and (len(feature_names_new) >= 6):
            # Old: [bias, err, demand, delta]
            # New base: [bias, err/ERR_SCALE, demand, delta/DELTA_SCALE, rate_prev/RATE_SCALE, demand_tout]
            try:
                a0, a1, a2, a3 = [float(th_list[i]) for i in range(4)]
                if not _all_finite([a0, a1, a2, a3]):
                    continue
                new_th = [0.0] * len(feature_names_new)
                new_th[0] = a0
                new_th[1] = a1 * ERR_SCALE
                new_th[2] = a2
                new_th[3] = a3 * DELTA_SCALE
                new_th[4] = 0.0
                new_th[5] = 0.0
                # extras start at 6 => 0.0
                out_theta[ctx_k] = _project_theta(new_th, feature_names_new)
                migrated = True
            except Exception:
                continue
        else:
            # Same-version path OR stored feature names exist => map by name when possible.
            if isinstance(stored_names, list) and stored_names and map_idx:
                new_th = [0.0] * len(feature_names_new)
                for j, nm in enumerate(feature_names_new):
                    idx = map_idx.get(str(nm))
                    if idx is None:
                        continue
                    try:
                        new_th[j] = float(th_list[idx])
                    except Exception:
                        new_th[j] = 0.0
                out_theta[ctx_k] = _project_theta(new_th, feature_names_new)
            else:
                # Fallback: if lengths match, accept
                if len(th_list) != len(feature_names_new):
                    continue
                try:
                    new_th = [float(v) for v in th_list]
                except Exception:
                    continue
                if not _all_finite(new_th):
                    continue
                out_theta[ctx_k] = _project_theta(new_th, feature_names_new)

        # Pdiag
        pd = None
        if isinstance(Pdiag_by_ctx, dict):
            pd = Pdiag_by_ctx.get(ctx_k)
        if isinstance(pd, (list, tuple)) and len(pd) == len(feature_names_new):
            try:
                pdv = [float(v) for v in pd]
                if _all_finite(pdv):
                    out_Pdiag[ctx_k] = [max(1e-6, min(1e12, v)) for v in pdv]
            except Exception:
                pass

        # meta
        rv = None
        nu = None
        if isinstance(resid_var_by_ctx, dict):
            rv = resid_var_by_ctx.get(ctx_k)
        if isinstance(n_updates_by_ctx, dict):
            nu = n_updates_by_ctx.get(ctx_k)

        try:
            rvf = float(rv) if rv is not None else 0.5 * 0.5
        except Exception:
            rvf = 0.5 * 0.5
        if not isfinite(rvf) or rvf <= 1e-9:
            rvf = 0.5 * 0.5

        try:
            nuf = int(nu) if nu is not None else 0
        except Exception:
            nuf = 0
        if nuf < 0:
            nuf = 0

        out_meta[ctx_k] = {"resid_var": rvf, "n_updates": nuf}

    if out_theta:
        _theta_by_unit_ctx[unit_name] = out_theta
        log.info("Daikin ML (%s): loaded %d contexts from %s (ver=%s, migrated=%s)", unit_name, len(out_theta), store, str(ver), str(migrated))
    else:
        log.info("Daikin ML (%s): no usable stored model in %s, starting fresh", unit_name, store)

    # Rebuild P matrices from stored Pdiag (or default)
    n = len(feature_names_new)
    P_by_ctx = {}
    for ctx_k, th in (_theta_by_unit_ctx.get(unit_name) or {}).items():
        pd = out_Pdiag.get(ctx_k)
        if isinstance(pd, list) and len(pd) == n and _all_finite(pd):
            P = [[0.0] * n for _ in range(n)]
            for i in range(n):
                P[i][i] = float(pd[i])
            P_by_ctx[ctx_k] = P
        else:
            P_by_ctx[ctx_k] = _make_eye(n, P0)

    _P_by_unit_ctx[unit_name] = P_by_ctx
    _meta_by_unit_ctx[unit_name] = out_meta

def _save_params_to_store(unit):
    unit_name = unit["name"]
    store = unit["STORE_ENTITY"]
    feature_names = _feature_names_by_unit.get(unit_name) or []
    n = len(feature_names)

    theta_clean = {}
    Pdiag_clean = {}
    resid_clean = {}
    nupd_clean = {}

    for ctx_k, th in (_theta_by_unit_ctx.get(unit_name) or {}).items():
        if not isinstance(th, (list, tuple)) or len(th) != n:
            continue
        if not _all_finite(th):
            continue
        theta_clean[str(ctx_k)] = [round(float(v), 6) for v in th]

        P = (_P_by_unit_ctx.get(unit_name) or {}).get(ctx_k)
        if isinstance(P, list) and len(P) == n:
            diag = []
            ok = True
            for i in range(n):
                try:
                    diag.append(float(P[i][i]))
                except Exception:
                    ok = False
                    break
            if ok and _all_finite(diag):
                Pdiag_clean[str(ctx_k)] = [round(float(v), 6) for v in diag]

        meta = (_meta_by_unit_ctx.get(unit_name) or {}).get(ctx_k) or {}
        try:
            rv = float(meta.get("resid_var", 0.25))
        except Exception:
            rv = 0.25
        if not isfinite(rv) or rv <= 1e-9:
            rv = 0.25
        resid_clean[str(ctx_k)] = round(rv, 6)

        try:
            nu = int(meta.get("n_updates", 0))
        except Exception:
            nu = 0
        if nu < 0:
            nu = 0
        nupd_clean[str(ctx_k)] = nu

    try:
        state.set(
            store,
            value=time.time(),
            model_version=MODEL_VERSION,
            feature_names=list(feature_names),
            theta_by_ctx=theta_clean,
            Pdiag_by_ctx=Pdiag_clean,
            resid_var_by_ctx=resid_clean,
            n_updates_by_ctx=nupd_clean,
        )
    except Exception as e:
        log.error("Daikin ML (%s): error saving model to %s: %s", unit_name, store, e)

def _init_context_params_if_needed(unit):
    unit_name = unit["name"]
    _init_unit_if_needed(unit_name)
    if _params_loaded.get(unit_name):
        return

    feature_names = _feature_names_for_unit(unit)
    _feature_names_by_unit[unit_name] = feature_names

    # Default empty structures
    _theta_by_unit_ctx[unit_name] = {}
    _P_by_unit_ctx[unit_name] = {}
    _meta_by_unit_ctx[unit_name] = {}

    _load_params_from_store(unit, feature_names)

    # If nothing loaded, leave empty and create contexts on demand
    _params_loaded[unit_name] = True

# ------------------------------------------------------------
# Tin history and 5-minute smoothed slope (linear regression)
# ------------------------------------------------------------
def _tin_hist_add(unit_name, ts, tin):
    if not (isfinite(ts) and isfinite(tin)):
        return
    hist = _tin_hist.get(unit_name) or []
    # Avoid duplicates: if last timestamp is extremely close, overwrite it.
    if hist:
        last_ts, _last_tin = hist[-1]
        if isfinite(last_ts) and abs(float(ts) - float(last_ts)) < 0.5:
            hist[-1] = (float(ts), float(tin))
        else:
            hist.append((float(ts), float(tin)))
    else:
        hist.append((float(ts), float(tin)))

    cutoff = float(ts) - (TIN_SLOPE_WINDOW_S + 30.0)
    new_hist = [(t, v) for (t, v) in hist if isfinite(t) and isfinite(v) and t >= cutoff]
    if len(new_hist) > 200:
        new_hist = new_hist[-200:]
    _tin_hist[unit_name] = new_hist

def _tin_smoothed_rate_5min(unit_name, now_ts):
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
        dt = t - t_mean
        dv = v - v_mean
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

# ------------------------------------------------------------
# Defrost / hold helper
# ------------------------------------------------------------
def _read_defrosting(u):
    LIQUID = u["LIQUID"]
    try:
        liquid = float(state.get(LIQUID) or 100.0)
    except Exception:
        liquid = 100.0
    if not isfinite(liquid):
        liquid = 100.0
    return (liquid < DEFROST_LIQUID_THRESHOLD), liquid

def _update_tin_history_only(u):
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
# 6) STARTUP + TRIGGERS
# ============================================================

_persist_all_stores()

# Build trigger expression: "sensor.a or sensor.b"
_INDOOR_TRIGGERS = []
for u in DAIKINS:
    ent = u.get("INDOOR")
    if ent and isinstance(ent, str):
        _INDOOR_TRIGGERS.append(ent)

_seen = set()
_INDOOR_TRIGGERS = [x for x in _INDOOR_TRIGGERS if not (x in _seen or _seen.add(x))]
_INDOOR_TRIGGER_EXPR = " or ".join(_INDOOR_TRIGGERS) if _INDOOR_TRIGGERS else None

def _get_trigger_entity_from_kwargs(kwargs):
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
                note="ML v2: discrete MPC target published on cron; init=0.0"
            )
        except Exception as e:
            log.error("Daikin ML (%s): failed to init learned demand sensor: %s", u["name"], e)
    log.info("Daikin ML MULTI v2: startup loaded for %d unit(s)", len(DAIKINS))

# Cron + state triggers:
if _INDOOR_TRIGGER_EXPR:
    @time_trigger("cron(*/1 * * * *)")
    @state_trigger(_INDOOR_TRIGGER_EXPR)
    def daikin_ml_controller(**kwargs):
        _update_nordpool_avg_window_hours_all()

        trig_ent = _get_trigger_entity_from_kwargs(kwargs)
        is_cron = (trig_ent is None)

        # If triggered by an indoor sensor update: update Tin history only.
        if not is_cron and trig_ent:
            for uu in DAIKINS:
                if uu.get("INDOOR") == trig_ent:
                    try:
                        _update_tin_history_only(uu)
                    except Exception as e:
                        log.debug("Daikin ML (%s): Tin history update failed: %s", uu.get("name", "?"), e)
                    break
            return

        # Cron run: full control
        for u in DAIKINS:
            try:
                _run_one_unit(u)
            except Exception as e:
                log.error("Daikin ML (%s): controller error: %s", u.get("name", "?"), e)
else:
    @time_trigger("cron(*/1 * * * *)")
    def daikin_ml_controller(**kwargs):
        _update_nordpool_avg_window_hours_all()
        for u in DAIKINS:
            try:
                _run_one_unit(u)
            except Exception as e:
                log.error("Daikin ML (%s): controller error: %s", u.get("name", "?"), e)

# ============================================================
# 7) CORE LOGIC PER UNIT
# ============================================================

def _enumerate_actions(u, select_entity, min_floor, band_upper_layer, unit_has_quiet, in_cooldown, err, deadband):
    """
    Enumerate feasible discrete actions based on the select options and quiet-layer capability.
    Returns list of dict:
      {"base": base_pct, "quiet": bool|None, "eff": effective_pct}
    """
    nums, _has_pct = _select_options_nums(select_entity)
    if not nums:
        nums = list(range(int(MIN_DEM), 101, 5))

    base_max = min(float(band_upper_layer), 100.0)
    base_min = max(float(min_floor), 0.0)

    bases = [float(n) for n in nums if (float(n) >= base_min - 1e-6 and float(n) <= base_max + 1e-6)]
    if not bases:
        # always include something
        bases = [float(_clip(base_min, 0.0, base_max))]

    actions = []
    for b in bases:
        if b < 100.0 - 1e-6:
            actions.append({"base": b, "quiet": True if unit_has_quiet else None, "eff": b})
        else:
            # base=100
            actions.append({"base": 100.0, "quiet": True if unit_has_quiet else None, "eff": 100.0})
            allow_105 = (
                unit_has_quiet
                and (not in_cooldown)
                and (band_upper_layer > 100.0 + 1e-6)
                and (err > deadband + 1e-6)
            )
            if allow_105:
                actions.append({"base": 100.0, "quiet": False, "eff": float(min(band_upper_layer, MAX_DEM_LAYER))})

    # Deduplicate by (base, quiet, eff)
    seen = set()
    out = []
    for a in actions:
        key = (round(float(a["base"]), 3), a["quiet"], round(float(a["eff"]), 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out

def _mpc_select_best_action(u, ctx, theta, feature_names, Tin, sp, Tout_sim, prev_eff, rate_prev, deadband,
                            step_up_limit, step_down_limit, actions):
    """
    Discrete MPC: evaluate candidate actions using the learned rate model, pick minimum cost.
    Respects step limits by filtering actions.
    Returns (best_action_dict, best_cost, debug_dict)
    """
    filtered = []
    for a in actions:
        eff = float(a["eff"])
        if eff - prev_eff > step_up_limit + 1e-6:
            continue
        if prev_eff - eff > step_down_limit + 1e-6:
            continue
        filtered.append(a)

    if not filtered:
        # fallback: keep current effective demand
        return {"base": min(prev_eff, 100.0), "quiet": None, "eff": prev_eff}, 1e9, {"note": "no_actions_after_step_filter"}

    best = None
    best_cost = None
    best_dbg = None

    for a in filtered:
        eff = float(a["eff"])
        Tin_sim = float(Tin)
        rate_sim = float(rate_prev)
        cost_err = 0.0
        cost_ov = 0.0

        for _k in range(MPC_STEPS):
            err_k = sp - Tin_sim
            # deadband-aware error for cost (don't chase tiny errors)
            e = max(abs(err_k) - deadband, 0.0)
            cost_err += e * e
            if Tin_sim > sp:
                o = Tin_sim - sp
                cost_ov += o * o

            x = _build_feature_vector(u, Tin_sim, sp, Tout_sim, eff, rate_sim)
            rate_hat = _predict_rate(theta, x)
            Tin_sim += rate_hat * CONTROL_DT_H
            rate_sim = rate_hat

        demand_norm = _clip(eff / 100.0, 0.0, 1.2)
        cost_energy = demand_norm * demand_norm * float(MPC_STEPS)
        d = eff - prev_eff
        cost_change = d * d

        total = (MPC_W_ERR * cost_err) + (MPC_W_OVERSHOOT * cost_ov) + (MPC_W_ENERGY * cost_energy) + (MPC_W_CHANGE * cost_change)

        if best_cost is None or total < best_cost:
            best_cost = total
            best = a
            best_dbg = {
                "cost_err": cost_err,
                "cost_overshoot": cost_ov,
                "cost_energy": cost_energy,
                "cost_change": cost_change,
                "steps": MPC_STEPS,
            }

    return best, float(best_cost if best_cost is not None else 1e9), best_dbg or {}

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

    STEP_LIMIT_HELPER = u["STEP_LIMIT_HELPER"]
    DEADBAND_HELPER = u["DEADBAND_HELPER"]
    ICING_CAP_HELPER = u["ICING_CAP_HELPER"]
    LEARNED_SENSOR = u["LEARNED_SENSOR"]

    now = time.time()

    # Read sensors
    try:
        Tin = float(state.get(INDOOR))
    except Exception:
        Tin = float("nan")

    try:
        Tout_raw = float(state.get(OUTDOOR))
    except Exception:
        Tout_raw = float("nan")

    if not (isfinite(Tin) and isfinite(Tout_raw)):
        log.info("Daikin ML (%s): sensors not ready; Tin=%s Tout=%s", unit_name, state.get(INDOOR), state.get(OUTDOOR))
        return

    # Compute effective setpoint (manual or auto base+bias)
    sp_info = _compute_effective_setpoint(u)
    sp = float(sp_info["sp"])

    # Control tuning helpers
    step_limit_current = _read_float_entity(STEP_LIMIT_HELPER, 10.0)
    deadband_current = _read_float_entity(DEADBAND_HELPER, 0.1)

    # Read previous select state
    prev_str = state.get(SELECT) or ""
    try:
        prev_base = float(str(prev_str).replace('%', ''))
    except Exception:
        prev_base = 60.0

    # Quiet outdoor switch state (if present)
    QUIET_SW = u.get("QUIET_OUTDOOR_SWITCH")
    quiet_state = None
    quiet_on = None
    unit_has_quiet = bool(QUIET_SW)

    if QUIET_SW:
        quiet_state = state.get(QUIET_SW)
        quiet_on = (quiet_state == "on")

    unit_max_dem = MAX_DEM_LAYER if unit_has_quiet else MAX_DEM

    # Compute previous "effective demand layer"
    prev_eff = float(prev_base)
    if isfinite(prev_base) and prev_base >= 100.0 - 1e-6 and unit_has_quiet and (quiet_on is False):
        prev_eff = unit_max_dem  # 105 layer

    # ------------------------------------------------------------
    # Defrost + post-defrost hold behavior (unchanged behavior, cleaned)
    # ------------------------------------------------------------
    defrosting, liquid = _read_defrosting(u)

    if _last_defrosting[unit_name] is None:
        _last_defrosting[unit_name] = defrosting

    # Entering defrost: capture current demand and clear Tin history
    if (_last_defrosting[unit_name] is False) and (defrosting is True):
        _held_select_option[unit_name] = prev_str
        _held_quiet_on[unit_name] = quiet_on if unit_has_quiet else None
        _hold_until[unit_name] = 0.0
        _tin_hist[unit_name] = []
        log.info(
            "Daikin ML (%s): entering defrost -> holding pre-defrost select=%s quiet=%s",
            unit_name, str(_held_select_option[unit_name]), str(_held_quiet_on[unit_name])
        )

    # Exiting defrost: start cooldown + user-configurable hold
    if (_last_defrosting[unit_name] is True) and (defrosting is False):
        _cooldown_until[unit_name] = now + COOLDOWN_MINUTES * 60.0

        hold_s = float(_post_defrost_hold_s(u))
        _hold_until[unit_name] = now + hold_s

        held_opt, held_quiet = _compute_post_defrost_hold_option_and_quiet(u, SELECT, unit_has_quiet)
        _held_select_option[unit_name] = held_opt
        _held_quiet_on[unit_name] = held_quiet

        _tin_hist[unit_name] = []
        log.info(
            "Daikin ML (%s): defrost ended -> cooldown %d min, post-defrost hold %.0f s, holding select=%s quiet=%s",
            unit_name, COOLDOWN_MINUTES, float(hold_s),
            str(_held_select_option[unit_name]), str(_held_quiet_on[unit_name])
        )

    _last_defrosting[unit_name] = defrosting
    hold_active = (now < float(_hold_until.get(unit_name) or 0.0))
    in_cooldown = (now < float(_cooldown_until.get(unit_name) or 0.0))

    # During defrost or post-defrost hold: enforce held demand and skip learning/control
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

        # Enforce quiet state
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

        # Publish learned demand as held
        try:
            held_num = float(str(held_opt).replace('%', '')) if held_opt else float(prev_base)
        except Exception:
            held_num = float(prev_base) if isfinite(prev_base) else 0.0
        held_num = _clip(held_num, 0.0, 100.0)

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
                setpoint=round(sp, 2),
                sp_mode=sp_info["sp_mode"],
            )
        except Exception as e:
            log.error("Daikin ML (%s): failed to publish held learned demand: %s", unit_name, e)
        return

    # Normal operation: update Tin history
    _tin_hist_add(unit_name, now, Tin)

    # ------------------------------------------------------------
    # Context selection
    # ------------------------------------------------------------
    Tout_bucket = int(round(Tout_raw))
    ctx_bin_c = u.get("CTX_BIN_C", 2.0)
    ctx = _ctx_key_for_outdoor(Tout_raw, ctx_bin_c)

    feature_names = _feature_names_by_unit.get(unit_name) or _feature_names_for_unit(u)
    _feature_names_by_unit[unit_name] = feature_names
    n_feat = len(feature_names)

    # Ensure context exists (initialize from nearest if possible)
    if ctx not in _theta_by_unit_ctx[unit_name]:
        nearest = _nearest_existing_ctx(_theta_by_unit_ctx[unit_name], ctx, ctx_bin_c)
        if nearest and nearest in _theta_by_unit_ctx[unit_name]:
            theta0 = list(_theta_by_unit_ctx[unit_name][nearest])
            log.info("Daikin ML (%s): new ctx=%s -> initialized from nearest ctx=%s", unit_name, ctx, nearest)
        else:
            # Safe physical-ish defaults (scaled feature space):
            # rate ≈ demand_coeff*demand_norm + loss_coeff*delta + small bias
            theta0 = [0.0] * n_feat
            if n_feat >= 6:
                theta0[2] = 5.0  # demand coefficient
                theta0[3] = 1.0  # heat loss coefficient (positive => delta term negative when Tout<Tin)
                theta0[4] = 0.3  # inertia
                theta0[5] = 0.5  # demand_tout interaction
            elif n_feat >= 4:
                theta0[2] = 5.0
                theta0[3] = 1.0
            theta0 = _project_theta(theta0, feature_names)
            log.info("Daikin ML (%s): new ctx=%s -> initialized defaults", unit_name, ctx)

        _theta_by_unit_ctx[unit_name][ctx] = theta0
        _P_by_unit_ctx[unit_name][ctx] = _make_eye(n_feat, P0)
        _meta_by_unit_ctx[unit_name][ctx] = {"resid_var": 0.25, "n_updates": 0}
        _save_params_to_store(u)

    theta = _theta_by_unit_ctx[unit_name][ctx]
    P = _P_by_unit_ctx[unit_name][ctx]
    meta = _meta_by_unit_ctx[unit_name].get(ctx) or {"resid_var": 0.25, "n_updates": 0}
    resid_var = float(meta.get("resid_var", 0.25))
    n_updates = int(meta.get("n_updates", 0))

    # Auto tune (internal)
    step_limit, deadband = _auto_tune_helpers(theta, unit_name, ctx, step_limit_current, deadband_current)

    # Global upper cap by mild/very cold bucket
    if Tout_bucket <= -5:
        global_upper = MAX_DEM
    else:
        global_upper = GLOBAL_MILD_MAX

    # Icing band cap
    icing_cap = _read_float_entity(ICING_CAP_HELPER, ICING_BAND_CAP_DEFAULT)
    icing_cap = _clip(icing_cap, MIN_DEM, MAX_DEM)
    in_icing_band = (Tout_bucket >= ICING_BAND_MIN and Tout_bucket <= ICING_BAND_MAX)
    band_upper = min(global_upper, icing_cap) if in_icing_band else global_upper

    # Apply max cap by outdoor band
    band_upper = min(band_upper, _max_demand_cap_for_outdoor(u, Tout_bucket, band_upper))

    # Minimum demand floor for this band
    min_floor = _min_demand_floor_for_outdoor(u, Tout_bucket)

    if min_floor > band_upper:
        log.warning(
            "Daikin ML (%s): min_floor %.1f > band_upper %.1f (conflict). Lifting upper to min_floor.",
            unit_name, float(min_floor), float(band_upper)
        )
        band_upper = min_floor

    # Cooldown rule: during cooldown, max=100 and quiet must be ON (no 105 layer)
    band_upper_layer = float(band_upper)
    unit_max_dem_eff = unit_max_dem
    if unit_has_quiet and band_upper >= 100.0 - 1e-6:
        band_upper_layer = float(unit_max_dem)
        if in_cooldown:
            band_upper_layer = 100.0
            unit_max_dem_eff = 100.0
    else:
        unit_max_dem_eff = min(unit_max_dem_eff, band_upper_layer)

    # If previous exceeds cap, enforce immediate cap and return
    if prev_eff > band_upper_layer + 1e-6:
        option_cap = _snap_to_select(SELECT, band_upper, -1)
        if option_cap and option_cap != prev_str:
            try:
                select.select_option(entity_id=SELECT, option=option_cap)
            except Exception:
                pass

        if unit_has_quiet and QUIET_SW and (quiet_on is False):
            try:
                switch.turn_on(entity_id=QUIET_SW)
                quiet_on = True
            except Exception:
                pass

        log.info(
            "Daikin ML (%s): CAP ENFORCED: prev=%s (eff=%.0f) -> %s (upper=%.0f, ctx=%s, Tout_bucket=%d)",
            unit_name, prev_str, prev_eff, option_cap, band_upper_layer, ctx, Tout_bucket
        )
        return

    # ------------------------------------------------------------
    # Measure smoothed rate and decide learning update
    # ------------------------------------------------------------
    rate = 0.0
    rate_bad = False
    rate_source = "unknown"
    rate_span_s = 0.0
    rate_n = 0

    sm_rate, used_smoothed, span_s, n_pts = _tin_smoothed_rate_5min(unit_name, now)
    if used_smoothed and isfinite(sm_rate):
        rate = float(sm_rate)
        rate_source = "smoothed_5min"
        rate_span_s = float(span_s)
        rate_n = int(n_pts)
    else:
        # Fallback derivative sensor (used for info only; we do NOT learn from it by default)
        try:
            rate = float(state.get(INDOOR_RATE) or 0.0)
        except Exception:
            rate = 0.0
        if not isfinite(rate):
            rate = 0.0
            rate_bad = True
        rate_source = "derivative_fallback"
        rate_span_s = float(span_s) if isfinite(span_s) else 0.0
        rate_n = int(n_pts) if isinstance(n_pts, int) else 0

    # Learning uses ONLY stable 5-minute smoothed slopes.
    stable_since_change = (now - float(_last_demand_change_ts.get(unit_name) or 0.0)) >= LEARN_SKIP_AFTER_DEMAND_CHANGE_S

    allow_learning = (
        (not defrosting)
        and (not hold_active)
        and (not in_cooldown)
        and used_smoothed
        and (not rate_bad)
        and stable_since_change
    )

    # Build features for learning based on the demand that generated this slope: prev_eff
    x_meas = _build_feature_vector(u, Tin, sp, Tout_raw, prev_eff, float(_last_rate.get(unit_name) or 0.0))
    y_meas = float(rate)

    if allow_learning:
        # Outlier gating based on residual variance EWMA
        y_hat = _predict_rate(theta, x_meas)
        resid = y_meas - y_hat
        if not isfinite(resid):
            resid = 0.0

        sigma = sqrt(max(1e-9, float(resid_var)))
        gate = False
        if n_updates >= LEARN_MIN_UPDATES_FOR_OUTLIER_GATE:
            if abs(resid) > max(RESID_GATE_ABS_MIN, RESID_GATE_K * sigma):
                gate = True

        if gate:
            # Still update residual variance slowly (so sigma can recover)
            resid_var = (1.0 - RESID_EWMA_ALPHA) * resid_var + RESID_EWMA_ALPHA * (resid * resid)
            meta["resid_var"] = float(resid_var)
            _meta_by_unit_ctx[unit_name][ctx] = meta
            log.debug(
                "Daikin ML (%s): learning gated by outlier (ctx=%s resid=%.3f sigma=%.3f y=%.3f y_hat=%.3f)",
                unit_name, ctx, resid, sigma, y_meas, y_hat
            )
        else:
            theta_new, P_new, did, resid2 = _rls_update(theta, P, x_meas, y_meas, lam=LAMBDA)
            if did and _all_finite(theta_new):
                theta_new = _project_theta(theta_new, feature_names)
                theta = theta_new
                P = P_new

                # Update residual variance EWMA with pre-update residual (resid)
                resid_var = (1.0 - RESID_EWMA_ALPHA) * resid_var + RESID_EWMA_ALPHA * (resid * resid)
                n_updates = n_updates + 1

                meta["resid_var"] = float(resid_var)
                meta["n_updates"] = int(n_updates)

                _theta_by_unit_ctx[unit_name][ctx] = theta
                _P_by_unit_ctx[unit_name][ctx] = P
                _meta_by_unit_ctx[unit_name][ctx] = meta

                _save_params_to_store(u)
            else:
                log.debug("Daikin ML (%s): RLS update skipped/failed (ctx=%s did=%s)", unit_name, ctx, str(did))
    else:
        # Keep structures, but log reason occasionally (debug)
        _theta_by_unit_ctx[unit_name][ctx] = theta
        _P_by_unit_ctx[unit_name][ctx] = P
        _meta_by_unit_ctx[unit_name][ctx] = meta

    # Store measured rate for next tick as rate_prev feature
    if used_smoothed and isfinite(rate):
        _last_rate[unit_name] = float(_clip(rate, -PRED_RATE_CLIP, PRED_RATE_CLIP))

    # ------------------------------------------------------------
    # CONTROL: choose next action
    # ------------------------------------------------------------
    err = float(sp - Tin)

    # Effective outdoor for prediction: mix current + forecast
    Tout_future = _avg_future_outdoor(WEATHER, OUTDOOR)
    Tout_sim = 0.5 * float(Tout_raw) + 0.5 * float(Tout_future)

    # Soft landing step limit modulation
    abs_err = abs(err)
    prev_err = _prev_err.get(unit_name)
    approaching = False
    if prev_err is not None and isfinite(float(prev_err)):
        pe = float(prev_err)
        if err > 0 and pe > 0 and (err < (pe - SOFT_APPROACH_EPS)):
            approaching = True
        elif err < 0 and pe < 0 and (err > (pe + SOFT_APPROACH_EPS)):
            approaching = True

    if approaching and abs_err <= SOFT_ERR_START:
        if SOFT_ERR_START > SOFT_ERR_END:
            soft_factor = _clip((abs_err - SOFT_ERR_END) / (SOFT_ERR_START - SOFT_ERR_END), 0.0, 1.0)
        else:
            soft_factor = 1.0
        step_limit_soft = max(SOFT_STEP_MIN, step_limit * soft_factor)
    else:
        step_limit_soft = float(step_limit)

    step_up_limit = float(COOLDOWN_STEP_UP) if in_cooldown else float(step_limit_soft)
    step_down_limit = float(step_limit_soft)

    stable_or_above = (abs_err <= deadband) or (err <= 0.0)

    # Enumerate candidate actions
    actions = _enumerate_actions(u, SELECT, min_floor, band_upper_layer, unit_has_quiet, in_cooldown, err, deadband)

    # Heuristic inside deadband: trim down demand to find minimum sustaining demand
    if abs(err) <= deadband:
        trim = EFF_TRIM_STEP_FAST if err < 0 else EFF_TRIM_STEP
        trim = min(float(trim), float(step_down_limit))
        target_eff = max(float(min_floor), prev_eff - trim)
        # Never keep 105 layer inside deadband
        if unit_has_quiet and target_eff > 100.0 and stable_or_above:
            target_eff = 100.0
        desired_eff = _clip(target_eff, min_floor, band_upper_layer)
        reason = "deadband_trim"
        mpc_dbg = {}
        mpc_cost = None
    else:
        # Outside deadband: choose action via MPC when model has enough data; else PI fallback
        if n_updates < MIN_UPDATES_FOR_MPC:
            # Safe PI-ish target (still constrained by step limits + caps)
            target_eff = prev_eff + (PI_KP_PCT_PER_C * err)
            # Clip to caps
            target_eff = _clip(target_eff, min_floor, band_upper_layer)
            # Respect step limits
            target_eff = _clip(target_eff, prev_eff - step_down_limit, prev_eff + step_up_limit)
            desired_eff = float(target_eff)
            reason = "pi_fallback"
            mpc_dbg = {"n_updates": int(n_updates)}
            mpc_cost = None
        else:
            # Restrict actions to monotonic direction when far from setpoint
            # (prevents weird oscillation when model is imperfect)
            filtered_actions = []
            if err > deadband + 1e-6:
                # Cold side: prefer not decreasing unless we're clearly approaching within soft zone
                if approaching and abs_err <= SOFT_ERR_START:
                    filtered_actions = actions
                else:
                    filtered_actions = [a for a in actions if float(a["eff"]) >= prev_eff - 1e-6]
                    if not filtered_actions:
                        filtered_actions = actions
            elif err < -deadband - 1e-6:
                # Hot side: prefer not increasing
                filtered_actions = [a for a in actions if float(a["eff"]) <= prev_eff + 1e-6]
                if not filtered_actions:
                    filtered_actions = actions
            else:
                filtered_actions = actions

            best_action, best_cost, dbg = _mpc_select_best_action(
                u=u,
                ctx=ctx,
                theta=theta,
                feature_names=feature_names,
                Tin=Tin,
                sp=sp,
                Tout_sim=Tout_sim,
                prev_eff=prev_eff,
                rate_prev=float(_last_rate.get(unit_name) or 0.0),
                deadband=deadband,
                step_up_limit=step_up_limit,
                step_down_limit=step_down_limit,
                actions=filtered_actions,
            )
            desired_eff = float(best_action["eff"])
            reason = "mpc"
            mpc_dbg = dbg
            mpc_cost = best_cost

    # Final safety clamps
    desired_eff = float(_clip(desired_eff, min_floor, band_upper_layer))

    # 105-layer stick fix: never keep 105 when stable/above or during cooldown
    if unit_has_quiet and desired_eff > 100.0 + 1e-6:
        if stable_or_above or in_cooldown:
            desired_eff = 100.0

    # Map desired effective demand to actual command: (select_option, quiet_state)
    if (unit_has_quiet and desired_eff > 100.0 + 1e-6):
        desired_base = 100.0
        desired_quiet = False
    else:
        desired_base = float(min(desired_eff, 100.0))
        # Prefer quiet ON except when explicitly using 105
        desired_quiet = True if unit_has_quiet else None

    desired_option = _snap_to_select(SELECT, desired_base, 0)

    # Apply cap enforcement post-snap
    try:
        opt_num = float(str(desired_option).replace('%', ''))
    except Exception:
        opt_num = desired_base
    if opt_num > band_upper + 1e-6:
        desired_option = _snap_to_select(SELECT, band_upper, -1)

    # Demand-change rate limiting (min interval between applied changes)
    min_interval_s = float(_demand_change_min_interval_s(u))
    current_sig = (str(prev_str), bool(quiet_on) if unit_has_quiet else None)
    desired_sig = (str(desired_option), bool(desired_quiet) if unit_has_quiet else None)

    if desired_sig != current_sig and min_interval_s > 0.0:
        last_ts = float(_last_demand_change_ts.get(unit_name) or 0.0)
        dt = now - last_ts
        if dt < min_interval_s:
            log.info(
                "Daikin ML (%s): demand change rate-limited (%.1fs < %.1fs). Keeping select=%s quiet=%s, wanted select=%s quiet=%s (reason=%s)",
                unit_name, dt, min_interval_s,
                str(prev_str), str(quiet_on),
                str(desired_option), str(desired_quiet),
                reason,
            )
            # publish sensor still (target for debugging)
            _publish_learned_sensor(
                u=u,
                ctx=ctx,
                Tout_bucket=Tout_bucket,
                Tin=Tin,
                Tout_raw=Tout_raw,
                Tout_future=Tout_future,
                sp=sp,
                sp_info=sp_info,
                defrosting=defrosting,
                hold_active=hold_active,
                in_cooldown=in_cooldown,
                liquid=liquid,
                rate=rate,
                rate_source=rate_source,
                rate_span_s=rate_span_s,
                rate_n=rate_n,
                quiet_state=quiet_state,
                prev_eff=prev_eff,
                desired_eff=desired_eff,
                desired_option=desired_option,
                desired_quiet=desired_quiet,
                reason=reason,
                n_updates=n_updates,
                resid_sigma=sqrt(max(1e-9, resid_var)),
                mpc_cost=mpc_cost,
                mpc_dbg=mpc_dbg,
                min_floor=min_floor,
                max_cap=band_upper,
                band_upper_layer=band_upper_layer,
            )
            _prev_err[unit_name] = err
            return

    # Apply select and quiet changes
    applied_any = False

    if desired_option and desired_option != prev_str:
        try:
            select.select_option(entity_id=SELECT, option=desired_option)
            applied_any = True
        except Exception as e:
            log.error("Daikin ML (%s): failed to set select %s -> %s: %s", unit_name, prev_str, desired_option, e)

    if unit_has_quiet and QUIET_SW and desired_quiet is not None:
        want_on = bool(desired_quiet)
        if want_on and (quiet_on is False):
            try:
                switch.turn_on(entity_id=QUIET_SW)
                quiet_on = True
                applied_any = True
            except Exception:
                pass
        elif (not want_on) and (quiet_on is True):
            try:
                switch.turn_off(entity_id=QUIET_SW)
                quiet_on = False
                applied_any = True
            except Exception:
                pass

    if applied_any:
        _last_demand_change_ts[unit_name] = now
        _last_demand_sig[unit_name] = desired_sig

    # Publish learned demand sensor (FINAL target)
    _publish_learned_sensor(
        u=u,
        ctx=ctx,
        Tout_bucket=Tout_bucket,
        Tin=Tin,
        Tout_raw=Tout_raw,
        Tout_future=Tout_future,
        sp=sp,
        sp_info=sp_info,
        defrosting=defrosting,
        hold_active=hold_active,
        in_cooldown=in_cooldown,
        liquid=liquid,
        rate=rate,
        rate_source=rate_source,
        rate_span_s=rate_span_s,
        rate_n=rate_n,
        quiet_state=quiet_state,
        prev_eff=prev_eff,
        desired_eff=desired_eff,
        desired_option=desired_option,
        desired_quiet=desired_quiet,
        reason=reason,
        n_updates=n_updates,
        resid_sigma=sqrt(max(1e-9, resid_var)),
        mpc_cost=mpc_cost,
        mpc_dbg=mpc_dbg,
        min_floor=min_floor,
        max_cap=band_upper,
        band_upper_layer=band_upper_layer,
    )

    _prev_err[unit_name] = err

    # Log
    theta_str = "[" + ", ".join([str(round(float(v), 4)) for v in theta[:min(8, len(theta))]]) + (", ..." if len(theta) > 8 else "") + "]"
    cool_str = "ACTIVE" if in_cooldown else "off"
    icing_str = "ON" if in_icing_band else "off"

    log.info(
        "Daikin ML (%s): ctx=%s | Tin=%.2f°C, Tout=%.2f°C (bucket=%d, →%.1f°C) | "
        "SP=%.2f (%s) err=%.2f db=%.2f | "
        "reason=%s n=%d sigma≈%.3f | "
        "prev=%s (eff=%.0f) -> target_eff=%.0f -> select=%s quiet=%s | "
        "cooldown=%s icing=%s cap=%.0f floor=%.0f | theta=%s",
        unit_name, ctx, Tin, Tout_raw, Tout_bucket, Tout_future,
        sp, sp_info["sp_mode"], err, deadband,
        reason, int(n_updates), sqrt(max(1e-9, resid_var)),
        str(prev_str), prev_eff, desired_eff, str(desired_option), str(desired_quiet),
        cool_str, icing_str, band_upper_layer, min_floor,
        theta_str,
    )

def _publish_learned_sensor(
    u,
    ctx,
    Tout_bucket,
    Tin,
    Tout_raw,
    Tout_future,
    sp,
    sp_info,
    defrosting,
    hold_active,
    in_cooldown,
    liquid,
    rate,
    rate_source,
    rate_span_s,
    rate_n,
    quiet_state,
    prev_eff,
    desired_eff,
    desired_option,
    desired_quiet,
    reason,
    n_updates,
    resid_sigma,
    mpc_cost,
    mpc_dbg,
    min_floor,
    max_cap,
    band_upper_layer,
):
    LEARNED_SENSOR = u["LEARNED_SENSOR"]
    unit_name = u["name"]

    try:
        state.set(
            LEARNED_SENSOR,
            value=round(float(desired_eff), 1),
            unit=unit_name,
            ctx=ctx,
            outdoor_bucket=int(Tout_bucket),
            outdoor=round(float(Tout_raw), 1),
            outdoor_future=round(float(Tout_future), 1),
            setpoint=round(float(sp), 2),
            sp_mode=sp_info.get("sp_mode"),
            sp_base=round(float(sp_info.get("sp_base", 0.0)), 2),
            sp_bias_degC=round(float(sp_info.get("sp_bias_degC", 0.0)), 3),
            bias_points=round(float(sp_info.get("bias_points", 0.0)), 3),
            bias_enabled=bool(sp_info.get("bias_enabled")),
            avg_window_h=round(float(sp_info.get("avg_window_h", AVG_WINDOW_REF_H)), 2),
            avg_window_factor=round(float(sp_info.get("avg_window_factor", 1.0)), 3),
            sp_min_guard=round(float(sp_info.get("sp_min_guard", MIN_GUARD_DEFAULT)), 2),
            sp_max_guard=round(float(sp_info.get("sp_max_guard", MAX_GUARD_DEFAULT)), 2),

            indoor=round(float(Tin), 2),
            err=round(float(sp - Tin), 3),

            defrosting=bool(defrosting),
            post_defrost_hold=bool(hold_active),
            hold_until=round(float(_hold_until.get(unit_name) or 0.0), 1),
            cooldown=bool(in_cooldown),
            liquid=round(float(liquid), 1),

            rate=round(float(rate), 4),
            rate_source=str(rate_source),
            rate_span_s=round(float(rate_span_s), 1),
            rate_n=int(rate_n),

            quiet_outdoor=str(quiet_state),
            prev_eff=round(float(prev_eff), 1),

            # control output
            desired_option=str(desired_option),
            desired_quiet=bool(desired_quiet) if desired_quiet is not None else None,
            reason=str(reason),

            # ML diagnostics
            n_updates=int(n_updates),
            resid_sigma=round(float(resid_sigma), 4),
            mpc_cost=round(float(mpc_cost), 4) if mpc_cost is not None else None,
            mpc_dbg=mpc_dbg,

            # clamps
            min_floor=round(float(min_floor), 1),
            max_cap=round(float(max_cap), 1),
            band_upper_layer=round(float(band_upper_layer), 1),
        )
    except Exception as e:
        log.error("Daikin ML (%s): failed to update learned demand sensor: %s", unit_name, e)

# ============================================================
# 8) SERVICES
# ============================================================
@service
def daikin_ml_step():
    """Run controller once (all units)."""
    daikin_ml_controller()

@service
def daikin_ml_reset():
    """Reset all learned ML params for all units."""
    global _theta_by_unit_ctx, _P_by_unit_ctx, _meta_by_unit_ctx, _params_loaded
    global _last_defrosting, _cooldown_until, _prev_err, _last_rate
    global _last_demand_change_ts, _last_demand_sig, _tin_hist
    global _hold_until, _held_select_option, _held_quiet_on

    for u in DAIKINS:
        unit_name = u["name"]
        _init_unit_if_needed(unit_name)

        old_cnt = len(_theta_by_unit_ctx.get(unit_name) or {})

        _theta_by_unit_ctx[unit_name] = {}
        _P_by_unit_ctx[unit_name] = {}
        _meta_by_unit_ctx[unit_name] = {}
        _params_loaded[unit_name] = False

        _last_defrosting[unit_name] = None
        _cooldown_until[unit_name] = 0.0
        _prev_err[unit_name] = None
        _last_rate[unit_name] = 0.0

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
                model_version=MODEL_VERSION,
                feature_names=_feature_names_for_unit(u),
                theta_by_ctx={},
                Pdiag_by_ctx={},
                resid_var_by_ctx={},
                n_updates_by_ctx={},
            )
            log.warning(
                "Daikin ML RESET (%s): cleared %d learned contexts, store cleared in %s",
                unit_name, old_cnt, u["STORE_ENTITY"],
            )
        except Exception as e:
            log.error("Daikin ML RESET (%s): error resetting params in %s: %s", unit_name, u["STORE_ENTITY"], e)

@service
def daikin_ml_persist():
    """Persist store entities (safety)."""
    _persist_all_stores()

