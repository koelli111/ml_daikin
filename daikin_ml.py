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
    #
    #     "MIN_DEM_FLOOR_M05_M10": "input_number.daikin2_min_dem_m05_m10",
    #     "MIN_DEM_FLOOR_M11_M15": "input_number.daikin2_min_dem_m11_m15",
    #     "MIN_DEM_FLOOR_LE_M16":  "input_number.daikin2_min_dem_le_m16",
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
# Read float helper + enforce min-demand floors by outdoor band
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

    # sanity
    floor = _clip(floor, 0.0, 100.0)
    # never below global MIN_DEM
    floor = max(MIN_DEM, floor)
    return floor


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
    # current outdoor
    Tout_cur = float("nan")
    try:
        Tout_cur = float(state.get(outdoor_entity))
    except Exception:
        Tout_cur = float("nan")

    # hourly forecast average (next FORECAST_H hours)
    temps = _read_hourly_forecast_temps(WEATHER_FORECAST_HOURLY_SENSOR, hours=FORECAST_H)
    Tout_fc = float("nan")
    if temps:
        Tout_fc = sum(temps) / float(len(temps))

    # pick conservative value
    candidates = []
    if isfinite(Tout_cur):
        candidates.append(Tout_cur)
    if isfinite(Tout_fc):
        candidates.append(Tout_fc)

    if not candidates:
        return 0.0

    # Conservative: colder of available estimates
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
        # Don't break control loop if this fails
        try:
            log.debug("Daikin ML: failed to update nordpool avg window: %s", e)
        except Exception:
            pass



# ============================================================
# 4) PER-LAITE TILA (THETA/P + DEFROST-COOLDOWN)
# ============================================================
_theta_by_unit_ctx = {}   # unit -> ctx -> theta[4]
_P_by_unit_ctx     = {}   # unit -> ctx -> P[4x4]
_params_loaded     = {}   # unit -> bool

_last_defrosting   = {}   # unit -> bool/None
_cooldown_until    = {}   # unit -> epoch float

# Track previous control error per unit (for soft-landing approach detection)
_prev_err          = {}   # unit -> last err (float)


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

# ============================================================
# 5) STARTUP + TRIGGERIT
#    FIX: Trigger reliably on INDOOR temperature changes (e.g. sensor.apollo_round)
# ============================================================

# Build a single pyscript trigger expression string: "sensor.a or sensor.b"
_INDOOR_TRIGGERS = []
for u in DAIKINS:
    ent = u.get("INDOOR")
    if ent and isinstance(ent, str):
        _INDOOR_TRIGGERS.append(ent)

# De-duplicate while preserving order
_seen = set()
_INDOOR_TRIGGERS = [x for x in _INDOOR_TRIGGERS if not (x in _seen or _seen.add(x))]

_INDOOR_TRIGGER_EXPR = " or ".join(_INDOOR_TRIGGERS) if _INDOOR_TRIGGERS else None

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

# Run on every INDOOR change (and also periodic cron as a safety net)
if _INDOOR_TRIGGER_EXPR:
    @time_trigger("cron(*/6 * * * *)")
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

        for u in DAIKINS:
            try:
                _run_one_unit(u)
            except Exception as e:
                log.error("Daikin ML (%s): controller error: %s", u.get("name", "?"), e)
else:
    @time_trigger("cron(*/6 * * * *)")
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

    # 5) Effective setpoint
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

    # Handle swapped guards safely
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
    #  - select=100 + quiet_outdoor ON  => effective 100
    #  - select=100 + quiet_outdoor OFF => effective 105
    QUIET_SW = u.get("QUIET_OUTDOOR_SWITCH")
    quiet_state = None
    quiet_on = None
    if QUIET_SW:
        quiet_state = state.get(QUIET_SW)
        quiet_on = (quiet_state == "on")

    # Quiet outdoor switch is OPTIONAL. If missing, max demand stays at 100.
    unit_has_quiet = bool(QUIET_SW)
    unit_max_dem = MAX_DEM_LAYER if unit_has_quiet else MAX_DEM

    prev = prev_base
    if isfinite(prev_base) and prev_base >= 100.0 - 1e-6 and (quiet_on is False):
        prev = unit_max_dem

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

    # If min_floor is higher than computed band_upper, lift the band_upper:
    # you requested minimum floor to be followed regardless.
    if min_floor > band_upper:
        band_upper = min_floor

    band_upper_layer = band_upper if band_upper < 100.0 else unit_max_dem

    # jos edellinen yli band_upper, tiputetaan heti
    if prev > band_upper_layer:
        option_cap = _snap_to_select(SELECT, band_upper, -1)
        if option_cap and option_cap != prev_str:
            select.select_option(entity_id=SELECT, option=option_cap)

        # If we are capping below 100, ensure quiet outdoor is ON (safe/quiet default)
        if band_upper < 100.0 and QUIET_SW and (quiet_on is False):
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

    err         = sp - Tin
    demand_norm = _clip(prev_eff / 100.0, 0.0, 1.0)
    x = [1.0, err, demand_norm, (Tout_raw - Tin)]
    y = rate

    allow_learning = (not defrosting) and (not in_cooldown) and (not rate_bad)

    if allow_learning:
        # backup in case of corruption
        theta_prev = theta[:]
        P_prev = [row[:] for row in P]

        theta_new, P_new = _rls_update(theta, P, x, y)

        # ensure finite
        if _all_finite(theta_new):
            theta = theta_new
            # Guard: if P update becomes non-positive / ill-conditioned, reset P for this ctx
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
            "Daikin ML (%s): learning paused (defrost=%s, cooldown=%s, rate_bad=%s, ctx=%s)",
            unit_name, defrosting, in_cooldown, rate_bad, ctx,
        )

    Tout_future = _avg_future_outdoor(WEATHER, OUTDOOR)
    Tout_eff    = 0.5 * Tout_raw + 0.5 * Tout_future
    dTin_target = _clip(KAPPA * err / HORIZON_H, -2.0, 2.0)

    num = dTin_target - (theta[0] + theta[1] * err + theta[3] * (Tout_eff - Tin))

    denom_raw = theta[2]
    # denom guard
    if (not isfinite(denom_raw)) or abs(denom_raw) < 1e-6:
        denom_raw = 5.0

    denom_sign = 1.0 if denom_raw >= 0 else -1.0
    denom_mag  = abs(denom_raw)
    if denom_mag < 0.5:
        denom_mag = 0.5

    dem_opt = _clip((num / (denom_mag * denom_sign)) * 100.0, 0.0, unit_max_dem)

    # dem_opt guard (belt-and-suspenders): never allow NaN into sensor/logics
    if not isfinite(dem_opt):
        log.error(
            "Daikin ML (%s): dem_opt became non-finite (ctx=%s). Falling back to prev_eff=%.1f",
            unit_name, ctx, prev_eff
        )
        dem_opt = prev_eff

    # Quiet outdoor extra layer (OPTIONAL): only if unit has quiet switch.
    if unit_has_quiet and dem_opt >= 100.0 - 1e-6 and err > deadband:
        over = _clip((err - deadband) * 2.0, 0.0, QUIET_LAYER_EXTRA)
        dem_opt = _clip(100.0 + over, 100.0, unit_max_dem)

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
            cooldown=in_cooldown,
            rate=round(rate, 4),
            quiet_outdoor=quiet_state,
            demand_layer_max=(unit_max_dem if (unit_has_quiet and band_upper >= 100.0) else band_upper),
            min_floor=round(float(min_floor), 1),
        )
    except Exception as e:
        log.error("Daikin ML (%s): failed to update learned demand sensor: %s", unit_name, e)

    # Deadband / efficiency:
    # Trim down while inside deadband, but never below min_floor.
    if abs(err) <= deadband:
        trim = EFF_TRIM_STEP_FAST if err < 0 else EFF_TRIM_STEP
        if trim > step_limit:
            trim = step_limit
        dem_target = max(min_floor, prev_eff - trim)
    else:
        dem_target = dem_opt

    # FIX: Don't "stick" in the 105% layer.
    # 105 is implemented as select=100 + quiet_outdoor OFF.
    # If stable (inside deadband) or above setpoint, force back to 100.
    if unit_has_quiet and prev_eff > 100.0 and (abs(err) <= deadband or err <= 0.0):
        dem_target = min(dem_target, 100.0)

    # Monotoninen ehto: jos ollaan kylmällä puolella, demand ei saa laskea
    if err > deadband and dem_target < prev_eff:
        dem_target = prev_eff

    # Soft landing: only when we're within SOFT_ERR_START and actually moving toward setpoint
    abs_err = abs(err)
    prev_err = _prev_err.get(unit_name)
    approaching = False
    if prev_err is not None and isfinite(float(prev_err)):
        # Approaching from BELOW: err stays positive and decreases toward 0
        if err > 0 and prev_err > 0 and (err < (prev_err - SOFT_APPROACH_EPS)):
            approaching = True
        # Approaching from ABOVE: err stays negative and increases toward 0
        elif err < 0 and prev_err < 0 and (err > (prev_err + SOFT_APPROACH_EPS)):
            approaching = True

    if approaching and abs_err <= SOFT_ERR_START:
        # 1.0 at edge of soften band -> minimal softening, 0.0 very close -> strongest softening
        if SOFT_ERR_START > SOFT_ERR_END:
            soft_factor = _clip((abs_err - SOFT_ERR_END) / (SOFT_ERR_START - SOFT_ERR_END), 0.0, 1.0)
        else:
            soft_factor = 1.0

        # Blend the target toward current demand to avoid harsh finish near setpoint
        dem_target = prev_eff + soft_factor * (dem_target - prev_eff)

        # Reduce step limit near setpoint (but keep some minimal authority)
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

    # Final clamp now respects the outdoor-band minimum floor
    dem_clip = _clip(dem_target, min_floor, band_upper_layer)

    # ------------------------------------------------------------
    # NEW FIX: When stable (inside deadband) OR above setpoint, never allow the 105-layer.
    # Force dem_clip to exactly 100 so we don't end up with select=100 + quiet_outdoor OFF.
    # ------------------------------------------------------------
    stable_or_above = (abs(err) <= deadband) or (err <= 0.0)
    if unit_has_quiet and stable_or_above and dem_clip >= 100.0 - 1e-6:
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
    # - For dem_clip <= 100: command select to that value.
    # - For dem_clip > 100 (up to 105): command select=100 and ensure quiet_outdoor is OFF.
    #   For dem_clip == 100: ensure quiet_outdoor is ON.
    desired_layer = float(dem_clip)
    desired_base = desired_layer if desired_layer < 100.0 else 100.0

    desired_option = _snap_to_select(SELECT, desired_base, 0)

    # Set select option if needed
    if desired_option and desired_option != prev_str:
        select.select_option(entity_id=SELECT, option=desired_option)

    # Quiet outdoor switching only matters at max demand layer.
    if QUIET_SW and desired_base >= 100.0 - 1e-6:
        # NEW FIX: When stable/above setpoint, always keep quiet ON at 100.
        if stable_or_above:
            want_quiet_on = True
        else:
            want_quiet_on = (desired_layer <= 100.0 + 1e-6)

        if want_quiet_on and (quiet_on is False):
            try:
                switch.turn_on(entity_id=QUIET_SW)
                quiet_on = True
            except Exception:
                pass
        elif (not want_quiet_on) and (quiet_on is True):
            try:
                switch.turn_off(entity_id=QUIET_SW)
                quiet_on = False
            except Exception:
                pass
    elif QUIET_SW and desired_base < 100.0 - 1e-6:
        # If we were previously in the extra layer (quiet OFF) and we drop below 100,
        # restore quiet ON (safe default).
        if prev >= 100.0 - 1e-6 and (quiet_on is False):
            try:
                switch.turn_on(entity_id=QUIET_SW)
                quiet_on = True
            except Exception:
                pass

    # Update previous error for soft-landing detection next tick
    _prev_err[unit_name] = err

    theta_str = "[" + ", ".join([str(round(float(v), 4)) for v in theta]) + "]"
    cool_str  = "ACTIVE" if in_cooldown else "off"
    icing_str = "ON" if in_icing_band else "off"
    log.info(
        "Daikin ML (%s): ctx=%s | Tin=%.2f°C, Tout=%.2f°C (bucket=%d, →%.1f°C), "
        "DEFROST=%s, cooldown=%s, rate_bad=%s, "
        "icing_band=%s, band_upper=%.0f%%, min_floor=%.0f%%, theta=%s | "
        "SP=%.2f, step_limit=%.1f, deadband=%.2f | "
        "prev=%s → opt≈%.0f%% → target≈%.0f%% → clip=%.0f%% → select=%s",
        unit_name, ctx, Tin, Tout_raw, Tout_bucket, Tout_future,
        str(defrosting), cool_str, str(rate_bad),
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


# ============================================================
# 8) STORE PERSIST (optional helper)
# ============================================================
@service
def daikin_ml_persist():
    """Persistoi store-entiteetit (varmistus)."""
    _persist_all_stores()
