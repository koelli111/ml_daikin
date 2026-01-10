# pyscript/daikin_ml_multi_config.py
# Configuration for Daikin ML MULTI controller.
# - Unit definitions (DAIKINS)
# - Global constants / helper entity IDs
#
# This file is imported by daikin_ml_multi.py.

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
        # NEW: Post-defrost behavior helpers (per unit)
        # - Demand % to hold after defrost ends
        # - Minutes to hold (during this time demand is NOT changed)
        # ------------------------------------------------------------
        "POST_DEFROST_DEMAND_HELPER": "input_number.daikin1_post_defrost_demand_pct",
        "POST_DEFROST_HOLD_MINUTES_HELPER": "input_number.daikin1_post_defrost_hold_minutes",
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
    #
    #     "POST_DEFROST_DEMAND_HELPER": "input_number.daikin2_post_defrost_demand_pct",
    #     "POST_DEFROST_HOLD_MINUTES_HELPER": "input_number.daikin2_post_defrost_hold_minutes",
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
SOFT_ERR_START = 0.3   # °C: start softening when |err| <= this AND moving toward setpoint
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
# Tracking-first demand control: keep Tin near effective setpoint
# (PI(D) feedback wrapped around the inverse-model demand dem_opt)
# ------------------------------------------------------------
# Units:
#  - TRACK_KP: demand-% per °C
#  - TRACK_KI: demand-% per (°C·hour)  (integrates err over time)
#  - TRACK_KD: demand-% per (°C/hour)  (damps using Tin slope)
TRACK_KP = 15.0
TRACK_KI = 10.0
TRACK_KD = 0.8

# Integral clamp in demand-% (prevents runaway)
TRACK_I_CLAMP = 40.0

# Don’t integrate tiny errors (noise)
TRACK_I_ERR_EPS = 0.02
TRACK_I_DEADBAND_FRACTION = 0.5

# Slight integral leak when inside deadband (prevents permanent bias lock-in)
TRACK_I_LEAK = 0.03  # 3% per control tick when |err| <= deadband

# Clamp rate used for KD damping (°C/h)
TRACK_RATE_CLAMP_CPH = 3.0

# Override demand-change min-interval when tracking is clearly off
TRACK_MIN_INTERVAL_OVERRIDE_ERR = 0.25  # °C

# Break out of post-defrost hold early if tracking error is large
HOLD_BREAK_ERR = 0.35  # °C

# ------------------------------------------------------------
# Default minimum interval between applied demand changes (seconds)
# This is overridden per-unit by DEMAND_CHANGE_MIN_INTERVAL_HELPER if present.
# ------------------------------------------------------------
DEMAND_CHANGE_MIN_INTERVAL_DEFAULT_S = 60.0


# ------------------------------------------------------------
# Instant demand control (no intentional delays)
# - Enables immediate reaction to sensor updates (state_trigger)
# - Disables demand-change rate limiting
# - Disables step/cooldown ramp limiting (applies computed demand immediately)
# ------------------------------------------------------------
INSTANT_DEMAND_CONTROL = True
# ------------------------------------------------------------
# Learning uses smoothed 5-minute slope of indoor temperature
# - Keeps a per-unit history of (timestamp, Tin)
# - If not enough history/span, falls back to the derivative sensor
# ------------------------------------------------------------
TIN_SLOPE_WINDOW_S = 5 * 60.0
TIN_SLOPE_MIN_SPAN_S = 4 * 60.0  # require at least ~4 minutes span for "true 5-minute" slope
TIN_SLOPE_MIN_SAMPLES = 3        # require a few points for smoothing/robustness

# ------------------------------------------------------------
# Defrost hold behavior
# - Freeze to pre-defrost demand during defrost
# - Keep same demand for N minutes after defrost ends (user selectable)
# - During hold: do not collect/use Tin measurements
# ------------------------------------------------------------
DEFROST_LIQUID_THRESHOLD = 20.0

# DEFAULTS (used if helpers missing)
POST_DEFROST_HOLD_DEFAULT_MIN = 5.0           # minutes
POST_DEFROST_DEMAND_DEFAULT_PCT = 60.0        # %

# ------------------------------------------------------------
# Store write throttling (Home Assistant state.set is relatively heavy)
# - Only write learned params at most every N seconds per unit unless forced
# ------------------------------------------------------------
STORE_SAVE_MIN_INTERVAL_S = 15 * 60.0

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
AVG_WINDOW_TEMP_COLD = -20.0   # at / below this => AVG_WINDOW_MIN_H
AVG_WINDOW_TEMP_WARM = 0.0     # at / above this => AVG_WINDOW_MAX_H
AVG_WINDOW_WRITE_EPS = 0.1     # don't spam helper writes
