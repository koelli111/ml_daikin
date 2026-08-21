# pyscript/daikin_ml_multi_config.py
# Configuration for the shared-zone Daikin controller.
# Release: 2026-08-21-separate-areas-r10-r2
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
        "ROLE": "assist",

        "INDOOR": "sensor.apollo_temp_1_95d874_temperature_probe",
        "SEPARATE_AREA_NAME": "Daikin1 area",
        "SEPARATE_INDOOR": "sensor.faikin_ac_home",
        "SEPARATE_SETPOINT_HELPER": "input_number.daikin1_area_setpoint",
        "SEPARATE_MODE_HELPER": "input_select.daikin1_area_hvac_mode",
        "SEPARATE_SUMMER_HELPER": "input_boolean.daikin1_area_summer_mode",
        "SEPARATE_MIN_ON_HELPER": "input_number.daikin1_area_minimum_on_minutes",
        "SEPARATE_MIN_OFF_HELPER": "input_number.daikin1_area_minimum_off_minutes",
        "SEPARATE_ALLOWED_MODES": ["off", "auto", "heat", "fan"],
        #"INDOOR2": "sensor.living_room_lampotila",  # optional secondary indoor temperature sensor (averaged with INDOOR when valid)
        "OUTDOOR": "sensor.iv_tulo_lampotila",

        "SELECT": "select.faikin_demand_control",
        "LIQUID": "sensor.faikin_liquid",
        "FAN_RPM_SENSOR": "sensor.faikin_fanfreq",

        # Canonical Daikin1 mappings. Keep these values in future releases
        # unless the corresponding Home Assistant entity IDs are renamed.
        #
        # Optional but strongly recommended for dual-unit efficiency learning.
        # POWER_SENSOR must report instantaneous electrical power in watts.
        # COP_SENSOR must report the unit's live calculated COP. A valid live
        # value is displayed immediately. It contributes to learned statistics
        # only after confirmed continuous compressor operation and is combined
        # across units by weighting each COP with that unit's electrical power.
        # CLIMATE must expose hvac_action. Learning accepts data only while its
        # hvac_action confirms heating/cooling/drying. HEATING_ENTITY remains a
        # defrost/running aid but does not replace hvac_action for learning.
        "POWER_SENSOR": "sensor.daikin_p40_power",
        "COP_SENSOR": "sensor.daikin_cop",
        # REQUIRED for heat/cool/dry on/off and mode control.
        "CLIMATE": "climate.faikin_mqtt_hvac",
        "HEATING_ENTITY": "sensor.faikin_comp",
        "HEATING_THRESHOLD": 0.0,
        # Recognized True/False states are authoritative even when unchanged;
        # fresh liquid temperature remains the fallback for invalid states.
        "DEFROST_ENTITY": "sensor.daikin1_defrost",

        # Relative rated/effective capacity. Keep 1.0 for equal-sized units.
        "CAPACITY_WEIGHT": 1.0,

        # The shared effective-setpoint supervisor must own mode and standby.
        # This has no effect until CLIMATE above is configured.
        "ALLOW_HVAC_OFF": True,
        "ALLOW_HVAC_MODE_CONTROL": True,
        # Requires fan_only in the climate entity's hvac_modes attribute.
        "ALLOW_FAN_ONLY_COIL_DRY": True,

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

    {
        "name": "daikin2",
        "ROLE": "lead",

        "INDOOR": "sensor.apollo_temp_1_95d874_temperature_probe",
        "SEPARATE_AREA_NAME": "Daikin2 area",
        "SEPARATE_INDOOR": "sensor.apollo_temp_1_95d874_temperature_probe",
        "SEPARATE_SETPOINT_HELPER": "input_number.daikin2_area_setpoint",
        "SEPARATE_MODE_HELPER": "input_select.daikin2_area_hvac_mode",
        "SEPARATE_SUMMER_HELPER": "input_boolean.daikin2_area_summer_mode",
        "SEPARATE_MIN_ON_HELPER": "input_number.daikin2_area_minimum_on_minutes",
        "SEPARATE_MIN_OFF_HELPER": "input_number.daikin2_area_minimum_off_minutes",
        "SEPARATE_ALLOWED_MODES": ["off", "auto", "heat", "cool", "dry", "fan"],
        # "INDOOR2": None,  # optional secondary indoor temperature sensor
        "OUTDOOR": "sensor.iv_tulo_lampotila",

        "SELECT": "select.94a9903980ec_demand_control",
        "LIQUID": "sensor.94a9903980ec_liquid",
        "FAN_RPM_SENSOR": "sensor.94a9903980ec_fanfreq",
        "POWER_SENSOR": "sensor.p40n_power",  # instantaneous W
        "COP_SENSOR": "sensor.daikin2_cop",   # live calculated COP
        "CLIMATE": "climate.94a9903980ec_mqtt_hvac",
        "HEATING_ENTITY": "sensor.94a9903980ec_comp",
        "HEATING_THRESHOLD": 0.0,
        "CAPACITY_WEIGHT": 1.0,

        # True lets the controller stop Daikin 2 above the 3/4 C hysteresis
        # and evaluate primary-only operation when it is more efficient.
        "ALLOW_HVAC_OFF": True,
        "ALLOW_HVAC_MODE_CONTROL": True,
        "ALLOW_FAN_ONLY_COIL_DRY": True,
        # Explicit state is authoritative when it is a recognized True/False
        # value, even if the entity has not changed or reported recently.
        "DEFROST_ENTITY": "sensor.daikin2_defrost",
        "QUIET_OUTDOOR_SWITCH": "switch.94a9903980ec_quiet_outdoor",

        "SP_HELPER": "input_number.daikin_setpoint",
        "STEP_LIMIT_HELPER": "input_number.daikin_step_limit",
        "DEADBAND_HELPER": "input_number.daikin_deadband",
        "ICING_CAP_HELPER": "input_number.daikin_icing_cap",
        "SP_BASE_HELPER": "input_number.daikin_setpoint_base",
        "PRICE_BIAS_HELPER": "input_number.daikin_price_bias_points",
        "PRICE_BIAS_ENABLED": "input_boolean.nordpool_bias_enabled",
        "MIN_TEMP_GUARD_HELPER": "input_number.daikin_min_temp_guard",
        "MAX_TEMP_GUARD_HELPER": "input_number.daikin_max_temp_guard",
        "MIN_DEM_FLOOR_M05_M10": "input_number.daikin_min_dem_m05_m10",
        "MIN_DEM_FLOOR_M11_M15": "input_number.daikin_min_dem_m11_m15",
        "MIN_DEM_FLOOR_LE_M16":  "input_number.daikin_min_dem_le_m16",

        "MAX_DEM_CAP_M05_M10": "input_number.daikin_max_dem_m05_m10",
        "MAX_DEM_CAP_M11_M15": "input_number.daikin_max_dem_m11_m15",
        "MAX_DEM_CAP_LE_M16":  "input_number.daikin_max_dem_le_m16",

        "DEMAND_CHANGE_MIN_INTERVAL_HELPER": "input_number.daikin_demand_change_min_interval_s",

        "POST_DEFROST_DEMAND_HELPER": "input_number.daikin_post_defrost_demand_pct",
        "POST_DEFROST_HOLD_MINUTES_HELPER": "input_number.daikin_post_defrost_hold_minutes",
    },
]

# Entity IDs can be misleading. The sensor below is physically an external
# outdoor-temperature probe even though its Finnish entity ID contains "tulo".
# The commissioning validator therefore treats it as explicitly confirmed
# outdoor air. Add an entity here only after verifying its physical location.
DUAL_CONFIRMED_OUTDOOR_SENSORS = [
    "sensor.iv_tulo_lampotila",
]

# ============================================================
# 2) SHARED PHYSICAL LIMITS
# ============================================================
MIN_DEM    = 30.0
MAX_DEM    = 100.0

# Extra "layer" for max demand using quiet outdoor switch:
#  - Demand 100 with quiet_outdoor ON  => effective demand 100
#  - Demand 100 with quiet_outdoor OFF => effective demand 105
MAX_DEM_LAYER = 105.0

GLOBAL_MILD_MAX = 95.0

ICING_BAND_MIN = -2.0
ICING_BAND_MAX =  4.0
ICING_BAND_CAP_DEFAULT = 80.0

AUTO_STEP_MIN  = 3.0
AUTO_STEP_MAX  = 20.0
AUTO_STEP_BASE = 10.0

AUTO_DB_MIN    = 0.05
AUTO_DB_MAX    = 0.50
AUTO_DB_BASE   = 0.10

# Shared PI integral leak inside the deadband.
TRACK_I_LEAK = 0.07

# ------------------------------------------------------------
# Default minimum interval between applied demand changes (seconds)
# This is overridden per-unit by DEMAND_CHANGE_MIN_INTERVAL_HELPER if present.
# ------------------------------------------------------------
DEMAND_CHANGE_MIN_INTERVAL_DEFAULT_S = 60.0

# Shared-zone temperature-rate window.
TIN_SLOPE_WINDOW_S = 15 * 60.0
TIN_SLOPE_MIN_SPAN_S = 12 * 60.0  # require at least 12 minutes of continuous confirmed operation
TIN_SLOPE_MIN_SAMPLES = 3

NORDPOOL_WINDOW_UPDATE_EVERY_MIN = 15


# ------------------------------------------------------------
# Defrost hold behavior
# - Freeze to pre-defrost demand during defrost
# - Keep same demand for N minutes after defrost ends (user selectable)
# - During hold: do not collect/use Tin measurements
# ------------------------------------------------------------
DEFROST_LIQUID_THRESHOLD = 20.0
# A liquid-temperature defrost is accepted only in heat mode.  Use two samples
# to reject a single bad reading; the controller retains pre-defrost heating
# evidence during this debounce.
DEFROST_DEBOUNCE_SAMPLES = 2
DEFROST_EXIT_HYSTERESIS_C = 2.0

# DEFAULTS (used if helpers missing)
POST_DEFROST_HOLD_DEFAULT_MIN = 0.0           # 0 = resume coordinated control immediately
POST_DEFROST_DEMAND_DEFAULT_PCT = 60.0        # %

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

# Dynamic mapping: colder -> shorter Nordpool averaging window (hours)
AVG_WINDOW_MIN_H = 2.0
AVG_WINDOW_MAX_H = 8.0
AVG_WINDOW_TEMP_COLD = -20.0   # at / below this => AVG_WINDOW_MIN_H
AVG_WINDOW_TEMP_WARM = 0.0     # at / above this => AVG_WINDOW_MAX_H
AVG_WINDOW_WRITE_EPS = 0.1     # don't spam helper writes


# ============================================================
# 3) SHARED-ZONE COORDINATION + ENERGY-EFFICIENCY LEARNING
# ============================================================
# Both configured units share one heat/logical-cool/dry supervisor. Daikin2 is
# the heating lead and Daikin1 may assist heating. In logical cool and dry,
# Daikin2 is the fixed refrigeration unit and Daikin1 is restricted to heat/
# reheat; neither runs an independent controller against the same room.
DUAL_ZONE_ENABLED = True
DUAL_PRIMARY_UNIT = "daikin2"
# Heating-only UI override. Off preserves Daikin2 as the first heating unit;
# on starts Daikin1 first and makes it the sole eligible mild-weather heater.
# Cooling, drying and coordinated-reheat roles are unaffected.
DUAL_DAIKIN1_HEATING_LEAD_HELPER = "input_boolean.daikin1_heating_lead"

# Persistent requested mode.  ``auto`` chooses heat/cool/dry/idle from the
# shared temperature and humidity sensors.  Manual modes remain available for
# commissioning. ``off + FAN`` runs both indoor fans without compressor
# conditioning. ``auto + fan`` uses normal controller auto selection and
# uses lowMedium for idle ventilation and summer conditioning, while active
# heating uses mediumHigh. It never requests physical
# auto or heat_cool.
# Both added modes use lowMedium while ventilating. Only off + FAN bypasses the thermostat and
# allocator; auto + fan retains them. Cool and dry are permitted only while
# summer mode is ON.
DUAL_HVAC_MODE_HELPER = "input_select.daikin_dual_hvac_mode"
DUAL_DEFAULT_HVAC_MODE = "auto"
DUAL_SUMMER_MODE_HELPER = "input_boolean.daikin_summer_mode"
DUAL_MODE_STATUS_SENSOR = "sensor.daikin_dual_mode_status"
DUAL_SHARED_FAN_MODE = "lowMedium"

# Automatic mode selection uses a wider neutral band than the individual
# manual thermostats.  This prevents heat/cool oscillation. A new candidate
# must remain valid for the confirmation interval, except severe low
# temperature can request heat immediately. Humidity-driven dry changes to the
# temperature-driven logical cooling controller only after the room becomes
# clearly warm. Temperature cooling uses physical ``cool`` while humidity
# drying uses the Daikin's native physical ``dry`` mode.
DUAL_AUTO_HEAT_ON_DELTA_C = 0.40
DUAL_AUTO_COOL_ON_DELTA_C = 0.40
DUAL_AUTO_MODE_CONFIRMATION_MINUTES = 3.0
DUAL_AUTO_EMERGENCY_HEAT_DELTA_C = 1.0
DUAL_AUTO_DRY_TO_COOL_DELTA_C = 0.60
DUAL_AUTO_HEAT_ON_DELTA_HELPER = "input_number.daikin_auto_heat_on_delta"
DUAL_AUTO_COOL_ON_DELTA_HELPER = "input_number.daikin_auto_cool_on_delta"
DUAL_AUTO_CONFIRMATION_HELPER = "input_number.daikin_auto_mode_confirmation_minutes"

# After a cool/dry compressor cycle, run only the indoor fan to dry the coil.
# The compressor off-time starts when cooling/drying ends and continues during
# this fan cycle. Any real heat/cool/dry demand interrupts fan_only immediately.
DUAL_COIL_DRY_ENABLED = True
DUAL_COIL_DRY_MINUTES = 15.0
DUAL_COIL_DRY_MINUTES_HELPER = "input_number.daikin_coil_dry_minutes"
DUAL_COIL_DRY_HVAC_MODE = "fan_only"

# The external shared-zone sensor and effective setpoint decide plant on/off.
# The physical Daikin target is deliberately held at an extreme value while
# active so the shared demand controller, rather than the unit's internal room
# thermostat, regulates output. Both summer modes are role-fixed: Daikin2 is
# the only unit allowed to provide temperature cooling or humidity drying, and
# use their corresponding physical ``cool`` and ``dry`` modes at 16 C and the
# advertised ``lowMedium`` option (fan level 2/5). The reconciler waits for the
# requested physical mode to be acknowledged before writing the authoritative
# 16 C target. Daikin1 is
# reserved for physical heat/reheat and every other physical role restores the
# fan setting to the advertised automatic option.
DUAL_SET_CLIMATE_TEMPERATURE = True
DUAL_HEAT_HVAC_TARGET_C = 31.0
DUAL_COOL_HVAC_TARGET_C = 16.0
DUAL_DRY_HVAC_TARGET_C = 16.0
DUAL_DRY_HVAC_MODE = "dry"
DUAL_COOLING_UNIT = "daikin2"
DUAL_DRYING_UNIT = "daikin2"
DUAL_DRY_REHEAT_UNIT = "daikin1"
# Daikin2 advertises: Auto, night, Low, lowMedium, Medium, mediumHigh, High.
DUAL_HEAT_FAN_MODE = "mediumHigh"
DUAL_COOL_FAN_MODE = "lowMedium"
DUAL_DRY_FAN_MODE = "lowMedium"
DUAL_DEFAULT_FAN_MODE = "auto"

# ============================================================
# 3A) SAFETY, COMMAND ACKNOWLEDGEMENT, RESTART STATE, DIAGNOSTICS
# ============================================================
# A separate Home Assistant automation in daikin_watchdog_package.yaml watches
# this heartbeat and reports anomalies. It is alert-only and never turns HVAC
# off when this controller fails to load or stops executing.
DUAL_CONTROLLER_VERSION = "2026-08-21-separate-areas-r10-r2"
DUAL_SEPARATE_AREAS_HELPER = "input_boolean.daikin_separate_areas"
DUAL_SEPARATE_AREAS_SENSOR = "sensor.daikin_separate_areas_status"
DUAL_SEPARATE_KP = 25.0
DUAL_SEPARATE_KI = 3.0
DUAL_SEPARATE_KD = 4.0
DUAL_SEPARATE_I_CLAMP = 25.0
DUAL_SEPARATE_CONTROL_INTERVAL_S = 20.0
DUAL_SEPARATE_PREDICTION_MINUTES = 10.0

# Daikin2 uses a fixed Demand select during physical cooling and physical
# drying. This invariant bypasses adaptive allocation, bounds, step limiting,
# post-defrost Demand holds and the normal Demand-change interval.
DUAL_DAIKIN2_SUMMER_DEMAND = 100.0
# Normal Summer Mode heating Demand. The adaptive CDP governor may temporarily
# exceed it up to DUAL_CDP_MAX_DEMAND when suppressed airflow is confirmed.
DUAL_SUMMER_HEAT_DEMAND_CAP = 30.0

# Adaptive Cold Draft Prevention release during Summer Mode heating. The
# ordinary cap remains 30. A confirmed-heating unit whose requested
# mediumHigh fan is persistently suppressed may rise in five-point Demand
# steps until airflow increases materially. The algorithm works in the fan
# sensor's native scale; it does not assume that fanfreq is literal RPM.
DUAL_CDP_RELEASE_ENABLED = True
DUAL_CDP_STARTUP_SECONDS = 20.0
DUAL_CDP_LOW_CONFIRM_SECONDS = 20.0
DUAL_CDP_STEP_INTERVAL_SECONDS = 20.0
DUAL_CDP_DEMAND_STEP = 5.0
DUAL_CDP_MAX_DEMAND = 60.0
# CDP is suspected below 75% of the calibrated requested-speed RPM and
# considered released above 85%. The gap provides hysteresis.
DUAL_CDP_RESTRICTED_RPM_RATIO = 0.75
DUAL_CDP_RELEASED_RPM_RATIO = 0.85
DUAL_CDP_RELEASE_CONFIRM_SECONDS = 20.0
DUAL_CDP_PROBE_DOWN_AFTER_SECONDS = 180.0
DUAL_CDP_LEARNING_STABLE_SECONDS = 180.0
DUAL_CDP_FAILURE_CONFIRM_SECONDS = 20.0
DUAL_CDP_OUTDOOR_BUCKET_C = 2.0
DUAL_CDP_OUTDOOR_STABILITY_C = 3.0
DUAL_CDP_MAX_LOOKUP_DISTANCE_C = 6.0
DUAL_CDP_FULL_CONFIDENCE = 3
DUAL_CDP_CONFIDENCE_DECAY_DAYS = 45.0
DUAL_FAN_CALIBRATION_SENSOR = "sensor.daikin_dual_fan_rpm_calibration"
DUAL_CDP_CURVE_SENSOR = "sensor.daikin_dual_cdp_temperature_curve"
DUAL_FAN_CALIBRATION_SETTLE_SECONDS = 45.0
DUAL_FAN_CALIBRATION_SAMPLE_SECONDS = 30.0
DUAL_FAN_CALIBRATION_SAMPLE_INTERVAL_SECONDS = 5.0
DUAL_HEARTBEAT_SENSOR = "sensor.daikin_dual_controller_heartbeat"
DUAL_HEALTH_SENSOR = "sensor.daikin_dual_health"
DUAL_COMMAND_STATUS_SENSOR = "sensor.daikin_dual_command_status"
DUAL_DAILY_STATS_SENSOR = "sensor.daikin_dual_daily_stats"
DUAL_COMMISSIONING_SENSOR = "sensor.daikin_dual_commissioning"
DUAL_REPLAY_SENSOR = "sensor.daikin_dual_history_replay"
DUAL_LANDING_STATUS_SENSOR = "sensor.daikin_dual_landing_status"
DUAL_COP_EFFICIENCY_SENSOR = "sensor.daikin_dual_cop_efficiency"
DUAL_RUNTIME_STORE_ENTITY = "pyscript.daikin_dual_runtime_store"

# Physical outputs are suppressed when this helper is on. The complete
# supervisor, demand calculator and diagnostics continue to run and report
# would-send commands, but learning and optimizer episodes are blocked.
DUAL_CONTROLLER_SHADOW_HELPER = "input_boolean.daikin_controller_shadow_mode"

# Sensor maximum ages use Home Assistant's last_reported timestamp when
# available, then last_updated. A finite indoor value remains valid for comfort
# control even when its report timestamp is old. Indoor report staleness is a
# diagnostic warning and suspends learning/optimization until fresh reporting
# resumes; it never freezes HVAC commands. Other sensor roles retain their
# feature-specific freshness requirements below. A recognized explicit defrost
# state is also state-based rather than heartbeat-based: its age is diagnostic
# only. Liquid-temperature fallback retains its maximum-age requirement.
#
# The single serialized housekeeping loop checks report timestamps every 10 s.
DUAL_SENSOR_REPORT_POLL_SECONDS = 10.0
DUAL_INDOOR_MAX_AGE_S = 5.0 * 60.0
DUAL_HUMIDITY_MAX_AGE_S = 10.0 * 60.0
DUAL_OUTDOOR_MAX_AGE_S = 15.0 * 60.0
DUAL_LIQUID_MAX_AGE_S = 3.0 * 60.0
DUAL_POWER_MAX_AGE_S = 2.0 * 60.0
DUAL_COP_MAX_AGE_S = 2.0 * 60.0

# A missing, unknown, unavailable or nonnumeric indoor sensor value freezes new
# temperature-dependent controller commands while preserving current equipment
# state. A finite but old indoor value remains usable. An unavailable climate
# entity also freezes new commands. No anomaly requests whole-system HVAC off.
# Humidity is mandatory only for logical humidity-driven dry; an active dry
# cycle holds its current state while humidity is invalid. A recognized
# explicit defrost True/False remains usable regardless of age; only invalid
# explicit states fall back to the age-limited liquid detector. Stale liquid
# disables controller-side fallback detection; stale outdoor data suspends
# optimization and uses the conservative fallback. Physical command failures
# remain unit-specific diagnostics.
#
# Critical recovery requires both this elapsed interval and this many
# consecutive healthy safety evaluations. During this interval commands remain
# frozen, but equipment is never switched off by the anomaly policy.
DUAL_CRITICAL_RECOVERY_SECONDS = 60.0
DUAL_CRITICAL_RECOVERY_VALID_READINGS = 4

# When outdoor temperature is stale, control uses the colder of the last-good
# reading and this fallback. This keeps heating available under conservative
# cold-weather Demand bounds, while all outdoor-context learning/optimization
# remains suspended until the real sensor is fresh again.
DUAL_OUTDOOR_STALE_FALLBACK_C = -5.0

# Deprecated compatibility settings retained for older dashboards. Degraded
# fallback target/Demand are not applied by the anomaly-hold policy.
DUAL_DEGRADED_RECOVERY_MINUTES = 1.0
DUAL_DEGRADED_FALLBACK_TARGET_C = 22.0
DUAL_DEGRADED_DEMAND = 100.0
DUAL_STARTUP_DEMAND_HOLD_S = 60.0

# Command verification is asynchronous. Mode/target/Demand/Quiet/off requests
# are checked every 10 seconds. A valid but not-yet-acknowledged expectation is
# never made terminal merely because the unit still reports another valid
# state. It enters bounded reconciliation cycles with exponential backoff and
# remains eligible until acknowledged or superseded by a newer desired state.
DUAL_COMMAND_RETRY_SECONDS = 10.0
DUAL_COMMAND_MAX_ATTEMPTS = 3
DUAL_COMMAND_RECOVERY_INITIAL_SECONDS = 30.0
DUAL_COMMAND_RECOVERY_MAX_SECONDS = 5.0 * 60.0
DUAL_COMMAND_RECOVERY_BACKOFF_FACTOR = 2.0
# Faikin/Home Assistant can optimistically publish ``cool`` before the indoor
# unit has completed a transition from native ``dry``.  Delay the authoritative
# 16 C write until the requested physical mode has remained visible for two housekeeping
# observations, then keep the temperature command pending until 16 C has also
# remained stable.  This prevents a delayed remembered-target report (18 C)
# from being accepted after the first apparent acknowledgement.
DUAL_SUMMER_MODE_SETTLE_SECONDS = 20.0
DUAL_SUMMER_TARGET_ACK_STABLE_SECONDS = 20.0
# A mode transition may briefly restore a remembered cooling setpoint such as
# 18 C. Give the authoritative 16 C target a longer bounded acknowledgement
# window. If it remains at 18 C, normal fail-tolerant reconciliation continues;
# it never accepts 18 C as the requested summer target.
DUAL_TARGET_COMMAND_MAX_ATTEMPTS = 6
DUAL_COMMAND_FAILURE_RETENTION_S = 30.0 * 60.0

# Daily counters are stored in the restart-safe runtime store. The replay ring
# is memory-only to avoid large recorder writes; 24 hours at five-minute
# resolution is 288 samples.
DUAL_STATS_MAX_SAMPLE_GAP_S = 120.0
DUAL_COMFORT_DEVIATION_C = 0.40
DUAL_REPLAY_RETENTION_HOURS = 24.0
DUAL_REPLAY_SAMPLE_INTERVAL_S = 5.0 * 60.0

# Existing price-bias points are heating-oriented. Invert their temperature
# direction in cool/dry so the same cheap/expensive signal remains sensible.
DUAL_INVERT_COOLING_PRICE_BIAS = True

# Shared supervisory thermostat helpers (all included in the YAML package).
DUAL_HEAT_ON_DELTA_C = 0.15
DUAL_HEAT_OFF_DELTA_C = 0.10
DUAL_COOL_ON_DELTA_C = 0.20
DUAL_COOL_OFF_DELTA_C = 0.10
DUAL_HEAT_ON_DELTA_HELPER = "input_number.daikin_heat_on_delta"
DUAL_HEAT_OFF_DELTA_HELPER = "input_number.daikin_heat_off_delta"
DUAL_COOL_ON_DELTA_HELPER = "input_number.daikin_cool_on_delta"
DUAL_COOL_OFF_DELTA_HELPER = "input_number.daikin_cool_off_delta"
DUAL_MIN_ON_MINUTES_DEFAULT = 20.0
DUAL_MIN_OFF_MINUTES_DEFAULT = 10.0
DUAL_MIN_ON_MINUTES_HELPER = "input_number.daikin_minimum_on_minutes"
DUAL_MIN_OFF_MINUTES_HELPER = "input_number.daikin_minimum_off_minutes"

# Drying demand. Invalid/unavailable humidity disables dry mode safely. With
# two suitable units, the lead physically dries while the assist heats to hold
# room temperature. A single configured unit retains the original temperature
# headroom/floor behavior.
DUAL_HUMIDITY_SENSOR = "sensor.apollo_temp_1_95d874_board_humidity"
DUAL_HUMIDITY_FILTER_MINUTES = 10.0
DUAL_HUMIDITY_TARGET_DEFAULT = 55.0
DUAL_HUMIDITY_ON_DELTA_DEFAULT = 3.0
DUAL_DRY_TEMP_MARGIN_C = 0.20
DUAL_HUMIDITY_TARGET_HELPER = "input_number.daikin_humidity_target"
DUAL_HUMIDITY_ON_DELTA_HELPER = "input_number.daikin_humidity_on_delta"
DUAL_DRY_TEMP_MARGIN_HELPER = "input_number.daikin_dry_temperature_margin"

# Optional shared room sensors. If empty, the primary unit's INDOOR/INDOOR2
# sensors are used. Use room sensors outside the direct discharge air streams.
DUAL_ZONE_INDOOR_SENSORS = [
    "sensor.apollo_temp_1_95d874_temperature_probe",
]

# Daikin 2 becomes eligible below this measured outdoor temperature. Hysteresis
# keeps the mode from toggling when the sensor moves around 3 C.
DUAL_ASSIST_ON_BELOW_C_DEFAULT = 3.0
DUAL_ASSIST_OFF_ABOVE_C_DEFAULT = 4.0
DUAL_ASSIST_ON_BELOW_HELPER = "input_number.daikin_dual_assist_on_below"
DUAL_ASSIST_OFF_ABOVE_HELPER = "input_number.daikin_dual_assist_off_above"

# Retained for helper compatibility only. Cooling is now always Daikin2-only,
# so these former second-cooling-unit staging thresholds are not used.
DUAL_COOL_ASSIST_ON_DELTA_C = 0.60
DUAL_COOL_ASSIST_OFF_DELTA_C = 0.15
DUAL_COOL_ASSIST_ON_DELTA_HELPER = "input_number.daikin_cool_assist_on_delta"
DUAL_COOL_ASSIST_OFF_DELTA_HELPER = "input_number.daikin_cool_assist_off_delta"
DUAL_COOL_RESPONSE_MINUTES = 30.0
DUAL_COOL_MIN_PROGRESS_CPH = 0.05

# Coordinated humidity drying / sensible reheat. This path requires the two
# explicit, distinct units above. Daikin2 removes humidity in physical ``dry``
# at 16 C and the ``lowMedium`` fan option (level 2/5). Daikin1 is staged in
# physical ``heat`` before the
# projected indoor temperature falls below the effective setpoint.
DUAL_DRY_REHEAT_ENABLED = True
DUAL_DRY_REHEAT_TARGET_OFFSET_C = 0.10
DUAL_DRY_REHEAT_ON_MARGIN_C = 0.10
DUAL_DRY_REHEAT_OFF_MARGIN_C = 0.30
DUAL_DRY_REHEAT_PREDICTION_MINUTES = 5.0
DUAL_DRY_REHEAT_MIN_ON_MINUTES = 10.0
DUAL_DRY_REHEAT_MIN_OFF_MINUTES = 5.0
DUAL_DRY_REHEAT_INITIAL_DEMAND = 35.0
DUAL_DRY_REHEAT_MIN_DEMAND = 30.0
DUAL_DRY_REHEAT_MAX_DEMAND = 70.0
DUAL_DRY_REHEAT_KP = 30.0
DUAL_DRY_REHEAT_KI = 4.0
DUAL_DRY_REHEAT_KD = 10.0
DUAL_DRY_REHEAT_I_CLAMP = 20.0

# If reheat cannot arrest the temperature fall, pause the lead's drying
# compressor at this floor while the assist continues heating. Drying resumes
# only after the room has recovered above the separate resume threshold and
# the lead has observed its minimum off time.
DUAL_DRY_LEAD_PAUSE_BELOW_SETPOINT_C = 0.30
DUAL_DRY_LEAD_RESUME_BELOW_SETPOINT_C = 0.05
DUAL_DRY_LEAD_MIN_OFF_MINUTES = 5.0

# Shared zone demand controller. Demand is expressed as combined effective
# demand-points, so two 40% units equal roughly 80 initial demand-points. The
# energy optimizer learns which allocation actually uses the least electricity.
DUAL_TRACK_KP = 10.0
DUAL_TRACK_KI = 2.0
DUAL_TRACK_KD = 0.5
# Mode-specific values fall back to the shared constants above. Cooling uses
# normalized error/rate direction. Dry converts humidity error to comparable
# controller units before applying its gentler PI values.
DUAL_HEAT_TRACK_KP = 10.0
DUAL_HEAT_TRACK_KI = 2.0
DUAL_HEAT_TRACK_KD = 0.5
DUAL_COOL_TRACK_KP = 10.0
DUAL_COOL_TRACK_KI = 2.0
DUAL_COOL_TRACK_KD = 0.5
DUAL_DRY_TRACK_KP = 8.0
DUAL_DRY_TRACK_KI = 1.0
DUAL_DRY_TRACK_KD = 0.3
DUAL_DRY_RH_ERROR_TO_CONTROL = 0.10
DUAL_TRACK_I_CLAMP = 35.0
DUAL_TOTAL_STEP_DEFAULT = 12.0
DUAL_TOTAL_STEP_HELPER = "input_number.daikin_dual_total_step_limit"
DUAL_RATE_CLAMP_CPH = 2.0

# Predictive heat soft landing. The ordinary PI controller still calculates
# every five minutes. A reduction-only governor normally acts once per minute,
# but switches to a 20-second cadence when a separate short-window rate
# confirms that the room is rapidly approaching the setpoint. Both the total
# and per-unit step limits remain mandatory. The rapid path can only lower
# Demand; it never raises output.
DUAL_HEAT_LANDING_START_C = 0.40
DUAL_HEAT_PREDICTION_MINUTES = 10.0
DUAL_FAST_TAPER_INTERVAL_S = 60.0
DUAL_RAPID_TAPER_INTERVAL_S = 20.0
DUAL_RAPID_TAPER_STEP_PER_ACTIVE_UNIT = 5.0
DUAL_RAPID_APPROACH_RATE_CPH = 0.30
DUAL_FAST_RATE_WINDOW_S = 180.0
DUAL_FAST_RATE_MIN_SPAN_S = 60.0
DUAL_FAST_RATE_MIN_SAMPLES = 4
DUAL_HEAT_HARD_STOP_ABOVE_C = 0.25
DUAL_HEAT_LANDING_I_DECAY = 0.50

# Active minimum-sustain discovery. After actual-running learning is available
# and conditions are steady near setpoint, test one 5-point-lower combined
# Demand level, or validate an already lower landing level, for 15 settled
# minutes. Accept it only if temperature remains stable; abort immediately if
# the room starts falling away.
DUAL_HEAT_SUSTAIN_PROBE_ENABLED = True
DUAL_HEAT_SUSTAIN_PROBE_STEP = 5.0
DUAL_HEAT_SUSTAIN_PROBE_HOLD_MINUTES = 15.0
DUAL_HEAT_SUSTAIN_PROBE_START_ERR_C = 0.10
DUAL_HEAT_SUSTAIN_PROBE_FAIL_ERR_C = 0.20
DUAL_HEAT_SUSTAIN_PROBE_MAX_FALL_CPH = 0.05
DUAL_HEAT_SUSTAIN_PROBE_SETTLE_TOLERANCE = 2.6
DUAL_HEAT_SUSTAIN_PROBE_COOLDOWN_MINUTES = 30.0

# Shared sustaining-demand learner (separate values per measured outdoor bucket)
DUAL_SUSTAIN_ALPHA = 0.05
DUAL_SUSTAIN_ERR_MAX_C = 0.20
DUAL_SUSTAIN_RATE_MAX_CPH = 0.08
DUAL_INITIAL_TOTAL_DEMAND = 80.0
DUAL_HEAT_INITIAL_TOTAL_DEMAND = 80.0
DUAL_COOL_INITIAL_TOTAL_DEMAND = 70.0
DUAL_DRY_INITIAL_TOTAL_DEMAND = 55.0

# Refrigeration demand ranges. Heating continues to use its outdoor-dependent
# floor, icing cap and cold-weather maximum logic.
DUAL_COOL_MIN_DEMAND = 30.0
DUAL_COOL_MAX_DEMAND = 100.0
DUAL_DRY_MIN_DEMAND = 30.0
DUAL_DRY_MAX_DEMAND = 70.0
DUAL_ZONE_STORE_ENTITY = "pyscript.daikin_dual_zone_store"
DUAL_TOTAL_DEMAND_SENSOR = "sensor.daikin_dual_total_demand"
DUAL_OPTIMIZER_SENSOR = "sensor.daikin_dual_efficiency_optimizer"
DUAL_LEARNING_STATUS_SENSOR = "sensor.daikin_dual_learning_status"

# Off/idle/fan_only and mode/allocation transitions clear the response window.
# The controller collects learning samples only while climate.hvac_action
# continuously confirms every expected unit. No sustain or optimizer update is
# allowed before this minimum actual-running span has elapsed. Keep this at
# least as long as TIN_SLOPE_MIN_SPAN_S (the controller enforces that minimum).
DUAL_LEARNING_MIN_ACTIVE_MINUTES = 12.0
DUAL_LEARNING_SAMPLE_MIN_INTERVAL_S = 30.0

# Optional combined instantaneous power sensor in watts. Leave None when each
# unit has its own POWER_SENSOR. A dedicated combined Daikin meter is valid;
# whole-house power is not, because unrelated loads would corrupt comparisons.
DUAL_TOTAL_POWER_SENSOR = None  # e.g. "sensor.daikin_heat_pumps_power"
DUAL_TOTAL_POWER_SENSOR_SCALE = 1.0

# Home Assistant helpers supplied in daikin_dual_helpers.yaml
DUAL_OPTIMIZER_MODE_HELPER = "input_select.daikin_dual_optimizer_mode"
DUAL_MANUAL_OVERRIDE_HELPER = "input_boolean.daikin_dual_optimizer_manual_override"

# Candidate allocation policies. Shares are capacity-weighted and normalized.
# Three policies keep learning tractable while retaining both single-unit
# choices and balanced operation. Existing action IDs are retained so learned
# statistics migrate without translation.
# A zero share requires CLIMATE plus ALLOW_HVAC_MODE_CONTROL=True for that unit;
# otherwise the action is excluded because standby cannot be guaranteed. In
# Daikin2-only heating, Daikin1's zero share is physical fan_only at lowMedium;
# its measured blower electricity remains part of the policy score.
DUAL_ACTIONS = [
    {"id": "balanced",    "shares": [0.50, 0.50]},
    {"id": "lead1_only",  "shares": [1.00, 0.00]},
    {"id": "lead2_only",  "shares": [0.00, 1.00]},
]
DUAL_DEFAULT_ACTION = "balanced"
DUAL_ALLOW_SINGLE_UNIT_ACTIONS = True

# Slow optimizer. Shadow mode observes/publishes recommendations; auto mode
# safely evaluates eligible actions and applies the best learned policy.
DUAL_EPISODE_MINUTES_DEFAULT = 60.0
DUAL_EPISODE_MINUTES_HELPER = "input_number.daikin_dual_episode_minutes"
DUAL_ACTION_SETTLE_MINUTES = 15.0
DUAL_MIN_VALID_POWER_FRACTION = 0.80
DUAL_MIN_SAMPLES_PER_ACTION = 3
DUAL_EXPLORATION_BONUS = 0.12
DUAL_SAFE_EXPLORE_ERR_C = 0.20
DUAL_ABORT_ERR_C_DEFAULT = 0.60
DUAL_ABORT_ERR_HELPER = "input_number.daikin_dual_abort_error"
DUAL_OUTDOOR_CONTEXT_STEP_C = 2.0

# Score weights. Energy is normalized by indoor-outdoor degree-hours, allowing
# nearby weather conditions to be compared more fairly. Comfort and defrost
# penalties keep the optimizer from selecting a low-energy but poor-heating mode.
DUAL_SCORE_COMFORT_WEIGHT = 2.0
DUAL_SCORE_DEFROST_WEIGHT = 0.20
DUAL_SCORE_HUMIDITY_WEIGHT = 0.05
DUAL_SCORE_CYCLE_WEIGHT = 0.01

# Direct COP efficiency measurement. COP is used as the primary heat-mode
# efficiency term when at least 80% of an episode has valid samples. The
# electrical-energy/degree-hour score remains the automatic fallback.
#
# Combined COP is never a simple average:
#   sum(unit COP * unit electrical W) / sum(unit electrical W)
#
# Heat is enabled by default because these entities represent heating COP.
# Cooling/drying continue to use electrical-energy scoring unless their
# measurement entities are explicitly known to report valid EER/COP there.
DUAL_COP_MODES = ("heat",)
DUAL_COP_MIN_VALUE = 0.50
DUAL_COP_MAX_VALUE = 12.0
DUAL_COP_MIN_UNIT_POWER_W = 150.0
DUAL_COP_MIN_VALID_FRACTION = 0.80
DUAL_COP_USE_FOR_SCORING = True

# Persist a matched operating map by mode, measured outdoor bucket, allocation
# and actual 5-point Demand values. This lets the controller compare examples
# such as lead1_only@70 against balanced@40/40 without blending their loads.
DUAL_COP_CONTEXT_OUTDOOR_STEP_C = 2.0
DUAL_COP_CONTEXT_DEMAND_STEP = 5.0
DUAL_COP_MAX_CONTEXTS = 300
DUAL_COP_MIN_CONTEXT_SECONDS = 10.0 * 60.0

# Store/publish throttling
DUAL_STORE_SAVE_MIN_INTERVAL_S = 30.0 * 60.0

# Existing system-wide defrost freeze safety timeout
SYSTEM_DEFROST_FREEZE_MAX_S = 30.0 * 60.0
