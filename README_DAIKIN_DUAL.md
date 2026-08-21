# Daikin automatic dual controller with COP efficiency learning

Release: `2026-08-21-separate-areas-r10-r2`

R10 adds an optional `input_boolean.daikin_separate_areas` topology. With the
switch off, the original shared-zone controller continues to use only
`sensor.apollo_temp_1_95d874_temperature_probe` and retains heating-lead,
dual-assistance, allocation, combined COP optimization and coordinated
defrost behaviour.

With the switch on, Daikin1 uses `sensor.faikin_ac_home` and Daikin2 uses
`sensor.apollo_temp_1_95d874_temperature_probe`. Each receives an independent
mode, setpoint, Summer Mode, thermostat, PI integral, minimum on/off timing,
temperature-rate and predictive-landing calculation, Demand calculation,
defrost path, CDP floor and COP/Demand learning context.
Shared allocation, heating-lead selection, dual assistance, standby-fan
circulation and cross-unit defrost freezing are bypassed. Daikin1 remains
heating/fan only; Daikin2 retains heat/cool/dry/fan capability.

New helpers are `input_select.daikin1_area_hvac_mode`,
`input_select.daikin2_area_hvac_mode`, `input_number.daikin1_area_setpoint`,
`input_number.daikin2_area_setpoint`, per-area minimum on/off helpers, and
`input_boolean.daikin1_area_summer_mode` /
`input_boolean.daikin2_area_summer_mode`. Configure their desired values before
enabling separate areas. `sensor.daikin_separate_areas_status` publishes both
zone states, sensor validity, temperatures, setpoints, errors, requested and
applied Demand, independent learning, and the active topology.

R9 publishes `sensor.daikin_dual_cdp_temperature_curve`. Its numeric state is
the currently selected highest per-unit CDP Demand floor, so it can be recorded
and graphed by Home Assistant. Attributes include `current_outdoor_temperature`,
`current_floor_by_unit`, the complete learned `curve`, and graph-ready
`daikin1_curve_points` / `daikin2_curve_points` arrays. Each point is
`[outdoor-temperature bucket midpoint C, minimum Demand %]`; detailed record
arrays include the learned insufficient boundary, confidence and sample count.

R8 learns the minimum Demand that continuously keeps CDP released separately
for each unit, calibrated fan setting and 2 C outdoor-temperature bucket. A
minimum is established from a tested boundary: a lower Demand that restored
CDP and the preceding Demand that sustained released airflow. Demand 30 can be
learned after three stable minutes without restriction.

The learned value becomes a temperature-dependent minimum for Summer heating.
Exact buckets are preferred; gaps are conservatively interpolated and rounded
up, with the colder learned neighbor never weakened. Sparse, low-confidence or
aged observations receive a five-point safety margin. Confidence decays every
45 days without reconfirmation. Learning is blocked by invalid outdoor data,
temperature movement over 3 C during a probe, mode/fan mismatch, missing fan
calibration, stopped heating, defrost and post-defrost recovery. Existing
learned records are retained when an experiment is blocked.

Persistent diagnostics are available in `cdp_temperature_curve`, including
sufficient and insufficient Demand, confidence, sample count and timestamp.

R7 lets heating efficiency learning use the same evidence hierarchy as CDP.
Physical climate mode must be `heat`; action `heat`/`heating` confirms running,
or the configured compressor/running sensor may confirm it when the action is
missing or nonstandard. Explicit compressor-off evidence still blocks
learning. Cooling and drying action requirements remain unchanged.

R6 originally confirmed released airflow in 20 seconds and probed Demand
downward every 60 seconds. R8 supersedes that persistent-learning cadence with
a three-minute sustaining observation. If airflow becomes restricted after a probe, the
last confirmed release Demand is restored immediately. The lowest confirmed
release Demand is persisted per unit and reused at the next heating start.

R5 retunes CDP release to the observed approximately one-minute physical
response. It stabilizes for 20 seconds, confirms suppression for 20 seconds,
then raises Demand by five points on every 20-second adaptive cycle until RPM
confirms that CDP has released or Demand reaches 60.

R4 also accepts the configured compressor/running sensor as positive heating
evidence whenever the physical climate mode is `heat`. This prevents an
incorrect or idle `hvac_action` attribute from permanently disabling CDP
release while the compressor is demonstrably active. CDP diagnostics now show
the reported HVAC mode, action and running evidence.

R3 fixes the CDP release governor so retained MQTT fan-RPM states are always
accepted when numeric, even when their value and timestamp have not changed.
It also accepts both `heating` and `heat` as the reported heating action while
the climate entity itself is in physical `heat` mode.

When Summer Mode is enabled, active heating normally remains at Demand `30`.
An adaptive Cold Draft Prevention release governor uses
`sensor.faikin_fanfreq` for Daikin1 and
`sensor.94a9903980ec_fanfreq` for Daikin2. If physical heating and
`mediumHigh` are confirmed but airflow remains suppressed after stabilization,
the affected unit rises in five-point Demand steps, up to `60`, until the fan
RPM confirms that Daikin's internal CDP restriction has released.

This release fixes an `auto + fan` desired-state lookup error which could
reapply `lowMedium` after heating had correctly requested `mediumHigh`.
Every actively heating unit now retains `mediumHigh`; idle ventilation,
cooling and drying retain `lowMedium`.

The governor waits 20 seconds after heating begins, confirms low airflow for
20 seconds, and permits at most one increase every 20 seconds. CDP is detected
below 75% of the calibrated RPM for the requested fan setting and considered
released above 85%. Release must remain confirmed for one minute. After ten stable minutes it tests one
five-point-lower Demand and restores escalation if airflow falls again. Missing
or invalid fan sensing, missing calibration, non-heating `hvac_action`,
mode/fan mismatches, defrost and post-defrost recovery all fail safely to Demand `30`. Turning Summer Mode off
restores the existing unrestricted heating Demand logic.

RPM calibration is explicit. Select shared mode `off + FAN`, then call
`pyscript.daikin_dual_calibrate_fan_rpm`. The controller pauses normal output,
places each unit in physical `fan_only`, and profiles every advertised fixed
fan setting. Each setting settles for 45 seconds and is sampled for 30 seconds;
the median RPM is stored. `auto` is excluded because it is not a fixed fan
speed. Calibration typically takes about 12–15 minutes for two five-speed
units. Both units return to `fan_only` + `lowMedium` afterward. Profiles persist
across restarts and are published in
`sensor.daikin_dual_fan_rpm_calibration`.

Daikin2 Demand is now fixed at `100` whenever Daikin2 is actively providing
physical cooling or logical drying. This fixed service value bypasses PI/ML
allocation, outdoor Demand bounds, step limiting, post-defrost holds and the
normal Demand-change interval. Idle ventilation retains its existing behavior.

The Home Assistant climate state `fan_only` is the successful physical
acknowledgement of shared `off + FAN`. The logical selector name `off_fan` is
never expected from the climate entity.

This release replaces the layered legacy/per-unit runtime with one shared-zone
controller. It retains mode-aware safety, fixed targets, physical cooling,
coordinated humidity drying/reheat, COP learning and predictive landing, while
making every physical command part of one generation-based desired state.

The canonical indoor sensor remains
`sensor.apollo_temp_1_95d874_temperature_probe`. The confirmed external sensor
remains `sensor.iv_tulo_lampotila`, despite its supply-air-like entity name.

- Daikin2 is the only unit permitted to run either logical cooling or humidity
  drying;
- logical cooling uses Daikin2 alone in physical `cool` at 16 C with fan
  `lowMedium`;
- humidity drying uses Daikin2 in native physical `dry` at 16 C with the exact
  `lowMedium` fan option (level 2/5);
- Daikin1 is hard-blocked from `cool` and `dry`; it remains available only as
  physical `heat`/reheat when summer-mode temperature support is required;
- the assist predicts the five-minute temperature trajectory and adds only the
  reheat needed to hold the effective indoor setpoint;
- if temperature still reaches 0.30 C below setpoint, lead drying pauses while
  assist heating continues, then resumes after temperature recovery and the
  lead minimum-off interval;
- if Daikin2 is the only configured unit it can still cool or dry without
  reheat; a Daikin1-only configuration cannot start either summer mode and
  reports `cooling_unit_unavailable` or `drying_unit_unavailable`;
- ordinary shared PI control remains on a true five-minute cadence;
- the reduction-only landing governor normally acts once per minute;
- a separate short-window temperature rate enables 20-second checks when the
  room is rapidly approaching the setpoint;
- rapid reductions are limited to five Demand points per active unit per
  action and still obey each unit's configured step limit;
- the fast path never increases Demand and does not feed ML learning;
- Daikin1 uses `sensor.daikin_cop`;
- Daikin1 uses `sensor.daikin1_defrost` as its authoritative explicit defrost
  state, with fresh `sensor.faikin_liquid` retained as a fallback for invalid
  or unavailable explicit states;
- Daikin2 uses `sensor.p40n_power` and `sensor.daikin2_cop`;
- Daikin2 uses `sensor.daikin2_defrost` as its explicit defrost state;
- one-unit and dual-unit COP is compared at matched outdoor and Demand loads;
- dual COP is weighted by each unit's measured electrical power;
- valid COP becomes the primary heat-mode optimizer metric, while the existing
  electrical-energy score remains the automatic fallback;
- every logical cooling request commands Daikin2 physical `cool`, target
  16 C and fan `lowMedium`;
- shared `off + FAN` continuously commands both units to physical `fan_only`
  with fan `lowMedium` and no temperature, Demand or Quiet command;
- shared `auto + fan` uses the controller's normal automatic heat/cool/dry/off
  selection, uses `mediumHigh` on each actively heating unit, and continuously commands
  both units to physical `fan_only` while conditioning is idle; it never
  commands physical `auto` or `heat_cool`;
- after a summer-mode transition, physical `cool` must remain stable for 20
  seconds before the controller sends 16 C;
- the first reported 16 C is provisional for another 20 seconds, so a delayed
  18 C device report revokes the acknowledgement and immediately reissues 16 C;
- logical cooling retains its temperature thermostat, 16 C device target,
  cooling Demand limits and learning context, but has no second-unit cooling
  staging;
- an independent, alert-only Home Assistant heartbeat watchdog;
- sensor-age validation using `last_reported` / `last_updated`;
- one serialized ten-second housekeeping loop for report tracking, safety,
  defrost, scheduled control and command reconciliation;
- finite indoor temperatures remain usable for comfort control even when their
  report timestamp is old;
- indoor report staleness is diagnostic-only and suspends learning/optimization
  until a new report arrives;
- anomaly-tolerant continuity: indoor/climate anomalies freeze commands while
  preserving current HVAC modes and Demand;
- no sensor, climate, heartbeat or command anomaly requests whole-system off;
- humidity, liquid, outdoor and terminal command failures degrade only their dependent
  control or diagnostic feature;
- an immediate, always-step-limited startup Demand;
- adaptive 60/20-second, rate-predictive downward Demand tapering;
- adaptive 20-second Summer heating CDP monitoring and minimum-output release;
- integral anti-windup and a hard heating comfort stop;
- active 5-point minimum-sustain tests;
- generation-based mode, target, fan, Demand, Quiet and off acknowledgement with
  fail-tolerant reconciliation, exponential backoff and stale-intent cancellation;
- Demand, Quiet and fan acknowledgement before an off unit is started;
- restart-safe per-unit compressor timers, desired state, defrost detector,
  post-defrost hold and coil-dry deadlines;
- health, command, daily-statistics and commissioning diagnostics;
- a full-controller no-output shadow mode; and
- a command-free replay service for the in-memory last 24 hours or supplied
  Home Assistant history JSON.

This Home Assistant Pyscript package controls the two configured Daikin heat
pumps in the same area. One shared supervisory thermostat uses the effective
setpoint to start and stop the system. A common demand controller allocates
output between the configured units, and a slow mode-separated optimizer
compares safe unit combinations when the required power and COP measurements
are available.

For heating, the optimizer evaluates only three allocations: balanced,
Daikin1 only and Daikin2 only. Cooling and drying bypass those choices and use
the fixed Daikin2-only allocation.

The old single-unit RLS controller, per-unit theta/P stores, weather-forecast
path, derivative-sensor dependency and independent state/time triggers have
been removed. One configured unit still works through the shared allocator;
there is no separate fallback controller.

Normal operation uses `auto`: the controller chooses heating,
temperature-driven cooling, humidity-driven drying or idle from the shared
temperature and humidity readings. Manual heat, cool, dry and off selections
remain available for commissioning. Two continuous manual shared modes are
also available: `off + FAN` runs both units in `fan_only`. `auto + fan` uses
the same thermostat and allocation optimizer as `auto`. Each actively heating
unit uses `mediumHigh`; idle ventilation, cooling and drying use `lowMedium`.
While no heat, cooling or drying cycle is active,
`auto + fan` explicitly runs both units in `fan_only`; accepted conditioning
demand interrupts idle ventilation immediately. Idle ventilation continues
while a compressor start is waiting for its minimum-off timer.
`cool` is a logical temperature-control
mode; its physical Daikin command is `cool`, 16 C and fan `lowMedium`.
Humidity-driven `dry` is a separate role-fixed path: Daikin2 uses physical
`cool`, 16 C and the exact `lowMedium` fan option (level 2/5), while Daikin1 can
provide reheat. In both summer modes, the refrigeration allocation contains
only Daikin2. After every logical cool or dry cycle, the previously active
refrigeration unit runs `fan_only`
for 15 minutes to dry the indoor coil. A new conditioning demand interrupts
that fan cycle immediately; otherwise the climate entity is turned off when
the timer expires.

Cooling and drying are available only while `input_boolean.daikin_summer_mode`
is on. Dry mode uses
`sensor.apollo_temp_1_95d874_board_humidity`. With two suitable units it keeps
drying through the effective setpoint by adding controlled sensible reheat.
Daikin1 is never assigned a refrigeration share in either summer mode.

## Files

- `daikin_ml_multi.py` -> `/config/pyscript/daikin_ml_multi.py`
- `daikin_ml_multi_config.py` -> `/config/pyscript/daikin_ml_multi_config.py`
- `daikin_dual_helpers.yaml` -> `/config/packages/daikin_dual_helpers.yaml`
- `daikin_watchdog_package.yaml` -> `/config/packages/daikin_watchdog_package.yaml`
- `daikin_diagnostics_dashboard.yaml` -> paste into a manual Lovelace card
- `validation/test_safety_commissioning.py` -> offline release validation only;
  do not copy to Home Assistant

Pyscript must already be installed. The existing Pyscript configuration must
allow the local `daikin_ml_multi_config.py` import.

This compatibility release avoids Python generator-expression AST nodes because
some deployed Pyscript interpreters raise `NotImplementedError: ast_generatorexp`.
Equivalent explicit loops and list comprehensions are used throughout.

This release also prevents off-state learning. Retained Demand-select values
are never treated as delivered output while the compressor is idle, off or in
`fan_only`.

## Safety architecture

The Pyscript controller updates
`sensor.daikin_dual_controller_heartbeat` after each successful controller
pass. `daikin_watchdog_package.yaml` checks that sensor from an ordinary Home
Assistant automation. Because the automation is not implemented in Pyscript,
it can still report a stale heartbeat when this Python file fails to load or
stops. The independent anomaly monitor also reacts to
`sensor.daikin_dual_health = fault` or
`sensor.daikin_dual_command_status = failed`, but never changes either climate.

Sensor freshness does not depend on the numeric value changing. Home Assistant
advances `last_reported` whenever an integration writes an entity, including a
write such as `22.13 C -> 22.13 C`. The serialized ten-second housekeeping
loop watches the configured indoor, humidity, outdoor, climate and
defrost/liquid entities. It:

- re-evaluates safety and freshness;
- advances critical recovery validation;
- refreshes `sensor.daikin_dual_health`;
- records `sensor_report_sequence`, `last_sensor_report_entities` and
  `last_sensor_report_at`;
- updates a single per-unit defrost detector only on distinct report tokens;
- runs 20-, 60- and 300-second work when due; and
- reconciles the newest complete desired generation before retrying commands.

There is no separate sensor-report, command-monitor, adaptive-control or
minute trigger. This removes concurrent mutation of controller globals.

A sensor report does not advance the 20-, 60- or 300-second schedules; it is
coalesced into the next ten-second snapshot. Each entry in
`sensor_status` exposes `timestamp_source`; the Apollo indoor sensor should
report `last_reported`. The controller falls back to `last_updated` for older
Pyscript installations, but that fallback cannot detect an entirely unchanged
state write.

The watchdog is deliberately disabled after first installation. Daikin1 is
already configured as `climate.faikin_mqtt_hvac` in its climate group. Restart
Home Assistant, run `pyscript.daikin_dual_validate_configuration`, test the
group, and only then enable `input_boolean.daikin_watchdog_enabled`.

Normal fixed device targets remain 31 C for heat and 16 C for logical cooling
and humidity drying. Daikin2 uses the advertised `lowMedium` option (level 2/5) for
both summer modes. Shared idle ventilation uses `lowMedium`; active heating
uses `mediumHigh`. Standby and timed post-cycle coil-dry resolve and restore
the entity's advertised automatic option (`Auto` here).
Logical cooling physically operates the Daikin in `cool` with fan `lowMedium`; it
remains temperature-controlled and never requires humidity.

These critical anomalies freeze new controller commands without switching
equipment off:

- every configured shared indoor sensor is missing, `unknown`, `unavailable`,
  nonnumeric or otherwise nonfinite;
- a configured climate entity is missing or unavailable.

A finite indoor value remains usable for temperature control even after
`DUAL_INDOOR_MAX_AGE_S`. Its `sensor_status` entry still shows `stale: true`,
but also shows `usable_for_control: true` and
`stale_is_warning_only: true`. Health remains `ok`, commands are not frozen,
and `diagnostic_warnings` contains
`indoor_sensor_report_stale_value_usable`. Learning and optimizer episodes are
suspended until the indoor entity reports again, preventing an old timestamp
from contributing new model evidence.

Any already-pending physical command is cancelled when the hold begins so an
old intent cannot execute after recovery. The current HVAC mode, device target,
Demand select and Quiet state are left unchanged. If the plant was inactive,
the anomaly does not start it. An explicit `off` selection remains
authoritative, as does disabling summer mode for an active cool/dry cycle;
these are operator/system stop decisions, not anomaly-induced shutdowns.

After a critical input recovers, commands remain frozen for approximately 60
seconds and at least four consecutive valid safety evaluations. Health
diagnostics retain the initial anomaly in `original_trigger` throughout this
interval and expose `recovery_remaining_seconds`. `last_critical_trigger`
remains available after recovery. `shutdown_required` remains `false`.

All other input failures are feature degradations:

- Humidity is mandatory only for logical humidity-driven `dry`. Invalid
  humidity prevents a new cycle from starting; an active dry/reheat cycle holds
  its current physical state until humidity is fresh. Auto heating and logical
  temperature cooling continue normally.
- A recognized explicit defrost state remains authoritative regardless of its
  report age. An old unchanged `True` or `False` is diagnostic only and never
  disables defrost detection.
- Missing, `unknown`, `unavailable` or unrecognized explicit defrost states
  fall back to liquid-temperature detection. Only stale/invalid liquid data
  then disables controller-side defrost detection for that unit. It never
  turns the climate entity off.
- Stale outdoor temperature suspends sustain/COP/allocation optimization.
  Comfort control continues using the colder of the last-good outdoor reading
  and the configured `-5 C` fallback, producing conservative cold-weather
  Demand bounds.
- A valid mode, target, Demand-select, Quiet or off mismatch remains
  recoverable regardless of how many acknowledgement windows it needs. Only an
  explicitly unsupported or invalid expectation becomes a unit-specific
  terminal diagnostic. Neither condition stops the other unit or the whole
  plant.
- A stale power or COP sample rejects only the affected efficiency sample. An
  intentionally off unit's stale 0 W reading does not invalidate a one-unit
  allocation episode.

Feature degradation may set health to `degraded`, but
`degraded_mode_active=false` and `shutdown_required=false`; HVAC remains under
normal thermostat control. Critical anomalies set `control_hold_active=true`
and `commands_frozen=true`, while still keeping `shutdown_required=false`.
The independent watchdog reports `fault`, failed command status or a stale
heartbeat and is always alert-only.

## Required entity configuration

Both units are active in `DAIKINS` in `daikin_ml_multi_config.py`.

- `daikin2` is the lead and `DUAL_PRIMARY_UNIT`.
- `daikin1` is the assist.
- Both units use `sensor.apollo_temp_1_95d874_temperature_probe`.
- Both units use `sensor.apollo_temp_1_derivative`.
- Both units use `sensor.iv_tulo_lampotila`.

The canonical Daikin1 runtime and learning mappings are:

```python
"POWER_SENSOR": "sensor.daikin_p40_power",
"COP_SENSOR": "sensor.daikin_cop",
"CLIMATE": "climate.faikin_mqtt_hvac",
"HEATING_ENTITY": "sensor.faikin_comp",
"HEATING_THRESHOLD": 0.0,
"DEFROST_ENTITY": "sensor.daikin1_defrost",
```

The power sensor must report instantaneous watts. Learning additionally
requires `climate.faikin_mqtt_hvac` to expose an active `hvac_action`.
`sensor.faikin_comp` remains a running/defrost aid and does not substitute for
`hvac_action` in learning. A recognized state from `sensor.daikin1_defrost` is
authoritative. If that state is invalid or unavailable, the controller falls
back to fresh liquid-temperature and running evidence.

The configured runtime values are:

| Unit | Role | Climate | Demand | Explicit defrost | Liquid fallback | Compressor | Quiet |
| --- | --- | --- | --- | --- | --- | --- | --- |
| daikin1 | assist | `climate.faikin_mqtt_hvac` | `select.faikin_demand_control` | `sensor.daikin1_defrost` | `sensor.faikin_liquid` | `sensor.faikin_comp` | `switch.faikin_quiet_outdoor` |
| daikin2 | lead | `climate.94a9903980ec_mqtt_hvac` | `select.94a9903980ec_demand_control` | `sensor.daikin2_defrost` | `sensor.94a9903980ec_liquid` | `sensor.94a9903980ec_comp` | `switch.94a9903980ec_quiet_outdoor` |

Daikin1 power and COP learning use `sensor.daikin_p40_power` and
`sensor.daikin_cop`. Daikin2 power and COP learning use
`sensor.p40n_power` and `sensor.daikin2_cop`.

Every configured unit must have:

```python
"ALLOW_HVAC_OFF": True,
"ALLOW_HVAC_MODE_CONTROL": True,
```

Each climate entity must expose a `fan_modes` attribute containing an automatic
option. In this installation, Daikin2 advertises the exact options `Auto`,
`night`, `Low`, `lowMedium`, `Medium`, `mediumHigh`, and `High`. Both logical
cooling and humidity drying use `lowMedium`, which is fan level 2/5. The resolver
always sends the exact spelling advertised by Home Assistant and never sends
an unresolved raw configuration value. If level 2/5 is unavailable, summer
compressor operation is blocked and configuration validation reports the
missing fan mode instead of repeatedly calling `climate.set_fan_mode` with an
invalid value.

Coil drying is enabled by default when the climate entity advertises
`fan_only` in its `hvac_modes` attribute. It can be disabled for a particular
unit with:

```python
"ALLOW_FAN_ONLY_COIL_DRY": False,
```

The controller deliberately blocks shared mode control and publishes
`missing_climate_mode_control` until both configured units have a climate
entity and mode control is enabled. This prevents Demand commands being sent
to an HVAC which the script cannot place in the correct mode or standby.

For learning, each climate entity must also expose its current `hvac_action`.
The optional `HEATING_ENTITY` helps defrost detection but is intentionally not
accepted as a substitute for `hvac_action` in sustain or efficiency learning.

## Active dual-unit operation

The supplied configuration is ready for the two named units. Do not comment
out Daikin2 unless that device is intentionally being removed from controller
ownership. With both dictionaries active:

- normal heating uses Daikin2 as lead by default and may stage Daikin1 as
  heating assist;
- `input_boolean.daikin1_heating_lead` changes only the first heating unit:
  off starts Daikin2 first, while on starts Daikin1 first and uses Daikin1 as
  the sole mild-weather heater above the assistance threshold;
- logical cooling uses only Daikin2; Daikin1 never stages as a second cooling
  unit;
- humidity-driven drying uses only Daikin2 in physical `dry`, at 16 C and the
  exact `lowMedium` fan option (level 2/5);
- temperature support during drying uses Daikin1 in `heat`;
- Daikin1 is hard-blocked from every logical cooling and drying command at the
  allocator, command, persisted-state and supervisor layers;
- Daikin2 alone receives the post-cycle `fan_only` command after a coordinated
  drying cycle;
- the independent watchdog covers both climate entities.

Use shared room sensors outside both discharge-air streams. Verify that
`OUTDOOR` measures actual outdoor air near the heat pumps. A ventilation supply
temperature altered by heat recovery is not suitable for the 3 C heating
assistance threshold or learned outdoor contexts. In this installation,
`sensor.iv_tulo_lampotila` is a physically confirmed external probe and is
listed under `DUAL_CONFIRMED_OUTDOOR_SENSORS`, so its misleading entity name
does not produce a commissioning warning. Its normal six-minute reporting
interval remains within `DUAL_OUTDOOR_MAX_AGE_S = 15 minutes`.

## Mode and summer controls

Select the persistent desired mode with:

```text
input_select.daikin_dual_hvac_mode
```

Options are `off`, `auto`, `heat`, `cool`, `dry`, `off + FAN` and
`auto + fan`. Use `auto` for normal
operation. Manual modes bypass automatic selection but retain their own
thermostat, summer, runtime and safety rules. Selecting `cool` always sends
physical `cool` mode, target 16 C and fan `lowMedium` to Daikin2 while continuing
to control by room temperature. Selecting `dry` remains humidity-driven and uses the fixed
Daikin2-cool/Daikin1-reheat roles described above.

`off + FAN` is continuous ventilation, distinct from the timed post-cycle coil
dry: both units remain in physical `fan_only` at `lowMedium` until another
shared mode is selected. `auto + fan` uses the controller's automatic choice
of heat, logical cool, logical dry or off. It retains the normal temperature,
Demand, Quiet, allocation and learning rules. Idle/cool/dry uses `lowMedium`,
while each actively heating unit uses `mediumHigh`. It never requests physical
`auto` or `heat_cool`.
Configuration validation reports an error if either unit lacks `fan_only` or
a fan option equivalent to `lowMedium`.

`input_boolean.daikin_summer_mode` is authoritative for cooling and drying:

- Summer off + cool/dry request: effective mode is `off`; both units are kept
  off and the status reports `summer_mode_off`.
- Summer on + cool/dry request: the selected mode may run when demand exists.
- Automatic and manually selected heating remain available regardless of
  summer mode.
- Turning summer mode off during cooling or drying stops refrigeration
  immediately; the safe fan-only coil-dry cycle may continue.
- In `auto`, summer off permits only heating or idle. Summer on permits all
  modes.

The helper represents the user's declaration that the cooling season is open.
If desired, a separate Home Assistant automation can turn it on and off by date.
The Python controller still treats the helper as the final safety gate.

## Automatic mode selection

`auto` uses a wider neutral band than manual mode thermostats so the system
does not bounce between heating and cooling. With the default 22.5 C setpoint:

| Automatic condition | Selected mode |
| --- | --- |
| Indoor <= 22.1 C (`SP - 0.40`) | Heat |
| Summer on and indoor >= 22.9 C (`SP + 0.40`) | Cool |
| Summer on, RH >= target + configured delta, and indoor >= `SP + 0.20` | Dry |
| None of the above | Idle/off |

Priority is heat, then logical cool, then humidity-driven dry. Logical cooling
therefore wins when the room is both clearly hot and humid, but its physical
Daikin mode is `cool`. Humidity-driven dry is used when humidity is high
but cooling is not yet needed. If the room becomes at least 0.60 C too warm
during humidity-driven drying, the controller confirms and changes to logical
cooling.

A new automatic candidate must remain true for three minutes before it is
selected. Severe cold at least 1.0 C below setpoint bypasses confirmation and
requests heat immediately, while still respecting compressor minimum-off time.
An active mode remains latched until its own stopping condition is reached.
Every compressor mode change passes through off; heat never changes directly
to cool/dry or vice versa.

Automatic selection uses the neutral base setpoint. After a mode is selected,
that mode's effective price-biased setpoint is used for actual control. This
prevents a heating-oriented price bias from distorting which mode is selected.

Adjust automatic behavior with:

- `input_number.daikin_auto_heat_on_delta`
- `input_number.daikin_auto_cool_on_delta`
- `input_number.daikin_auto_mode_confirmation_minutes`

## Effective-setpoint thermostat

All configured units are supervised as one plant. With one unit this is a
single-unit plant; after the second is configured they do not independently
decide when the shared zone needs conditioning.

When Daikin2 is the only unit allocated heating output, Daikin1 remains active
in physical `fan_only` with fan `lowMedium`. It receives no heating target,
Demand or Quiet command. If Daikin1 begins heating through a balanced or
Daikin1-only allocation, it immediately returns to physical `heat` with the
normal `mediumHigh` heating fan. Daikin1's circulation-fan power is included in
Daikin2-only energy and combined-COP scoring.

The heating-lead toggle does not change cooling, native drying, coordinated
reheat, CDP, assistance-temperature hysteresis, Demand limits, fan settings or
optimizer policy definitions. Below the assistance threshold, the optimizer
may still select either single unit or the balanced allocation as before. A
lead change during active heating is applied on the next normal control pass.

Default start/stop rules are:

| Mode | Start | Stop |
| --- | --- | --- |
| Heat | indoor <= setpoint - 0.15 C | indoor >= setpoint + 0.10 C |
| Cool | indoor >= setpoint + 0.20 C | indoor <= setpoint - 0.10 C |
| Dry | RH >= target + 3 points and indoor >= setpoint + 0.20 C | RH <= target or indoor <= setpoint |

The defaults are adjustable with the supplied helpers. Minimum plant runtime is
20 minutes and minimum off time is 10 minutes. Normal stops caused by summer
mode, explicit mode change or the dry-mode temperature floor do not wait for
minimum runtime. Invalid sensors freeze commands and do not stop the plant.

The effective setpoint remains the shared room-comfort target used to start,
stop and modulate the system. It is not written to the Daikin climate entity.
While active, every participating unit receives a fixed device target:

- Heat: 31 C
- Logical cool on Daikin2, physically running as `cool`: 16 C and fan 2/5
- Humidity dry on Daikin2, physically running as `dry`: 16 C and fan 2/5

These values keep the Daikin's internal thermostat from prematurely limiting
the shared demand controller. They are configured with
`DUAL_HEAT_HVAC_TARGET_C`, `DUAL_COOL_HVAC_TARGET_C` and
`DUAL_DRY_HVAC_TARGET_C`. Post-cycle `fan_only` receives no temperature
command and restores fan `auto`. The
Mode-status and Demand diagnostics expose the logical mode and its corresponding
physical `cool` or `dry` mode.

The fixed target is checked on every ten-second reconciliation pass and
reasserted by the supervisor every minute. A manual, device-originated or external change from
31 C while heating, or from 16 C while cooling/drying, therefore creates a
tracked target command. Summer target commands receive six quick attempts per
reconciliation cycle. A valid mismatch never becomes terminal merely because
that window expires: the command changes to `recovering` and continues after
30, 60 and 120 seconds, with subsequent intervals capped at five minutes. The
same behavior applies to Demand, HVAC mode, fan and Quiet expectations.

If a mode transition temporarily restores a remembered 18 C target, the
still-valid 16 C intent therefore remains active until it has been observed
continuously for 20 seconds or is superseded by a newer desired generation.
The controller never accepts 18 C as
the desired summer target. Persisted cool/dry state is canonicalized to 16 C
and `lowMedium` before startup reconciliation, so a stale stored 18 C target is
never replayed after restart. During an anomaly, reassertion and retries are
frozen and the last physical target is preserved; the controller never
substitutes the shared comfort setpoint.

When humidity drying starts, the command sequence is intentionally ordered:
fan `lowMedium` is acknowledged first, physical `dry` is acknowledged second,
`dry` must remain visible for 20 seconds, and only then is 16 C written. The 16 C command remains in
`verifying` for another 20 seconds. If the device publishes a delayed 18 C
remembered target during that window, the controller revokes the provisional
acknowledgement and immediately writes 16 C again without creating a new
desired generation or a failure.

Commissioning validation also compares the required 16 C target with the
climate entity's advertised `min_temp`. If `min_temp` is above 16 C, validation
fails with an explicit `climate_min_temp_..._above_required_summer_target_16.0`
issue; that means the Home Assistant climate entity cannot accept the required
target until its integration/device range is corrected.

## Fail-tolerant command reconciliation

The command tracker distinguishes a delayed acknowledgement from a real
invalid command:

- `pending`: the first quick acknowledgement cycle is active;
- `verifying`: the summer target currently reports 16 C but has not yet
  completed its 20-second stable acknowledgement window;
- `recovering`: Home Assistant reports another valid value, or a transient
  service error occurred; reconciliation continues with exponential backoff;
- `waiting_entity`: the entity is `unknown` or `unavailable`; no attempt is
  consumed and no command is sent until the entity returns;
- `acknowledged`: the actual state matches the newest desired generation;
- `failed`: Home Assistant's advertised `options`, `fan_modes`, `hvac_modes`,
  `min_temp`/`max_temp`, or an explicit service validation error proves the
  requested value invalid.

For example, Demand `expected: 95` with `actual: 100` remains under
`pending`/`recovering`; it is not placed in `failures` and does not degrade
controller health. A new desired generation immediately supersedes those
retries, preventing an old 95 percent command from executing after the
controller has moved to another Demand or mode.

Existing Nordpool setpoint bias remains supported. Its direction is inverted in
cool/dry mode: a signal which preheats by raising the heating setpoint precools
by lowering the cooling setpoint.

## Predictive heating Demand landing

The ordinary shared PI calculation still runs every five minutes. A separate
reduction-only governor normally acts every 60 seconds and predicts the
temperature error ten minutes ahead:

```text
projected error =
    current error - max(temperature rise, 0) * 10 / 60
```

In parallel, a dedicated three-minute history estimates the recent approach
rate from fresh indoor samples. Once at least four samples spanning 60 seconds
show a rise of at least 0.30 C/h and the rate projection has entered the
landing band, the governor switches to a 20-second cadence. The conservative
maximum of the long learning slope and the short landing slope is used, so a
new acceleration is not hidden by the 15-minute learning window.

Tapering begins when the projected error enters 0.40 C. The landing ceiling
falls progressively from the active action's normal maximum to its
outdoor-dependent minimum Demand. The governor can only reduce Demand. The
combined-total and per-unit step limits remain mandatory; rapid actions use
the lower of the configured total step and five points per active unit.

While landing is active:

- positive integral stops accumulating and decays by 50 percent per full
  control pass;
- positive integral is cleared when the projection reaches the setpoint;
- Demand reductions can occur every minute, or every 20 seconds during a
  confirmed rapid approach, instead of waiting five minutes;
- only rapid downward changes may shorten the normal per-unit Demand-change
  interval; upward changes always retain the configured interval;
- the normal 20-minute minimum-on timer keeps the compressor on only at tapered
  Demand; and
- `SP + 0.25 C` is a hard comfort stop which overrides minimum-on time.

The sustaining learner now actively tests a Demand five points below its
current heating estimate, or validates an already lower level reached by the
landing governor without raising output. A test starts only after at least 12
minutes of confirmed compressor operation and while temperature is steady near setpoint.
The candidate must first be reached, then remain stable for 15 minutes. It is
accepted only if temperature does not fall away; otherwise the previous
sustain level is restored and probing pauses for 30 minutes.

Key defaults in `daikin_ml_multi_config.py` are:

```python
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
DUAL_HEAT_SUSTAIN_PROBE_STEP = 5.0
DUAL_HEAT_SUSTAIN_PROBE_HOLD_MINUTES = 15.0
```

## Fan-only coil drying

When a cool or dry compressor cycle ends, only the units that were conditioning
the room are switched to `fan_only`. The default duration is 15 minutes and is
shown in `sensor.daikin_dual_mode_status` as:

```text
state: fan_only_coil_dry
fan_only_active: true
coil_dry_source_mode: cool or dry
coil_dry_units: [daikin2]
coil_dry_remaining_seconds: ...
```

If heat, cool or dry becomes genuinely needed while fan-only is running, the
fan cycle stops immediately. The compressor still observes the existing
minimum-off timer. Time spent in fan-only counts toward that off timer because
the compressor is already stopped. If no new demand occurs, the climate is
turned off after 15 minutes.

If `fan_only` is not listed in a unit's `hvac_modes`, coil drying is skipped for
that unit and it is turned off. The duration helper is
`input_number.daikin_coil_dry_minutes`.

## Heating and defrost

The second heating unit becomes eligible below the outdoor threshold, default
3.0 C, and is removed above 4.0 C. Above that band the shared controller uses
the lead unit only.

Liquid-temperature defrost detection is valid only when:

1. The requested and effective system mode is `heat`.
2. The unit is in heat or an unknown-compatible mode, never cool/dry/off.
3. The unit was producing heat when the low-liquid sequence began.
4. Liquid temperature remains below the threshold for the configured debounce.

An optional explicit `DEFROST_ENTITY` per unit takes precedence. Recognized
states are `on`, `true`, `yes`, `defrost`, `defrosting`, `active`, `off`,
`false`, `no`, `idle` and `inactive` (case-insensitive). These are state-based,
so no freshness heartbeat is required. Invalid explicit states fall back to the
fresh liquid-temperature detector. With one unit, its own demand and Quiet
state are frozen during validated defrost. With two or more configured units,
the shared freeze starts only when at least two units were heating and one
enters validated defrost; both demand-select and Quiet states are captured. PI,
learning and demand changes pause. The safety timeout is 30 minutes.

Changing away from heat immediately clears defrost, post-defrost hold and
cooldown state. Low liquid temperature in cool or dry therefore cannot trigger
or prolong a defrost freeze.

## Cooling allocation

Cooling never uses the heating 3 C outdoor threshold.

With both units configured, Daikin2 is the only cooling allocation. Daikin1 is
never made eligible for refrigeration by error size, runtime, optimizer choice,
fallback or restart recovery. Daikin1 may only remain off or provide physical
heat/reheat when summer-mode temperature support requires it. Cooling retains
its own sustaining context and Demand controller, separate from heating and
humidity-driven drying.

## Drying staging

The configured humidity sensor is filtered over ten minutes. Invalid or
unavailable humidity prevents humidity-driven dry mode from starting and stops
an active dry cycle safely, but has no effect on heat or temperature-driven
logical cooling.

Defaults:

- Humidity target: 55 percent
- Start threshold: 58 percent
- Daikin2-without-reheat temperature start margin: 0.20 C above effective setpoint
- Daikin2-without-reheat temperature stop floor: effective setpoint
- Coordinated reheat target: effective setpoint + 0.10 C
- Reheat start: measured or projected temperature at setpoint + 0.10 C
- Reheat stop: temperature at setpoint + 0.30 C and no longer falling
- Reheat Demand range: 30–70 percent
- Reheat minimum on/off: 10/5 minutes
- Lead drying pause: setpoint - 0.30 C
- Lead drying resume: setpoint - 0.05 C after five minutes off

When a distinct lead and assist unit are configured, humidity-driven drying is
a coordinated mixed-mode service:

1. Daikin2 alone runs physical `dry` at 16 C and `lowMedium` fan (level 2/5); its Demand
   follows humidity error and humidity response and remains capped at 70 percent.
2. Daikin1 initially remains off with fan `auto` while the room is warm.
3. Every 20 seconds the controller evaluates the Apollo indoor probe and both
   the short- and long-window temperature rates.
4. Daikin1 enters physical `heat`, fan `auto`, with the fixed 31 C device target when
   measured or projected temperature reaches the reheat-on threshold.
5. Its independent temperature controller adjusts heating Demand while all
   ordinary per-unit Demand step limits remain enforced.
6. If reheat cannot arrest the fall, the lead's drying compressor is paused at
   the hard floor; assist heating remains available until temperature recovers.
7. If the heating assist defrosts, lead drying pauses immediately and the
   assist's pre-defrost Demand is held until heat production can recover.
8. When humidity reaches its target, Daikin1 is turned off and only Daikin2's
   previously active drying coil receives the normal `fan_only` cycle. Its fan
   selection is restored to `auto` before that transition.

Mixed physical `dry` + `heat` is deliberately excluded from allocation and COP ranking:
the units are providing different latent and sensible services, so their COP
values are not a fair same-service allocation comparison. The Demand diagnostic
reports `optimizer_mode: suspended_mixed_service` and
`learning_block_reason: coordinated_heat_dry_service` during this state.

## Efficiency learning

Learning contexts begin with the logical operating mode, so heat,
temperature-driven cool and humidity-driven dry samples can never contaminate
one another. Temperature cool uses physical `cool`; humidity dry uses physical
`dry`. Both use fan `lowMedium` only on Daikin2, and their logical control objectives remain separate.
Contexts additionally include outdoor and load buckets; humidity-driven dry
contexts include a humidity bucket.

Scores combine:

- inverse combined COP in heating when at least 80 percent of the episode has
  valid COP samples;
- otherwise electrical energy normalized by a mode-specific load measure;
- temperature comfort deviation;
- humidity deviation in dry mode;
- defrost cost in heat mode only; and
- compressor cycles.

Use real instantaneous power in watts. Configure either both unit
`POWER_SENSOR` values or one dedicated `DUAL_TOTAL_POWER_SENSOR`. Do not use a
whole-house meter. If a power sensor reports kW, set its scale to `1000.0`.

For direct COP learning, every running unit needs both its own
`POWER_SENSOR` and `COP_SENSOR`. The live combined value is:

```text
(COP1 × watts1 + COP2 × watts2) / (watts1 + watts2)
```

A simple arithmetic mean is not used. The controller integrates estimated
thermal output (`COP × electrical power`) and electrical input across the
episode, so the final episode COP is energy-weighted.

COP is accepted only in configured COP modes, currently `heat`, after the same
12-minute confirmed-runtime learning gate. Values outside 0.5–12.0, stale
values, readings below 150 W per running unit, startup, idle, fan-only,
defrost, and unexpected-unit operation are excluded. Excluding a COP sample
does not stop heating: the optimizer falls back to its measured
electrical-energy score.

In addition to long optimizer episodes, a matched operating map is persisted
by:

```text
logical mode | measured outdoor bucket | allocation | actual unit Demand levels
```

Demand is stored in real 5-point device levels. This keeps, for example,
`lead1_only` at 70 separate from `balanced` at 40/40. A context is shown as a
comparison candidate after ten valid minutes. The 300 most recently used
contexts are retained.

Optimizer modes:

- `disabled`: use the safe configured allocation.
- `shadow`: score the safe allocation and publish recommendations without
  experimenting.
- `auto`: evaluate eligible policies and use learned results.

Begin in `shadow`. A mode, setpoint, staging or manual-override change discards
the active episode rather than scoring incomparable data.

Learning is additionally gated by actual compressor operation:

- `heat` requires `hvac_action: heating`;
- logical `cool` requires physical climate mode `cool` and
  `hvac_action: cooling`;
- `dry` requires physical climate mode `dry` and accepts
  `hvac_action: cooling` or `drying`;
- `idle`, `off`, `fan_only`, defrost, an unexpected running unit, or a mode
  mismatch immediately clears the response window and discards the optimizer
  episode;
- a mode or allocation change starts a completely new response window; and
- sustain and optimizer learning begin only after 12 continuous minutes of
  confirmed actual operation and the normal slope/steady-state gates also pass.

The ordinary humidity history remains available for dry-mode selection while
the dry response learner uses a separate actual-running-only history. This
prevents passive temperature or humidity drift during an off period from being
attributed to a retained Demand value.

`sensor.daikin_dual_learning_status` is the authoritative live diagnostic. For
example, when the controller requests conditioning but the compressor is idle:

```text
state: blocked
actual_running: false
learning_allowed: false
learning_block_reason: hvac_not_running
active_learning_span_seconds: 0
```

Upgrading from `2026-07-30-dual-controller-simplified` preserves the shared
sustain, allocation-policy and COP map stored in
`pyscript.daikin_dual_zone_store`. No learning reset is required. Obsolete
per-unit RLS stores and learned-demand sensors are no longer read or written.
This release also persists each unit's learned released-airflow level in the
same store. The complete per-unit fan-setting RPM profiles are also persisted.

## Published entities

- `sensor.daikin_dual_mode_status`: requested, automatically selected and
  effective mode; selection/candidate reason and timer; compressor and
  fan-only state; coil-dry timer; minimum-off time; indoor temperature,
  setpoint and humidity
- `sensor.daikin_dual_total_demand`: mode, total and per-unit demand, staging,
  temperature rate, humidity and power; PI terms, raw/governed total,
  projected error/temperature, landing cap, actual select Demand, minimum-on
  remainder, sustain-probe state and per-unit `cdp_release` diagnostics
- `sensor.daikin_dual_fan_rpm_calibration`: calibration progress and the
  persisted per-unit RPM profile for every advertised fixed fan setting
- `sensor.daikin_dual_landing_status`: adaptive predictive landing state, ETA,
  cap, step-limited target, selected 60/20-second cadence, short/long rates,
  rate source and active sustain test
- `sensor.daikin_dual_efficiency_optimizer`: mode-specific context,
  recommendation, episode and action scores, COP coverage and COP-based score
- `sensor.daikin_dual_cop_efficiency`: live power-weighted combined COP,
  per-unit COP/power/estimated thermal output, actual Demand levels, current
  matched-context statistics and the best comparable load/allocation contexts
- `sensor.daikin_dual_learning_status`: actual-running evidence, continuous
  learning span, blocking reason, expected/confirmed units and per-unit
  `hvac_action`
- `sensor.daikin_dual_controller_heartbeat`: last successful pass as epoch,
  sequence number, execution time and controller version
- `sensor.daikin_dual_health`: `ok`, feature-only `degraded`, command-hold
  `degraded`, or `fault`; `control_hold_active`, `commands_frozen`, the invariant
  `shutdown_required=false`, feature degradations, sensor ages, original/last
  critical trigger, `recovery_remaining_seconds`, consecutive recovery readings,
  report-monitor sequence/latest entities, outdoor fallback, optimizer
  suspension, command failures, timer remainders and next control time
- `sensor.daikin_dual_command_status`: last requested/acknowledged value,
  desired generation, pending/recovering commands, retry cycle/backoff,
  superseded intents, terminal failures and command age
- `sensor.daikin_dual_daily_stats`: per-unit runtime and starts, measured
  energy, defrost minutes, comfort-deviation minutes and degree-minutes
- `sensor.daikin_dual_commissioning`: entity/capability validation result with
  exact issues and warnings
- `sensor.daikin_dual_history_replay`: command-free replay result and tested
  thresholds
- `pyscript.daikin_dual_zone_store`: persisted mode-separated sustaining and
  policy statistics plus the matched COP-by-load map
- `pyscript.daikin_dual_runtime_store`: restart-safe per-unit compressor
  timers, authoritative desired state, defrost state, post-defrost holds,
  mode, coil-dry and daily-statistics state
- `pyscript.daikin_system_defrost_freeze`: heat-only defrost freeze state

## Services

- `pyscript.daikin_ml_step`: request an immediate controller pass
- `pyscript.daikin_ml_persist`: persist the shared optimizer and runtime stores
- `pyscript.daikin_dual_optimizer_reset`: clear allocation-policy scores
  and the matched COP map
- `pyscript.daikin_dual_cop_reset`: clear only the matched COP-by-load map
  while retaining sustain and optimizer episode scores
- `pyscript.daikin_ml_reset`: clear shared sustain, policy and COP learning
  while preserving physical plant state and compressor timers
- `pyscript.daikin_dual_validate_configuration`: validate all configured
  entities, HVAC modes, Demand options, helper presence, watchdog group and
  power units
- `pyscript.daikin_dual_apply_recommended_defaults`: explicitly apply the
  documented defaults once; this is never run automatically
- `pyscript.daikin_dual_replay_history`: replay the internally collected last
  24 hours since the latest Pyscript reload, or a supplied JSON sample list,
  without sending any HVAC commands. The replay ring itself is not persisted,
  avoiding large recorder writes; safety timers and daily totals are persisted.

`input_boolean.daikin_controller_shadow_mode` suppresses every physical mode,
target, Demand, Quiet and off service call. The controller still calculates
and publishes `would_send` commands. Learning is blocked because no calculated
output may be treated as actual output.

## Helper state restoration

The supplied YAML intentionally contains no `initial:` keys. Home Assistant
therefore restores the last selected values instead of resetting them on every
restart. After adding these helpers for the first time, select `auto` and set
your desired values once. The new automatic/coil helpers have conservative
minimum values matching the recommended defaults: 0.40 C, 0.40 C, 3 minutes
and 15 minutes.

## Installation and commissioning

1. Back up the existing Pyscript and configuration.
2. Copy both Python files and both package YAML files to the paths listed
   above.
3. Keep both supplied unit mappings unchanged unless the corresponding Home
   Assistant entity IDs are renamed. Confirm `DUAL_PRIMARY_UNIT` is `daikin2`.
4. Confirm `group.daikin_watchdog_climates` contains
   both `climate.faikin_mqtt_hvac` and
   `climate.94a9903980ec_mqtt_hvac`. Do not enable the watchdog yet.
5. Restart Home Assistant so package helpers and Pyscript are loaded.
6. Run `pyscript.daikin_dual_validate_configuration`; resolve every `fail`
   issue. The supplied `sensor.iv_tulo_lampotila` mapping is already confirmed
   as genuine outside air in `DUAL_CONFIRMED_OUTDOOR_SENSORS`. For Daikin2,
   confirm the reported `min_temp` is no higher than 16 C.
7. Optionally run `pyscript.daikin_dual_apply_recommended_defaults` once.
8. Turn on `input_boolean.daikin_controller_shadow_mode`, select `auto`, and
   verify the mode, command and health diagnostics without physical output.
9. Turn shadow mode off. No learning reset is required when upgrading from the
   immediately preceding release.
10. Select `off + FAN`, run `pyscript.daikin_dual_calibrate_fan_rpm`, and wait
    until `sensor.daikin_dual_fan_rpm_calibration` reports `complete`. Do not
    change the shared mode during calibration.
11. Keep `input_boolean.daikin_summer_mode` off initially.
12. Select `heat` and verify `sensor.daikin_dual_mode_status` is not blocked.
13. Confirm the effective setpoint starts/stops the lead unit with hysteresis.
14. Verify the command status normally moves from `pending` to `acknowledged`
    for mode, 31 C target and Demand. A summer 16 C target additionally passes
    through `verifying` for 20 seconds. `recovering` is safe and self-healing;
    investigate only if it persists. Do not continue with a terminal `failed`
    command.
15. Verify `sensor.daikin_dual_learning_status` reports `warming_up` only while
   the climate entity's `hvac_action` confirms real operation.
16. Verify a real heat-mode defrost freezes the Demand and Quiet state of all
    units that were participating in heating.
17. Verify Daikin1 stages as the assist below the heating assistance threshold
   and the shared freeze requires both units to have been heating.
18. Enable summer mode, select `cool`, and verify Daikin2 acknowledges physical
    `cool`, target 16 C and fan `lowMedium`, while the logical mode remains `cool`.
    Confirm low liquid temperature does not set the defrost entity or freeze
    Demand.
19. Select `dry` with humidity above target and
    verify `sensor.daikin_dual_mode_status` reports
    `coordinated_reheat_drying: true`. Confirm Daikin2 is in physical `dry`
    with target 16 C and `lowMedium` fan (level 2/5). Confirm Daikin1 remains off while warm,
    enters only `heat` near the effective setpoint with target 31 C and fan
    `mediumHigh`, and never receives cool/dry. Test the pause/resume thresholds
    conservatively.
20. Confirm a cool/dry stop restores the remaining coil-dry deadline across a
    test restart instead of starting another 15 minutes, and confirm Daikin2's
    fan is restored to `auto` when leaving humidity drying.
21. Confirm every power sensor is in watts and verify
    `sensor.daikin_dual_cop_efficiency` shows a valid live Daikin1 COP as soon
    as the configured COP and power readings are valid. Its
    `learning_eligible` attribute remains false until the 12-minute
    actual-running gate opens.
22. Verify the configured Daikin2 power/COP mappings produce a power-weighted
    combined COP and both unit Demand values appear in the context key. Collect
    at least ten valid minutes for each load being compared.
23. Collect optimizer shadow data before considering optimizer `auto` mode.
24. Test the independent watchdog by temporarily using a three-minute timeout
    in a controlled period. Confirm the persistent notification appears and
    neither climate entity changes mode. Restore four minutes and enable the
    watchdog for normal use.

If a Daikin climate entity does not expose `hvac_action`, control continues but
learning remains blocked with `hvac_action_unknown`; this is intentional. A
compressor-frequency/running entity can still improve defrost detection, but
it does not replace `hvac_action` for learning. If an explicit defrost binary
sensor is available, configure `DEFROST_ENTITY`; it is preferred over liquid
inference and does not require periodic reports while its recognized state
remains unchanged. This release configures both units as:

```python
"DEFROST_ENTITY": "sensor.daikin1_defrost",  # Daikin1
"DEFROST_ENTITY": "sensor.daikin2_defrost",
```
