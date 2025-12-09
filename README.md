Online-RLS (adaptive) controller for Daikin demand-select.

Requirements: 
Pyscript Python scripting (HACS)
Helpers & sensors & hardware

Features:
Demand control: minimum 30%, maximum 100%
6-hour outdoor forecast average as feed-forward
forced directional step (always at least step_limit toward the setpoint when outside the deadband)
deadband, step limit, monotonicity, and select option snapping
learning (RLS) is frozen during defrost (sensor.faikin_liquid < 20)
parameter storage in a persistent STORE_ENTITY (only theta is stored, not P)
DYNAMIC ELECTRICITY PRICE CONTROL: sensor.day_ahead_price
UI-adjustable settings: electricity-price sensitivity, min/max scaling factor, maximum setpoint drop
