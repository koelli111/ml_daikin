# pyscript/daikin_ml_multi.py
# Shared-zone Daikin controller for one or two units.
# Release: 2026-08-21-separate-areas-r10-r2 (Pyscript-compatible)

import time
from math import isfinite, log as _math_log

import json

# Extra stability tuning (CPU-fix v3)
ABOVE_SP_FAST_DOWN_MULT = 2.0  # faster demand decrease when above effective setpoint
ABOVE_SP_FAST_DOWN_MIN  = 2.0  # minimum % points to drop per control tick when above setpoint


# ============================================================
# Runtime helpers: per-tick cache + write-guards (HA friendly)
# ============================================================

class _TickCache:
    """Small per-tick cache to avoid repeated state.get/state.getattr calls.

    NOTE: Do not use __slots__ here. Some Pyscript/HA internals may attach
    wrapper attributes to objects (and/or use weak references). A normal
    instance __dict__ avoids those runtime errors.
    """
    def __init__(self):
        self._state = {}
        self._attrs = {}

    def get(self, entity_id, default=None):
        if not entity_id:
            return default
        if entity_id in self._state:
            v = self._state[entity_id]
        else:
            try:
                v = state.get(entity_id)
            except Exception:
                v = None
            self._state[entity_id] = v
        return default if v is None else v

    def getattr(self, entity_id, default=None):
        if not entity_id:
            return default
        if entity_id in self._attrs:
            a = self._attrs[entity_id]
        else:
            try:
                a = state.getattr(entity_id)
            except Exception:
                a = None
            self._attrs[entity_id] = a
        return default if a is None else a

    def get_float(self, entity_id, default=float("nan")):
        v = self.get(entity_id, None)
        try:
            f = float(v)
        except Exception:
            f = float(default)
        return f if isfinite(f) else float(default)

    def get_str(self, entity_id, default=""):
        v = self.get(entity_id, None)
        return str(v) if v is not None else str(default)



def _get_indoor_temp(u, cache=None):
    """Return indoor temperature as float.

    Supports an optional secondary indoor sensor (unit key: 'INDOOR2').
    - If both sensors provide finite numeric values, returns their average.
    - If one sensor is NaN / non-numeric / unavailable, uses the other one.
    - If neither is valid, returns NaN.
    """
    ent1 = u.get("INDOOR")
    ent2 = u.get("INDOOR2")

    def _read(ent):
        if not ent or not isinstance(ent, str):
            return float("nan")
        try:
            if cache is not None:
                return float(cache.get_float(ent, default=float("nan")))
            return float(state.get(ent))
        except Exception:
            return float("nan")

    t1 = _read(ent1)
    t2 = _read(ent2)

    ok1 = isfinite(t1)
    ok2 = isfinite(t2)

    if ok1 and ok2:
        return 0.5 * (t1 + t2)
    if ok1:
        return t1
    if ok2:
        return t2
    return float("nan")
# Avoid spamming HA service calls if nothing changes
def _set_input_number_if_needed(entity_id, value, eps=0.01):
    try:
        cur = float(state.get(entity_id))
        if isfinite(cur) and isfinite(float(value)) and abs(cur - float(value)) <= float(eps):
            return False
    except Exception:
        pass
    try:
        input_number.set_value(entity_id=entity_id, value=value)
        return True
    except Exception as e:
        log.debug("Daikin ML: input_number.set_value failed for %s -> %s: %s", entity_id, value, e)
        return False

def _set_switch_if_needed(entity_id, on):
    try:
        cur = str(state.get(entity_id) or "").lower()
        want_on = bool(on)
        is_on = (cur == "on")
        if want_on == is_on:
            return False
    except Exception:
        pass
    expected = "on" if bool(on) else "off"
    return _command_request("switch", entity_id, expected, "demand_quiet")

def _set_select_if_needed(entity_id, option):
    try:
        cur = state.get(entity_id)
        if cur is not None and str(cur) == str(option):
            return False
    except Exception:
        pass
    return _command_request("select", entity_id, str(option), "demand_limit")

# Cache select numeric options to avoid repeated state.getattr parsing.
_SELECT_OPTS_CACHE = {}  # (select_entity, options_tuple) -> (nums, has_pct)



# ============================================================
# CONFIG IMPORT
# (Units + global constants live in daikin_ml_multi_config.py)
# ============================================================
from daikin_ml_multi_config import *

# Diagnostic inventory only; no state trigger is registered from this list.
_INDOOR_TRIGGERS = []
for _configured_unit in DAIKINS:
    for _indoor_key in ("INDOOR", "INDOOR2"):
        _indoor_entity = _configured_unit.get(_indoor_key)
        if _indoor_entity and _indoor_entity not in _INDOOR_TRIGGERS:
            _INDOOR_TRIGGERS.append(_indoor_entity)

# ============================================================
# PHYSICAL COMMAND TRACKING + FULL-CONTROLLER SHADOW MODE
# ============================================================
# Every physical command is represented by an expectation. A separate 10-second
# monitor acknowledges it from entity state, keeps a valid mismatch under
# backoff-based reconciliation, or records a proven invalid expectation.
# Keeping this below the config import lets the write guards above use it
# without duplicating service-call logic.
_command_pending = {}       # "kind|entity" -> command dict
_command_last = {}          # same key -> last completed/would-send command
_command_failures = {}      # same key -> retained failed command
_command_sequence = 0
_command_suppressed_count = 0
_command_cancelled_count = 0
_entity_generation = {}    # physical entity -> newest desired-state generation

# One complete desired record owns every controllable output of a unit.  The
# reconciler is the only normal path that turns these records into service
# calls.  This prevents independent mode/off/target commands from surviving a
# newer, contradictory plant decision.
_desired_generation = 0
_desired_units = {}         # unit name -> authoritative desired record
_unit_runtime = {}          # unit name -> restart-safe physical on/off timers
_DESIRED_KEEP = object()


def _controller_shadow_enabled(cache=None):
    ent = globals().get("DUAL_CONTROLLER_SHADOW_HELPER", "input_boolean.daikin_controller_shadow_mode")
    try:
        value = cache.get_str(ent, default="off") if cache is not None else str(state.get(ent) or "off")
    except Exception:
        value = "off"
    return str(value).strip().lower() == "on"


def _command_key(kind, entity_id):
    return "%s|%s" % (str(kind), str(entity_id))


def _anomaly_control_hold_active():
    """Return True while an anomaly freezes physical controller commands."""
    return bool(
        globals().get("_safety_degraded_active", False) or
        str(globals().get("_safety_health", "ok")) == "fault"
    )


def _command_actual(kind, entity_id):
    try:
        if kind == "target":
            attrs = state.getattr(entity_id) or {}
            return attrs.get("temperature")
        if kind == "fan":
            attrs = state.getattr(entity_id) or {}
            return attrs.get("fan_mode")
        return state.get(entity_id)
    except Exception:
        return None


def _command_matches(kind, actual, expected):
    if actual is None:
        return False
    if kind == "target":
        try:
            return abs(float(actual) - float(expected)) < 0.11
        except Exception:
            return False
    if kind == "mode":
        # Home Assistant's canonical HVAC state for ventilation is
        # ``fan_only``. Keep acknowledgement based on the physical state; the
        # shared selector label ``off_fan`` is only a logical controller mode.
        def _physical_mode(value):
            text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
            while "__" in text:
                text = text.replace("__", "_")
            if text in ("fan", "fanonly", "fan_only"):
                return "fan_only"
            return text
        return _physical_mode(actual) == _physical_mode(expected)
    return str(actual).strip().lower() == str(expected).strip().lower()


def _command_stable_ack_seconds(kind, expected):
    """Require a dwell only for the fixed summer target.

    MQTT climate entities can report an optimistic mode/temperature update
    before the indoor unit has finished applying the mode transition.  The
    unit may then publish its remembered cooling target (commonly 18 C).  A
    first 16 C report is therefore provisional: keep the same command alive
    until 16 C has remained stable across later housekeeping observations.
    """
    if str(kind) != "target":
        return 0.0
    try:
        value = float(expected)
    except Exception:
        return 0.0
    summer_targets = []
    for key, default in (
        ("DUAL_COOL_HVAC_TARGET_C", 16.0),
        ("DUAL_DRY_HVAC_TARGET_C", 16.0),
    ):
        try:
            candidate = float(globals().get(key, default))
        except Exception:
            continue
        if isfinite(candidate):
            summer_targets.append(candidate)
    for candidate in summer_targets:
        if abs(value - candidate) < 0.11:
            return max(
                0.0,
                float(globals().get(
                    "DUAL_SUMMER_TARGET_ACK_STABLE_SECONDS", 20.0,
                )),
            )
    return 0.0


def _command_entity_available(entity_id):
    """Return False only for a positively unavailable entity state."""
    try:
        raw = state.get(entity_id)
    except Exception:
        return False
    if raw is None:
        return False
    return str(raw).strip().lower() not in (
        "", "unknown", "unavailable", "none",
    )


def _command_capability_error(kind, entity_id, expected):
    """Reject only expectations contradicted by advertised capabilities.

    Missing capability attributes are treated as unknown, not unsupported. That
    distinction keeps startup and temporarily incomplete MQTT discovery
    fail-tolerant while still preventing permanent retries of a value Home
    Assistant has explicitly declared invalid.
    """
    try:
        attrs = state.getattr(entity_id) or {}
    except Exception:
        attrs = {}

    advertised = None
    label = None
    if kind == "select":
        advertised = attrs.get("options")
        label = "options"
    elif kind == "fan":
        advertised = attrs.get("fan_modes")
        label = "fan_modes"
    elif kind == "mode":
        advertised = attrs.get("hvac_modes")
        label = "hvac_modes"

    if isinstance(advertised, (list, tuple)) and advertised:
        supported = False
        for option in advertised:
            if _command_matches(kind, option, expected):
                supported = True
                break
        if not supported:
            return "unsupported_expected_%s_not_in_%s" % (
                str(expected), str(label),
            )

    if kind == "target":
        try:
            value = float(expected)
        except Exception:
            return "invalid_target_%s" % str(expected)
        try:
            minimum = attrs.get("min_temp")
            if minimum is not None and value < float(minimum) - 0.01:
                return "unsupported_target_%.1f_below_min_temp_%.1f" % (
                    value, float(minimum),
                )
        except Exception:
            pass
        try:
            maximum = attrs.get("max_temp")
            if maximum is not None and value > float(maximum) + 0.01:
                return "unsupported_target_%.1f_above_max_temp_%.1f" % (
                    value, float(maximum),
                )
        except Exception:
            pass
    return None


def _command_error_is_permanent(error):
    """Classify explicit validation/capability errors as non-recoverable."""
    text = str(error or "").strip().lower()
    for marker in (
        "not_valid", "not valid", "unsupported", "not supported",
        "invalid option", "out of range", "outside the allowed",
        "servicevalidationerror",
    ):
        if marker in text:
            return True
    return False


def _command_recovery_delay(cmd, retry_s):
    initial = max(
        retry_s,
        float(globals().get("DUAL_COMMAND_RECOVERY_INITIAL_SECONDS", 30.0)),
    )
    maximum = max(
        initial,
        float(globals().get("DUAL_COMMAND_RECOVERY_MAX_SECONDS", 300.0)),
    )
    factor = max(
        1.0,
        float(globals().get("DUAL_COMMAND_RECOVERY_BACKOFF_FACTOR", 2.0)),
    )
    cycle = max(1, int((cmd or {}).get("recovery_cycles") or 1))
    try:
        delay = initial * (factor ** max(0, cycle - 1))
    except Exception:
        delay = maximum
    return min(maximum, max(retry_s, delay))


def _command_mark_terminal(key, cmd, error, actual=None, now=None):
    """Retain a real invalid/unsupported command failure for diagnostics."""
    now = time.time() if now is None else float(now)
    cmd["status"] = "failed"
    cmd["failure_class"] = "terminal"
    cmd["failed_at"] = now
    cmd["last_error"] = str(error)
    if actual is not None:
        cmd["actual"] = actual
    _command_failures[key] = cmd
    _command_pending.pop(key, None)
    log.error(
        "Daikin command terminal failure: %s %s expected=%s actual=%s error=%s",
        cmd.get("kind"), cmd.get("entity_id"), cmd.get("expected"),
        cmd.get("actual"), str(error),
    )


def _command_attempt(key, cmd, now, retry_s):
    """Attempt a command once, or wait safely for entity recovery.

    Returns ``terminal`` only when advertised capabilities or the service call
    explicitly prove the expectation invalid. Availability gaps and ordinary
    service errors remain recoverable.
    """
    kind = cmd.get("kind")
    entity_id = cmd.get("entity_id")
    expected = cmd.get("expected")
    if not _command_entity_available(entity_id):
        cmd["status"] = "waiting_entity"
        cmd["availability_waits"] = int(cmd.get("availability_waits") or 0) + 1
        cmd["last_warning"] = "entity_unavailable_waiting"
        cmd["next_retry_at"] = now + retry_s
        return "waiting"

    capability_error = _command_capability_error(kind, entity_id, expected)
    if capability_error:
        _command_mark_terminal(
            key, cmd, capability_error,
            actual=_command_actual(kind, entity_id), now=now,
        )
        return "terminal"

    try:
        # A new physical write starts a new stability-observation window.
        cmd["match_since"] = 0.0
        _command_service_call(kind, entity_id, expected)
        cmd["last_error"] = None
        cmd["status"] = (
            "recovering" if int(cmd.get("recovery_cycles") or 0) > 0
            else "pending"
        )
        outcome = "sent"
    except Exception as e:
        cmd["last_error"] = str(e)
        cmd["service_error_count"] = int(cmd.get("service_error_count") or 0) + 1
        if _command_error_is_permanent(e):
            _command_mark_terminal(
                key, cmd, str(e),
                actual=_command_actual(kind, entity_id), now=now,
            )
            return "terminal"
        cmd["status"] = "retrying"
        cmd["last_warning"] = "transient_service_error"
        outcome = "transient_error"

    cmd["attempts"] = int(cmd.get("attempts") or 0) + 1
    cmd["attempts_in_cycle"] = int(cmd.get("attempts_in_cycle") or 0) + 1
    cmd["last_attempt_at"] = now
    cmd["next_retry_at"] = now + retry_s
    return outcome


def _command_service_call(kind, entity_id, expected):
    if kind == "mode":
        climate.set_hvac_mode(entity_id=entity_id, hvac_mode=str(expected))
    elif kind == "off":
        climate.turn_off(entity_id=entity_id)
    elif kind == "target":
        climate.set_temperature(entity_id=entity_id, temperature=round(float(expected), 1))
    elif kind == "fan":
        climate.set_fan_mode(entity_id=entity_id, fan_mode=str(expected))
    elif kind == "select":
        select.select_option(entity_id=entity_id, option=str(expected))
    elif kind == "switch":
        if str(expected).lower() == "on":
            switch.turn_on(entity_id=entity_id)
        else:
            switch.turn_off(entity_id=entity_id)
    else:
        raise ValueError("unsupported command kind: %s" % str(kind))


def _command_snapshot(cmd, now=None):
    now = time.time() if now is None else float(now)
    out = dict(cmd or {})
    requested_at = float(out.get("requested_at") or now)
    out["age_seconds"] = max(0.0, now - requested_at)
    return out


def _command_recovering_count():
    count = 0
    for cmd in _command_pending.values():
        if str((cmd or {}).get("status")) in (
            "recovering", "retrying", "waiting_entity",
        ):
            count += 1
    return count


def _command_verifying_count():
    count = 0
    for cmd in _command_pending.values():
        if str((cmd or {}).get("status")) == "verifying":
            count += 1
    return count


def _command_status_value():
    if _command_failures:
        return "failed"
    if _command_recovering_count():
        return "recovering"
    if _command_verifying_count():
        return "verifying"
    if _command_pending:
        return "pending"
    return "acknowledged"


def _publish_command_status():
    sensor = globals().get("DUAL_COMMAND_STATUS_SENSOR", "sensor.daikin_dual_command_status")
    now = time.time()
    pending = []
    completed = []
    failed = []
    for cmd in _command_pending.values():
        pending.append(_command_snapshot(cmd, now))
    for cmd in _command_last.values():
        completed.append(_command_snapshot(cmd, now))
    for cmd in _command_failures.values():
        failed.append(_command_snapshot(cmd, now))
    latest = None
    for cmd in completed + pending + failed:
        if latest is None or float(cmd.get("requested_at") or 0.0) > float(latest.get("requested_at") or 0.0):
            latest = cmd
    recovering_count = _command_recovering_count()
    verifying_count = _command_verifying_count()
    if failed:
        value = "failed"
    elif recovering_count:
        value = "recovering"
    elif verifying_count:
        value = "verifying"
    elif pending:
        value = "pending"
    elif latest and latest.get("status") == "would_send":
        value = "shadow"
    elif latest and latest.get("status") in (
        "suppressed_anomaly_hold", "cancelled_anomaly_hold"
    ):
        value = "held"
    else:
        value = "acknowledged"
    try:
        state.set(
            sensor,
            value=value,
            shadow_mode=_controller_shadow_enabled(),
            pending_count=len(pending),
            recovering_count=int(recovering_count),
            verifying_count=int(verifying_count),
            failed_count=len(failed),
            terminal_failure_count=len(failed),
            control_hold_active=_anomaly_control_hold_active(),
            suppressed_count=int(_command_suppressed_count),
            cancelled_count=int(_command_cancelled_count),
            last_command=latest,
            pending=pending,
            failures=failed,
            desired_generation=int(_desired_generation),
            desired_units=dict(_desired_units),
            entity_generations=dict(_entity_generation),
            unit_runtime=dict(_unit_runtime),
        )
    except Exception as e:
        log.debug("Daikin dual: failed to publish command status: %s", e)


def _command_request(
    kind, entity_id, expected, reason, allow_during_anomaly=False,
    generation=None,
):
    """Request one physical state and let the monitor verify/retry it."""
    global _command_sequence, _command_suppressed_count
    if not entity_id:
        return False
    now = time.time()
    key = _command_key(kind, entity_id)
    if generation is None:
        generation = int(_entity_generation.get(str(entity_id), 0))
    generation = int(generation)
    newest_generation = int(_entity_generation.get(str(entity_id), generation))
    if generation < newest_generation:
        return False
    existing = _command_pending.get(key)
    actual = _command_actual(kind, entity_id)
    if _command_matches(kind, actual, expected):
        stable_s = _command_stable_ack_seconds(kind, expected)
        if (
            existing is not None and
            str(existing.get("expected")) == str(expected) and
            stable_s > 0.0
        ):
            match_since = float(existing.get("match_since") or 0.0)
            if match_since <= 0.0:
                match_since = now
                existing["match_since"] = now
            existing["actual"] = actual
            if now - match_since < stable_s:
                existing["status"] = "verifying"
                existing["last_warning"] = "stable_ack_pending"
                existing["stable_ack_seconds"] = stable_s
                _publish_command_status()
                return False
        old = _command_pending.pop(key, None)
        if old is not None:
            old["status"] = "acknowledged"
            old["acknowledged_at"] = now
            old["actual"] = actual
            _command_last[key] = old
        _command_failures.pop(key, None)
        _publish_command_status()
        return False

    # An anomaly must never create a physical transition, especially an HVAC
    # off command. Record the suppressed intent for diagnostics, but do not
    # queue it: a stale command must not execute later when recovery completes.
    if _anomaly_control_hold_active() and not bool(allow_during_anomaly):
        _command_sequence += 1
        _command_suppressed_count += 1
        cmd = {
            "sequence": int(_command_sequence),
            "generation": generation,
            "kind": str(kind),
            "entity_id": str(entity_id),
            "expected": expected,
            "actual": actual,
            "reason": str(reason),
            "requested_at": now,
            "completed_at": now,
            "attempts": 0,
            "status": "suppressed_anomaly_hold",
            "last_error": None,
        }
        _command_last[key] = cmd
        _command_pending.pop(key, None)
        _publish_command_status()
        return False

    if existing is not None and str(existing.get("expected")) == str(expected):
        if float(existing.get("match_since") or 0.0) > 0.0:
            # A late device report invalidated the provisional acknowledgement.
            # Retry on this monitor pass instead of waiting for an old backoff.
            existing["match_since"] = 0.0
            existing["last_warning"] = "stable_ack_lost"
            existing["next_retry_at"] = now
            if str(existing.get("status")) == "verifying":
                existing["status"] = (
                    "recovering"
                    if int(existing.get("recovery_cycles") or 0) > 0
                    else "pending"
                )
        return False
    failed = _command_failures.get(key)
    if (
        failed is not None and
        int(failed.get("generation") or 0) == generation and
        str(failed.get("expected")) == str(expected)
    ):
        # A terminal failure belongs to this desired generation. Reconciliation
        # must not create an infinite new retry cycle; only a newer desired
        # generation may try again.
        return False

    _command_sequence += 1
    cmd = {
        "sequence": int(_command_sequence),
        "generation": generation,
        "kind": str(kind),
        "entity_id": str(entity_id),
        "expected": expected,
        "actual": actual,
        "reason": str(reason),
        "requested_at": now,
        "last_attempt_at": 0.0,
        "next_retry_at": now,
        "attempts": 0,
        "status": "pending",
        "last_error": None,
        "match_since": 0.0,
        "allow_during_anomaly": bool(allow_during_anomaly),
    }
    if _controller_shadow_enabled():
        cmd["status"] = "would_send"
        cmd["completed_at"] = now
        _command_last[key] = cmd
        _command_pending.pop(key, None)
        _command_failures.pop(key, None)
        _publish_command_status()
        return True

    _command_failures.pop(key, None)
    _command_pending[key] = cmd
    retry_s = max(
        5.0, float(globals().get("DUAL_COMMAND_RETRY_SECONDS", 10.0))
    )
    _command_attempt(key, cmd, now, retry_s)
    _publish_command_status()
    return True


def _command_monitor_tick():
    """Acknowledge and retry commands; called by its own 10-second trigger."""
    now = time.time()
    retry_s = max(5.0, float(globals().get("DUAL_COMMAND_RETRY_SECONDS", 10.0)))
    default_max_attempts = max(
        1, int(globals().get("DUAL_COMMAND_MAX_ATTEMPTS", 3))
    )
    retention = max(60.0, float(globals().get("DUAL_COMMAND_FAILURE_RETENTION_S", 1800.0)))
    for key in list(_command_failures.keys()):
        cmd = _command_failures.get(key) or {}
        if now - float(cmd.get("failed_at") or now) > retention:
            _command_failures.pop(key, None)
    for key in list(_command_pending.keys()):
        cmd = _command_pending.get(key)
        if cmd is None:
            continue
        kind = cmd.get("kind")
        entity_id = cmd.get("entity_id")
        if int(cmd.get("generation") or 0) < int(
            _entity_generation.get(str(entity_id), 0)
        ):
            cmd["status"] = "superseded"
            cmd["completed_at"] = now
            _command_last[key] = cmd
            _command_pending.pop(key, None)
            continue
        expected = cmd.get("expected")
        actual = _command_actual(kind, entity_id)
        cmd["actual"] = actual
        if _command_matches(kind, actual, expected):
            stable_s = _command_stable_ack_seconds(kind, expected)
            if stable_s > 0.0:
                match_since = float(cmd.get("match_since") or 0.0)
                if match_since <= 0.0:
                    match_since = now
                    cmd["match_since"] = now
                if now - match_since < stable_s:
                    cmd["status"] = "verifying"
                    cmd["last_warning"] = "stable_ack_pending"
                    cmd["stable_ack_seconds"] = stable_s
                    continue
            cmd["status"] = "acknowledged"
            cmd["acknowledged_at"] = now
            _command_last[key] = cmd
            _command_pending.pop(key, None)
            _command_failures.pop(key, None)
            continue
        if float(cmd.get("match_since") or 0.0) > 0.0:
            cmd["match_since"] = 0.0
            cmd["last_warning"] = "stable_ack_lost"
            cmd["next_retry_at"] = min(
                now, float(cmd.get("next_retry_at") or now)
            )
            if str(cmd.get("status")) == "verifying":
                cmd["status"] = (
                    "recovering"
                    if int(cmd.get("recovery_cycles") or 0) > 0
                    else "pending"
                )
        if (
            _anomaly_control_hold_active() and
            not bool(cmd.get("allow_during_anomaly"))
        ):
            # Safety evaluation cancels pending commands when an anomaly first
            # appears. This guard also closes the narrow race where the monitor
            # sees the hold before that cancellation pass.
            continue
        if now < float(cmd.get("next_retry_at") or 0.0):
            continue
        attempts = int(cmd.get("attempts") or 0)
        command_max_attempts = default_max_attempts
        if kind == "target":
            command_max_attempts = max(
                default_max_attempts,
                int(globals().get(
                    "DUAL_TARGET_COMMAND_MAX_ATTEMPTS",
                    default_max_attempts,
                )),
            )
        attempts_in_cycle = int(cmd.get("attempts_in_cycle") or 0)
        if attempts_in_cycle >= command_max_attempts:
            cmd["recovery_cycles"] = int(cmd.get("recovery_cycles") or 0) + 1
            cmd["attempts_in_cycle"] = 0
            cmd["status"] = "recovering"
            cmd["failure_class"] = "recoverable_mismatch"
            cmd["last_error"] = None
            cmd["last_warning"] = "acknowledgement_delayed"
            if not float(cmd.get("recovering_since") or 0.0):
                cmd["recovering_since"] = now
            cmd["next_retry_at"] = now + _command_recovery_delay(cmd, retry_s)
            log.warning(
                "Daikin command still reconciling: %s %s expected=%s actual=%s cycle=%s next_retry_s=%.0f",
                kind, entity_id, expected, actual,
                cmd.get("recovery_cycles"),
                float(cmd.get("next_retry_at") or now) - now,
            )
            continue
        _command_attempt(key, cmd, now, retry_s)
    _publish_command_status()


def _cancel_pending_commands_for_anomaly(reason="anomaly_control_hold"):
    """Cancel pending physical intents without touching the HVAC equipment."""
    global _command_cancelled_count
    now = time.time()
    for key in list(_command_pending.keys()):
        cmd = _command_pending.get(key)
        if cmd is None:
            continue
        if bool(cmd.get("allow_during_anomaly")):
            continue
        _command_pending.pop(key, None)
        cmd = dict(cmd)
        cmd["status"] = "cancelled_anomaly_hold"
        cmd["completed_at"] = now
        cmd["cancel_reason"] = str(reason)
        _command_last[key] = cmd
        _command_cancelled_count += 1
    _publish_command_status()


def _unit_name(u):
    return str((u or {}).get("name", "daikin?"))


def _desired_entities(u):
    out = []
    for key in ("CLIMATE", "SELECT", "QUIET_OUTDOOR_SWITCH"):
        entity_id = (u or {}).get(key)
        if entity_id:
            out.append(str(entity_id))
    return out


def _supersede_unit_commands(u, generation, reason="new_desired_state"):
    """Invalidate every older command touching this physical unit."""
    global _command_cancelled_count
    generation = int(generation)
    entities = set(_desired_entities(u))
    for entity_id in entities:
        _entity_generation[entity_id] = generation
    now = time.time()
    for key in list(_command_pending.keys()):
        cmd = _command_pending.get(key)
        if not cmd or str(cmd.get("entity_id")) not in entities:
            continue
        if int(cmd.get("generation") or 0) >= generation:
            continue
        _command_pending.pop(key, None)
        cmd = dict(cmd)
        cmd["status"] = "superseded"
        cmd["completed_at"] = now
        cmd["superseded_by_generation"] = generation
        cmd["supersede_reason"] = str(reason)
        _command_last[key] = cmd
        _command_cancelled_count += 1


def _desired_update_unit(
    u, active=_DESIRED_KEEP, mode=_DESIRED_KEEP,
    target=_DESIRED_KEEP, demand_option=_DESIRED_KEEP,
    quiet=_DESIRED_KEEP, fan_mode=_DESIRED_KEEP, reason="controller",
    force_stop=_DESIRED_KEEP, force_generation=False,
):
    """Merge one decision into the unit's complete desired state."""
    global _desired_generation
    name = _unit_name(u)
    old = _desired_units.get(name) or {
        "active": False,
        "mode": "off",
        "target": None,
        "demand_option": None,
        "quiet": None,
        "fan_mode": globals().get("DUAL_DEFAULT_FAN_MODE", "auto"),
        "force_stop": False,
        "reason": "startup",
    }
    new = dict(old)
    if active is not _DESIRED_KEEP:
        new["active"] = bool(active)
    if mode is not _DESIRED_KEEP:
        new["mode"] = str(mode or "off").strip().lower()
    if target is not _DESIRED_KEEP:
        new["target"] = target
    if demand_option is not _DESIRED_KEEP:
        new["demand_option"] = demand_option
    if quiet is not _DESIRED_KEEP:
        new["quiet"] = quiet
    if fan_mode is not _DESIRED_KEEP:
        new["fan_mode"] = fan_mode
    if force_stop is not _DESIRED_KEEP:
        new["force_stop"] = bool(force_stop)
    new["reason"] = str(reason)
    new["unit_name"] = name

    compare_keys = (
        "active", "mode", "target", "demand_option", "quiet", "fan_mode",
        "force_stop",
    )
    changed = False
    for key in compare_keys:
        if old.get(key) != new.get(key):
            changed = True
            break
    if changed or "generation" not in old or bool(force_generation):
        _desired_generation += 1
        new["generation"] = int(_desired_generation)
        new["updated_at"] = time.time()
        _desired_units[name] = new
        _supersede_unit_commands(
            u, new["generation"], reason="desired:%s" % str(reason)
        )
    else:
        new["generation"] = int(old.get("generation") or 0)
        new["updated_at"] = old.get("updated_at")
        _desired_units[name] = new
    return dict(new)


def _unit_runtime_record(u):
    name = _unit_name(u)
    runtime = _unit_runtime.get(name)
    if not isinstance(runtime, dict):
        runtime = {
            "initialized": False,
            "active": False,
            "mode": "unknown",
            "mode_changed_at": 0.0,
            "started_at": 0.0,
            "stopped_at": 0.0,
            "last_transition_at": 0.0,
        }
        _unit_runtime[name] = runtime
    return runtime


def _refresh_unit_runtime(u, cache=None, now=None):
    """Track each climate entity's physical state independently."""
    cache = cache or _TickCache()
    now = time.time() if now is None else float(now)
    runtime = _unit_runtime_record(u)
    climate_ent = u.get("CLIMATE")
    raw = cache.get_str(climate_ent, default="").strip().lower()
    if raw in ("", "unknown", "unavailable", "none"):
        return runtime
    active = raw != "off"
    previous_mode = str(runtime.get("mode") or "unknown").strip().lower()
    if not bool(runtime.get("initialized")):
        runtime["initialized"] = True
        # A loaded timestamp is valid only when it describes the same physical
        # state. Otherwise reconstruct the transition at startup.
        if bool(runtime.get("active")) != active:
            runtime["started_at"] = now if active else 0.0
            runtime["stopped_at"] = now if not active else 0.0
            runtime["last_transition_at"] = now
        elif active and not float(runtime.get("started_at") or 0.0):
            runtime["started_at"] = now
            runtime["last_transition_at"] = now
        elif not active and not float(runtime.get("stopped_at") or 0.0):
            # Without a retained timer, use a conservative full off interval.
            runtime["stopped_at"] = now
            runtime["last_transition_at"] = now
    elif bool(runtime.get("active")) != active:
        runtime["started_at"] = now if active else 0.0
        runtime["stopped_at"] = now if not active else 0.0
        runtime["last_transition_at"] = now
    if previous_mode != raw or not float(runtime.get("mode_changed_at") or 0.0):
        runtime["mode_changed_at"] = now
    runtime["active"] = active
    runtime["mode"] = raw
    return runtime


def _refresh_all_unit_runtimes(cache=None, now=None):
    cache = cache or _TickCache()
    now = time.time() if now is None else float(now)
    for u in DAIKINS:
        _refresh_unit_runtime(u, cache, now)


def _unit_minimum_times(cache=None):
    cache = cache or _TickCache()
    limits = _dual_supervisor_limits(cache)
    return (
        max(0.0, float(limits.get("min_on_min", 0.0))) * 60.0,
        max(0.0, float(limits.get("min_off_min", 0.0))) * 60.0,
    )


def _actual_matches_entity(kind, entity_id, expected):
    return _command_matches(kind, _command_actual(kind, entity_id), expected)


def _reconcile_desired_unit(u, cache=None, now=None):
    """Apply one desired generation with safe start/stop sequencing."""
    cache = cache or _TickCache()
    now = time.time() if now is None else float(now)
    name = _unit_name(u)
    desired = _desired_units.get(name)
    if not desired:
        return
    desired_mode = str(desired.get("mode") or "off").strip().lower()
    if (
        bool(desired.get("active")) and
        desired_mode in ("cool", "dry") and
        u is not _dual_summer_conditioning_unit(desired_mode)
    ):
        # Final command-layer invariant: even a direct desired-state write or
        # a stale persisted record cannot make Daikin1 refrigerate.
        desired = _desired_update_unit(
            u,
            active=False,
            mode="off",
            target=None,
            fan_mode=_dual_desired_fan_mode(u, "off"),
            reason="reconcile_%s_role_rejected" % desired_mode,
            force_stop=False,
        )
    elif bool(desired.get("active")) and desired_mode in ("cool", "dry"):
        # A persisted record or future direct caller cannot weaken the fixed
        # summer contract. Both logical modes require Daikin2 at 16 C / lowMedium.
        expected_target = _dual_climate_target_temperature(desired_mode)
        expected_fan = _dual_desired_fan_mode(u, desired_mode)
        target_ok = (
            expected_target is not None and
            _command_matches("target", desired.get("target"), expected_target)
        )
        fan_ok = (
            _dual_fan_mode_semantic(desired.get("fan_mode")) ==
            _dual_fan_mode_semantic(expected_fan)
        )
        if not target_ok or not fan_ok:
            desired = _desired_update_unit(
                u,
                active=True,
                mode=desired_mode,
                target=(
                    round(float(expected_target), 1)
                    if expected_target is not None else None
                ),
                fan_mode=expected_fan,
                reason="reconcile_%s_parameters_canonicalized" % desired_mode,
                force_stop=False,
            )
    generation = int(desired.get("generation") or 0)
    climate_ent = u.get("CLIMATE")
    select_ent = u.get("SELECT")
    quiet_ent = u.get("QUIET_OUTDOOR_SWITCH")
    runtime = _refresh_unit_runtime(u, cache, now)
    min_on_s, min_off_s = _unit_minimum_times(cache)
    shadow = _controller_shadow_enabled(cache)

    if not bool(desired.get("active")):
        if not climate_ent:
            return
        fan_mode = _dual_resolve_fan_mode(
            u,
            desired.get("fan_mode") or globals().get("DUAL_DEFAULT_FAN_MODE", "auto"),
            cache,
        )
        if fan_mode is not None:
            # Restore the normal fan selection before leaving the drying role.
            # Do not delay an off command if the fan acknowledgement is slow.
            _command_request(
                "fan", climate_ent, fan_mode,
                desired.get("reason", "desired_fan_restore"),
                generation=generation,
            )
        on_elapsed = now - float(runtime.get("started_at") or now)
        force_stop = bool(desired.get("force_stop"))
        if bool(runtime.get("active")) and not force_stop and on_elapsed < min_on_s:
            return
        _command_request(
            "off", climate_ent, "off", desired.get("reason", "desired_off"),
            allow_during_anomaly=force_stop,
            generation=generation,
        )
        return

    # Configure Demand and Quiet before starting an off unit or changing mode.
    demand_ready = True
    demand_option = desired.get("demand_option")
    if select_ent and demand_option is not None:
        _command_request(
            "select", select_ent, str(demand_option),
            desired.get("reason", "desired_demand"),
            generation=generation,
        )
        demand_ready = shadow or _actual_matches_entity(
            "select", select_ent, str(demand_option)
        )

    quiet_ready = True
    quiet = desired.get("quiet")
    if quiet_ent and quiet is not None:
        quiet_expected = "on" if bool(quiet) else "off"
        _command_request(
            "switch", quiet_ent, quiet_expected,
            desired.get("reason", "desired_quiet"),
            generation=generation,
        )
        quiet_ready = shadow or _actual_matches_entity(
            "switch", quiet_ent, quiet_expected
        )

    requested_fan_mode = desired.get("fan_mode")
    fan_mode = _dual_resolve_fan_mode(u, requested_fan_mode, cache)
    fan_ready = requested_fan_mode is None
    if climate_ent and requested_fan_mode is not None:
        # Never send a guessed fan label. Home Assistant rejects values which
        # are not present in the climate entity's advertised fan_modes list.
        # An unresolved active-mode request therefore blocks compressor start
        # and is reported by commissioning validation, without service spam.
        fan_ready = fan_mode is not None
        if fan_mode is not None:
            _command_request(
                "fan", climate_ent, fan_mode,
                desired.get("reason", "desired_fan"),
                generation=generation,
            )
            fan_ready = shadow or _actual_matches_entity(
                "fan", climate_ent, fan_mode
            )

    if not climate_ent or not (demand_ready and quiet_ready and fan_ready):
        return
    if not bool(runtime.get("active")):
        off_elapsed = now - float(runtime.get("stopped_at") or 0.0)
        if runtime.get("stopped_at") and off_elapsed < min_off_s:
            return

    logical_mode = str(desired.get("mode") or "off").lower()
    physical_mode = _dual_physical_hvac_mode(logical_mode, u)
    _command_request(
        "mode", climate_ent, physical_mode,
        desired.get("reason", "desired_mode"),
        generation=generation,
    )
    mode_ready = shadow or _actual_matches_entity(
        "mode", climate_ent, physical_mode
    )
    target = desired.get("target")
    target_ready = bool(mode_ready)
    if (
        target_ready and target is not None and not shadow and
        logical_mode in ("cool", "dry")
    ):
        # Let the requested refrigeration mode settle before applying the
        # authoritative 16 C target. This covers both physical cool and dry.
        settle_s = max(
            0.0,
            float(globals().get(
                "DUAL_SUMMER_MODE_SETTLE_SECONDS", 20.0,
            )),
        )
        mode_changed_at = float(runtime.get("mode_changed_at") or now)
        target_ready = now - mode_changed_at >= settle_s
    if target_ready and target is not None:
        _command_request(
            "target", climate_ent, target,
            desired.get("reason", "desired_target"),
            generation=generation,
        )


def _reconcile_all_desired(cache=None, now=None):
    cache = cache or _TickCache()
    now = time.time() if now is None else float(now)
    for u in DAIKINS:
        _reconcile_desired_unit(u, cache, now)
    _publish_command_status()

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

def _select_options_nums(select_entity):
    attrs = state.getattr(select_entity) or {}
    opts = attrs.get("options") or []
    try:
        key = (str(select_entity), tuple([str(o) for o in opts]))
    except Exception:
        key = (str(select_entity), None)

    cached = _SELECT_OPTS_CACHE.get(key)
    if cached is not None:
        return cached[0], cached[1]

    nums = []
    for o in opts:
        try:
            s = str(o).replace('%', '').strip()
            v = float(s)
            if isfinite(v):
                nums.append(v)
        except Exception:
            pass
    nums.sort()
    has_pct = False
    if len(opts) > 0 and ('%' in str(opts[0])):
        has_pct = True

    # de-dup
    if nums:
        out_nums = []
        last = None
        for n in nums:
            if last is None or abs(n - last) > 1e-9:
                out_nums.append(n)
            last = n
        nums = out_nums

    _SELECT_OPTS_CACHE[key] = (nums, has_pct)
    return nums, has_pct

def _with_pct(select_entity):
    attrs = state.getattr(select_entity) or {}
    opts = attrs.get("options") or []
    return ('%' in (opts[0] if opts else ''))


def _parse_select_numeric(select_entity, option_str, default=60.0):
    """
    Parse a select option like '70%' -> 70.0.
    If parsing fails (eg. 'auto'), fall back to nearest available numeric option (or default).
    """
    try:
        v = float(str(option_str).replace('%', '').strip())
        if isfinite(v):
            return float(v)
    except Exception:
        pass

    nums, _has_pct = _select_options_nums(select_entity)
    if nums:
        try:
            d0 = float(default)
        except Exception:
            d0 = nums[0]
        best = nums[0]
        bestd = abs(nums[0] - d0)
        for n in nums:
            d = abs(n - d0)
            if d < bestd:
                bestd = d
                best = n
        return float(best) if isfinite(float(best)) else float(default)

    try:
        d = float(default)
        return d if isfinite(d) else 60.0
    except Exception:
        return 60.0

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

    unit_has_quiet = bool(u.get("QUIET_OUTDOOR_SWITCH"))
    hi = MAX_DEM_LAYER if unit_has_quiet else 100.0

    cap = _clip(cap, 0.0, hi)
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

# ------------------------------------------------------------
# NEW: post-defrost demand + hold minutes (helpers)
# ------------------------------------------------------------
def _post_defrost_hold_s(u):
    ent = u.get("POST_DEFROST_HOLD_MINUTES_HELPER")
    if ent:
        minutes = _read_float_entity(ent, POST_DEFROST_HOLD_DEFAULT_MIN)
    else:
        minutes = POST_DEFROST_HOLD_DEFAULT_MIN
    minutes = _clip(minutes, 0.0, 120.0)  # 0..120 min
    return float(minutes) * 60.0

def _post_defrost_demand_pct(u):
    ent = u.get("POST_DEFROST_DEMAND_HELPER")
    if ent:
        pct = _read_float_entity(ent, POST_DEFROST_DEMAND_DEFAULT_PCT)
    else:
        pct = POST_DEFROST_DEMAND_DEFAULT_PCT
    return _clip(pct, 0.0, 100.0)

def _compute_post_defrost_hold_option_and_quiet(u, select_entity, unit_has_quiet):
    """
    Returns (held_select_option_str, held_quiet_bool_or_none)
    """
    pct = float(_post_defrost_demand_pct(u))
    opt = _snap_to_select(select_entity, pct, 0)
    if unit_has_quiet:
        # During post-defrost hold, keep quiet ON always (safe + prevents 105 layer antics)
        return opt, True
    return opt, None


# ============================================================
# Nordpool avg window dynamic update (based on hourly forecast)
# ============================================================
def _compute_outdoor_effective_temp_for_window(outdoor_entity):
    """Use the measured outdoor sensor only; weather forecasts are excluded."""
    try:
        value = float(state.get(outdoor_entity))
        return value if isfinite(value) else 0.0
    except Exception:
        return 0.0


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
            _set_input_number_if_needed(NORDPOOL_AVG_WINDOW_HELPER, round(new_h, 2), eps=AVG_WINDOW_WRITE_EPS)
    except Exception as e:
        try:
            log.debug("Daikin ML: failed to update nordpool avg window: %s", e)
        except Exception:
            pass


def _update_nordpool_avg_window_hours_multi():
    """
    Compute and write input_number.nordpool_avg_window_hours based on the coldest effective
    outdoor temperature across all configured units (forecast+outdoor).
    """
    try:
        temps_eff = []
        for uu in DAIKINS:
            oe = uu.get("OUTDOOR")
            if oe:
                try:
                    temps_eff.append(_compute_outdoor_effective_temp_for_window(oe))
                except Exception:
                    pass
        if not temps_eff:
            return
        temp_eff = min(temps_eff)
        new_h = float(_temp_to_avg_window_hours(temp_eff))
        prev_h = float(state.get(NORDPOOL_AVG_WINDOW_HELPER) or new_h)
        if (not isfinite(prev_h)) or abs(prev_h - new_h) > AVG_WINDOW_WRITE_EPS:
            _set_input_number_if_needed(NORDPOOL_AVG_WINDOW_HELPER, round(new_h, 2), eps=AVG_WINDOW_WRITE_EPS)
    except Exception as e:
        try:
            log.debug("Daikin ML: avg window update failed: %s", e)
        except Exception:
            pass


# Per-unit applied-Demand cadence retained by the shared allocator.
_last_demand_change_ts = {}
_last_demand_sig = {}

# # System-wide demand freeze while one of multiple simultaneously-heating units
# is in defrost.  This is deliberately separate from the per-unit defrost hold:
# - at defrost entry, capture every participating unit's demand + Quiet state
# - enforce those exact states for the whole defrost
# - pause all demand control, Tin history, PI and ML updates
# - resume only after none of the participating units is defrosting
SYSTEM_DEFROST_FREEZE_ENTITY = globals().get(
    "SYSTEM_DEFROST_FREEZE_ENTITY", "pyscript.daikin_system_defrost_freeze"
)
SYSTEM_DEFROST_FREEZE_MAX_S = float(globals().get(
    "SYSTEM_DEFROST_FREEZE_MAX_S", 30.0 * 60.0
))

_system_defrost_freeze_active = False
_system_defrost_freeze_started = 0.0
_system_defrost_freeze_units = set()
_system_defrosted_units = set()
_system_frozen_select = {}     # unit -> captured select option
_system_frozen_quiet = {}      # unit -> captured Quiet bool/None
_system_prev_defrost = {}      # unit -> bool/None from previous monitor tick
_system_prev_heating = {}      # unit -> bool from previous monitor tick
_defrost_detectors = {}        # unit -> report-token driven detector state
_dual_post_defrost_holds = {}  # unit -> {until, demand, started_at}
_dual_cdp_release = {}         # unit -> adaptive Summer heating airflow state
_dual_cdp_learned_demand = {}  # unit -> lowest confirmed CDP-release Demand
_dual_cdp_temperature_curve = {}  # unit -> fan -> outdoor bucket -> bracket
_dual_fan_rpm_profiles = {}    # unit -> fan semantic -> calibrated RPM record
_dual_fan_calibration_active = False
_dual_fan_calibration_status = {
    "state": "idle", "reason": "not_run", "unit": None, "fan_mode": None,
}
_separate_area_state = {}      # unit -> independent thermostat/PI runtime
_separate_area_learning = {}   # unit/context -> independent COP/Demand means
_separate_topology_previous = False

# Shared dual-zone controller and slow energy optimizer state
_dual_cold_active = False
_dual_tin_hist = []             # list[(timestamp, zone_temperature)]
_dual_humidity_hist = []        # list[(timestamp, relative_humidity)]
_dual_learning_humidity_hist = []  # dry-mode learning only; never includes off/fan_only drift
_dual_err_int = 0.0
_dual_last_ctrl_ts = 0.0
_dual_last_sp = float("nan")
_dual_last_control_mode = None
_dual_last_total_target = float("nan")
_dual_last_startup_demand = None
_dual_sustain_by_ctx = {}       # outdoor context -> combined demand points
_dual_policy_stats = {}         # context -> action -> online score statistics
_dual_cop_stats = {}            # matched mode/outdoor/action/actual-demand COP map
_dual_current_cop_snapshot = {
    "valid": False,
    "reason": "startup",
    "combined_cop": None,
    "context": None,
}
_dual_active_action_id = None
_dual_episode = None
_dual_store_loaded = False
_dual_last_store_save_ts = 0.0
_dual_last_store_payload = None
_dual_optimizer_last_sample_ts = 0.0
_dual_optimizer_prev_power_w = None
_dual_optimizer_prev_cop_power_w = None
_dual_optimizer_prev_thermal_w = None
_dual_optimizer_prev_heating = {}
_dual_action_changed_at = 0.0
_dual_last_fast_taper_ts = 0.0
_dual_fast_tin_hist = []        # short-window samples used only by the landing governor
_dual_landing_status = {
    "active": False,
    "reason": "startup",
    "projected_error": None,
    "projected_temperature": None,
    "eta_to_setpoint_minutes": None,
    "landing_cap": None,
    "step_limited_total": None,
    "rapid_approach": False,
    "control_interval_seconds": None,
    "fast_rate_cph": None,
    "long_rate_cph": None,
    "rate_source": "none",
}
_dual_sustain_probe = {
    "state": "idle",
    "context": None,
    "candidate": None,
    "baseline": None,
    "started_at": 0.0,
    "settled_at": 0.0,
    "cooldown_until": 0.0,
    "reason": "startup",
}

# Learning is valid only after climate.hvac_action has continuously confirmed
# the commanded conditioning mode.  These fields deliberately track actual
# operation separately from the supervisor's requested/latched active state.
_dual_learning_evidence_ok = False
_dual_learning_allowed = False
_dual_learning_started_at = 0.0
_dual_learning_signature = None
_dual_learning_block_reason = "startup"
_dual_learning_actual_running = False
_dual_learning_expected_units = []
_dual_learning_actual_units = []
_dual_learning_unit_states = {}
_dual_learning_last_reset_reason = "startup"
_dual_learning_last_sample_ts = 0.0

# Effective-setpoint supervisory thermostat.  The requested mode comes from a
# Home Assistant input_select; auto resolves to one selected conditioning mode,
# and cool/dry are additionally gated by summer mode. This state is shared
# because both units condition one zone.
_dual_supervisor_initialized = False
_dual_zone_active = False
_dual_active_mode = "off"
_dual_zone_started_at = 0.0
_dual_zone_stopped_at = 0.0
_dual_mode_changed_at = 0.0
_dual_assist_by_mode = {"heat": False, "cool": False, "dry": False}
_dual_last_supervisor_reason = "startup"

# Coordinated humidity drying. The lead removes latent heat in physical dry
# while the assist supplies only the sensible heat needed to hold the zone.
# These states are independent of the ordinary same-mode allocation optimizer.
_dual_dry_reheat_active = False
_dual_dry_reheat_started_at = 0.0
_dual_dry_reheat_stopped_at = 0.0
_dual_dry_reheat_integral = 0.0
_dual_dry_reheat_last_ctrl_ts = 0.0
_dual_dry_reheat_last_demand = None
_dual_dry_lead_paused = False
_dual_dry_lead_paused_at = 0.0
_dual_dry_lead_resumed_at = 0.0
_dual_dry_coordination_status = {
    "available": False,
    "active": False,
    "lead_paused": False,
    "reason": "startup",
}

# Automatic mode-selection latch and confirmation timer. The selected mode is
# deliberately separate from the requested mode so diagnostics can show
# requested=auto and selected=heat/cool/dry/off.
_dual_auto_selected_mode = "off"
_dual_auto_candidate_mode = "off"
_dual_auto_candidate_since = 0.0
_dual_auto_selection_reason = "auto_startup"
# ``auto + fan`` uses the normal controller auto state machine. This flag only
# overrides the desired fan option; it never changes the selected HVAC mode.
_dual_auto_fan_override_active = False

# Post-cooling/drying fan-only coil drying. This is not plant/compressor
# activity: minimum compressor off-time continues to accrue while it runs.
_dual_coil_dry_active = False
_dual_coil_dry_started_at = 0.0
_dual_coil_dry_until = 0.0
_dual_coil_dry_source_mode = None
_dual_coil_dry_units = set()
_dual_coil_dry_reason = None

# Safety/health state. Anomalies may freeze new controller commands, but never
# request HVAC off. Other sensor faults degrade only the dependent feature.
_safety_health = "degraded"
_safety_reasons = ["startup"]
_safety_degraded_active = False
_safety_valid_since = 0.0
_safety_valid_readings = 0
_safety_critical_active = False
_safety_recovery_active = False
_safety_recovery_remaining_s = 0.0
_safety_original_trigger = []
_safety_original_trigger_at = 0.0
_safety_last_critical_trigger = []
_safety_last_critical_trigger_at = 0.0
_safety_feature_degradations = []
_safety_diagnostic_warnings = []
_sensor_status = {}
_sensor_last_good = {}
_sensor_timestamp_sources = {}
_sensor_report_tokens = {}
_sensor_report_sequence = 0
_sensor_report_last_entities = []
_sensor_report_last_at = 0.0
_heartbeat_sequence = 0
_heartbeat_last_success = 0.0
_heartbeat_last_error = None
_controller_error_count = 0

# Restart-safe operational state and daily diagnostics.
_runtime_store_loaded = False
_runtime_store_last_payload = None
_daily_stats_date = ""
_daily_stats = {}
_stats_last_ts = 0.0
_stats_prev_running = {}
_stats_prev_power_w = None
_runtime_stats_save_bucket = None
_replay_samples = []
_replay_last_sample_ts = 0.0

def _requested_hvac_mode(cache=None):
    """Return the user's persistent requested mode."""
    cache = cache or _TickCache()
    ent = globals().get("DUAL_HVAC_MODE_HELPER", "input_select.daikin_dual_hvac_mode")
    try:
        value = cache.get_str(ent, default=globals().get("DUAL_DEFAULT_HVAC_MODE", "heat"))
    except Exception:
        value = globals().get("DUAL_DEFAULT_HVAC_MODE", "heat")
    value = str(value).strip().lower()
    aliases = {
        "off + fan": "off_fan",
        "off+fan": "off_fan",
        "off_fan": "off_fan",
        "auto + fan": "auto_fan",
        "auto+fan": "auto_fan",
        "auto_fan": "auto_fan",
    }
    value = aliases.get(value, value)
    return value if value in (
        "off", "auto", "heat", "cool", "dry", "off_fan", "auto_fan",
    ) else "off"


def _summer_mode_enabled(cache=None):
    cache = cache or _TickCache()
    ent = globals().get("DUAL_SUMMER_MODE_HELPER", "input_boolean.daikin_summer_mode")
    try:
        return cache.get_str(ent, default="off").strip().lower() == "on"
    except Exception:
        return False


def _effective_hvac_mode(cache=None):
    """Return (requested, permitted mode, summer_enabled, block_reason)."""
    cache = cache or _TickCache()
    requested = _requested_hvac_mode(cache)
    summer = _summer_mode_enabled(cache)
    selected = requested
    if requested in ("auto", "auto_fan"):
        selected = str(_dual_auto_selected_mode or "off").strip().lower()
        if selected not in ("off", "heat", "cool", "dry"):
            selected = "off"
    if selected in ("cool", "dry") and not summer:
        return requested, "off", summer, "summer_mode_off"
    if selected == "cool" and _dual_cooling_unit() is None:
        return requested, "off", summer, "cooling_unit_unavailable"
    if selected == "dry" and _dual_drying_unit() is None:
        return requested, "off", summer, "drying_unit_unavailable"
    return requested, selected, summer, None


def _dual_named_unit(config_key, default_name=None):
    wanted = str(globals().get(config_key, default_name or "") or "").strip()
    for u in DAIKINS:
        if str(u.get("name") or "").strip() == wanted:
            return u
    return None


def _dual_drying_unit():
    """Return the only unit permitted to remove humidity."""
    return _dual_named_unit("DUAL_DRYING_UNIT", "daikin2")


def _dual_cooling_unit():
    """Return the only unit permitted to provide temperature cooling."""
    return _dual_named_unit("DUAL_COOLING_UNIT", "daikin2")


def _dual_dry_reheat_unit():
    """Return the distinct unit reserved for temperature support in dry."""
    configured = _dual_named_unit("DUAL_DRY_REHEAT_UNIT", "daikin1")
    if configured is not None and configured is not _dual_drying_unit():
        return configured
    return None


def _dual_is_drying_unit(u):
    drying = _dual_drying_unit()
    return bool(drying is not None and u is drying)


def _dual_is_cooling_unit(u):
    cooling = _dual_cooling_unit()
    return bool(cooling is not None and u is cooling)


def _dual_summer_conditioning_unit(logical_mode):
    """Return the fixed refrigeration unit for logical cool or dry."""
    mode = str(logical_mode or "").strip().lower()
    if mode == "cool":
        return _dual_cooling_unit()
    if mode == "dry":
        return _dual_drying_unit()
    return None


def _dual_sanitize_loaded_summer_role_desires():
    """Canonicalize persisted summer roles, 16 C targets and lowMedium fan."""
    corrected = []
    for u in DAIKINS:
        name = _unit_name(u)
        desired = _desired_units.get(name) or {}
        mode = str(desired.get("mode") or "off").strip().lower()
        if mode == "auto_fan":
            # Releases before 2026-08-10-auto-fan-controller-logic stored
            # auto_fan as a physical HVAC intent. Never replay that obsolete
            # command after restart; the normal auto supervisor will rebuild
            # the selected heat/cool/dry/off intent on its first pass.
            _desired_update_unit(
                u,
                active=False,
                mode="off",
                target=None,
                demand_option=None,
                quiet=None,
                fan_mode=str(globals().get("DUAL_SHARED_FAN_MODE", "lowMedium")),
                reason="loaded_physical_auto_fan_rejected",
                force_stop=False,
            )
            corrected.append(name)
            continue
        if not bool(desired.get("active")) or mode not in ("cool", "dry"):
            continue
        if u is not _dual_summer_conditioning_unit(mode):
            _desired_update_unit(
                u,
                active=False,
                mode="off",
                target=None,
                fan_mode=_dual_desired_fan_mode(u, "off"),
                reason="loaded_%s_role_rejected" % mode,
                force_stop=False,
            )
            corrected.append(name)
            continue
        expected_target = _dual_climate_target_temperature(mode)
        expected_fan = _dual_desired_fan_mode(u, mode)
        target_ok = (
            expected_target is not None and
            _command_matches("target", desired.get("target"), expected_target)
        )
        fan_ok = (
            _dual_fan_mode_semantic(desired.get("fan_mode")) ==
            _dual_fan_mode_semantic(expected_fan)
        )
        if not target_ok or not fan_ok:
            _desired_update_unit(
                u,
                active=True,
                mode=mode,
                target=(
                    round(float(expected_target), 1)
                    if expected_target is not None else None
                ),
                fan_mode=expected_fan,
                reason="loaded_%s_parameters_canonicalized" % mode,
                force_stop=False,
            )
            corrected.append(name)
    return corrected


def _dual_physical_hvac_mode(logical_mode, u=None):
    """Map logical control mode to the mode commanded to the Daikin.

    Temperature-driven cooling remains a separate logical mode so it keeps its
    temperature thermostat, PI controller, Demand limits and learning context.
    Temperature cooling uses physical ``cool`` and humidity drying uses native
    physical ``dry`` at 16 C on their configured Daikin2 conditioning unit.
    They remain separate
    logical modes because cooling is temperature-controlled while drying is
    humidity-controlled. Both use fan lowMedium. The distinct Daikin1 reheat unit
    remains available for physical heat and can never be allocated to cooling
    or drying.
    """
    mode = str(logical_mode or "").strip().lower()
    if mode == "off_fan":
        return str(globals().get("DUAL_COIL_DRY_HVAC_MODE", "fan_only")).strip().lower()
    if mode == "auto_fan":
        selected = str(_dual_auto_selected_mode or "off").strip().lower()
        return _dual_physical_hvac_mode(selected, u)
    if mode == "cool":
        if u is not None and not _dual_is_cooling_unit(u):
            return "off"
        return "cool"
    if mode == "dry":
        if u is not None and not _dual_is_drying_unit(u):
            return "off"
        return str(globals().get("DUAL_DRY_HVAC_MODE", "dry")).strip().lower()
    return mode


def _dual_logical_mode_from_physical(actual_mode, preferred_mode=None):
    """Recover the logical refrigeration mode from the physical Daikin mode."""
    actual = str(actual_mode or "off").strip().lower()
    preferred = str(preferred_mode or "off").strip().lower()
    if actual == "dry" and preferred == "cool":
        return "cool"
    if actual == str(globals().get("DUAL_DRY_HVAC_MODE", "dry")).lower() and preferred == "dry":
        return "dry"
    return actual


def _dual_fan_modes(u, cache=None):
    cache = cache or _TickCache()
    attrs = cache.getattr((u or {}).get("CLIMATE"), {}) or {}
    raw_modes = attrs.get("fan_modes") or []
    if not isinstance(raw_modes, (list, tuple)):
        return []
    return [str(value).strip() for value in raw_modes if str(value).strip()]


def _dual_fan_mode_semantic(value):
    """Normalize common HA/Faikin fan labels to one physical fan level."""
    token = str(value or "").strip().lower()
    compact = "".join([
        char for char in token
        if char not in (" ", "_", "-", "/")
    ])
    aliases = {
        "a": "auto",
        "auto": "auto",
        "automatic": "auto",
        "q": "night",
        "quiet": "night",
        "night": "night",
        "1": "level1",
        "15": "level1",
        "low": "level1",
        "2": "level2",
        "25": "level2",
        "lowmedium": "level2",
        "mediumlow": "level2",
        "3": "level3",
        "35": "level3",
        "medium": "level3",
        "4": "level4",
        "45": "level4",
        "mediumhigh": "level4",
        "highmedium": "level4",
        "5": "level5",
        "55": "level5",
        "high": "level5",
    }
    return aliases.get(compact)


def _dual_resolve_fan_mode(u, requested, cache=None):
    """Return only an exact HA-advertised option; never return a raw guess."""
    if requested is None:
        return None
    wanted = str(requested).strip()
    if not wanted:
        return None
    modes = _dual_fan_modes(u, cache)
    if not modes:
        return None

    # Prefer an exact case-insensitive match and preserve HA's exact spelling.
    for option in modes:
        if option.lower() == wanted.lower():
            return option

    wanted_semantic = _dual_fan_mode_semantic(wanted)
    if wanted_semantic is None:
        return None
    for option in modes:
        if _dual_fan_mode_semantic(option) == wanted_semantic:
            return option
    return None


def _dual_desired_fan_mode(u, logical_mode):
    mode = str(logical_mode or "").strip().lower()
    # Heating must take precedence over the auto + fan lowMedium override.
    # This also covers the coordinated dry-mode reheat unit.
    if mode == "heat":
        return str(globals().get("DUAL_HEAT_FAN_MODE", "mediumHigh"))
    if mode == "off_fan" or bool(_dual_auto_fan_override_active):
        return str(globals().get("DUAL_SHARED_FAN_MODE", "lowMedium"))
    if mode == "cool" and _dual_is_cooling_unit(u):
        return str(globals().get("DUAL_COOL_FAN_MODE", "lowMedium"))
    if mode == "dry" and _dual_is_drying_unit(u):
        return str(globals().get("DUAL_DRY_FAN_MODE", "lowMedium"))
    return str(globals().get("DUAL_DEFAULT_FAN_MODE", "auto"))


def _dual_apply_auto_fan_override():
    """Apply auto + fan speeds without overriding active heating."""
    if not bool(_dual_auto_fan_override_active):
        return
    for u in DAIKINS:
        desired = _desired_units.get(_unit_name(u)) or {}
        wanted = _dual_desired_fan_mode(u, desired.get("mode"))
        _desired_update_unit(
            u,
            fan_mode=wanted,
            reason=(
                "auto_fan_mediumhigh_heat"
                if str(desired.get("mode") or "").lower() == "heat"
                else "auto_fan_lowmedium_override"
            ),
        )


def _unit_hvac_state(u, cache=None):
    """Return normalized climate mode/action plus explicit running evidence."""
    cache = cache or _TickCache()
    climate_ent = u.get("CLIMATE")
    actual_mode = "unknown"
    action = "unknown"
    if climate_ent:
        actual_mode = cache.get_str(climate_ent, default="unknown").strip().lower()
        attrs = cache.getattr(climate_ent, {}) or {}
        action = str(attrs.get("hvac_action") or "unknown").strip().lower()

    running = None
    running_ent = u.get("HEATING_ENTITY") or u.get("RUNNING_ENTITY")
    if running_ent:
        raw = cache.get(running_ent, None)
        if raw is not None:
            s = str(raw).strip().lower()
            if s in ("on", "heat", "heating", "cool", "cooling", "dry", "drying", "true", "yes", "active", "running"):
                running = True
            elif s in ("off", "idle", "false", "no", "inactive", "stopped", "unavailable", "unknown"):
                running = False
            else:
                try:
                    value = float(raw)
                    threshold = float(u.get("HEATING_THRESHOLD", u.get("RUNNING_THRESHOLD", 0.0)))
                    if isfinite(value) and isfinite(threshold):
                        running = value > threshold
                except Exception:
                    pass

    if running is None:
        if action in ("heating", "cooling", "drying", "fan"):
            running = True
        elif action in ("idle", "off"):
            running = False

    return {
        "mode": actual_mode,
        "action": action,
        "running": running,
    }


def _unit_reported_heating(u, cache):
    """Return heat-production evidence, including the dry-mode reheat unit."""
    requested, effective, _summer, _reason = _effective_hvac_mode(cache)
    coordinated_reheat = _dual_is_dry_reheat_assist(u)
    if not (effective == "heat" or coordinated_reheat):
        return False
    info = _unit_hvac_state(u, cache)
    actual_mode = info.get("mode")
    action = info.get("action")
    if actual_mode in ("cool", "dry", "fan_only", "off"):
        return False
    if action == "heating":
        return True
    if action in ("idle", "off", "cooling", "drying", "fan"):
        return False
    if info.get("running") is not None:
        return bool(info.get("running")) and actual_mode in ("heat", "unknown")
    if actual_mode == "heat":
        return True
    return None


_EXPLICIT_DEFROST_TRUE_STATES = (
    "on", "true", "yes", "defrost", "defrosting", "active",
)
_EXPLICIT_DEFROST_FALSE_STATES = (
    "off", "false", "no", "idle", "inactive",
)
_EXPLICIT_DEFROST_STATES = (
    "on", "true", "yes", "defrost", "defrosting", "active",
    "off", "false", "no", "idle", "inactive",
)


def _explicit_defrost_info(u, cache):
    ent = u.get("DEFROST_ENTITY")
    if not ent:
        return None
    info = _sensor_freshness(
        "defrost_%s" % str(u.get("name", "unit")), ent,
        globals().get("DUAL_LIQUID_MAX_AGE_S", 180.0), cache, False, True,
        _EXPLICIT_DEFROST_STATES,
    )
    interpreted = None
    if info.get("usable_for_control"):
        value = str(info.get("value")).strip().lower()
        interpreted = value in _EXPLICIT_DEFROST_TRUE_STATES
    info["state_recognized"] = bool(interpreted is not None)
    info["interpreted_defrosting"] = interpreted
    return info


def _read_defrosting(u, cache=None):
    """Return the single authoritative, report-driven detector result."""
    result = _defrost_detector_update(u, cache=cache)
    liquid = result.get("liquid")
    if liquid is None or not isfinite(liquid):
        liquid = 100.0
    return result.get("defrosting"), float(liquid)


def _defrost_report_token(entity_id):
    if not entity_id:
        return None
    ts = _entity_reported_timestamp(entity_id)
    if ts is not None:
        source = _sensor_timestamp_sources.get(str(entity_id), "timestamp")
        return "%s:%.6f" % (str(source), float(ts))
    try:
        return "value:%s" % str(state.get(entity_id))
    except Exception:
        return None


def _defrost_detector_update(u, cache=None):
    """Update one detector only when a distinct sensor report is observed.

    A recognized explicit state is authoritative regardless of age. Liquid
    fallback retains freshness, hysteresis, and a distinct-report debounce.
    Merely evaluating this function again cannot advance the debounce.
    """
    cache = cache or _TickCache()
    name = _unit_name(u)
    detector = _defrost_detectors.get(name)
    if not isinstance(detector, dict):
        detector = {
            "initialized": False,
            "defrosting": None,
            "source": "none",
            "low_count": 0,
            "explicit_token": None,
            "liquid_token": None,
            "last_transition_at": 0.0,
        }
        _defrost_detectors[name] = detector

    _requested, effective, _summer, _reason = _effective_hvac_mode(cache)
    hvac = _unit_hvac_state(u, cache)
    actual_mode = hvac.get("mode")
    heat_mode = bool(
        (effective == "heat" or _dual_is_dry_reheat_assist(u)) and
        actual_mode not in ("cool", "dry", "fan_only", "off")
    )
    liquid_ent = u.get("LIQUID")
    liquid_info = _sensor_freshness(
        "liquid_%s" % name, liquid_ent,
        globals().get("DUAL_LIQUID_MAX_AGE_S", 180.0), cache, True,
    ) if liquid_ent else None
    liquid = (
        liquid_info.get("value")
        if liquid_info and liquid_info.get("fresh") else float("nan")
    )
    liquid_valid = liquid is not None and isfinite(liquid)

    old_state = detector.get("defrosting")
    if not heat_mode:
        detector["initialized"] = True
        detector["defrosting"] = False
        detector["source"] = "mode_gate"
        detector["low_count"] = 0
    else:
        explicit_info = _explicit_defrost_info(u, cache)
        explicit_valid = bool(
            explicit_info and explicit_info.get("state_recognized")
        )
        if explicit_valid:
            explicit_state = bool(
                explicit_info.get("interpreted_defrosting")
            )
            detector["explicit_token"] = _defrost_report_token(
                u.get("DEFROST_ENTITY")
            )
            detector["initialized"] = True
            detector["defrosting"] = explicit_state
            detector["source"] = "explicit"
            detector["low_count"] = 0
        elif liquid_valid:
            token = _defrost_report_token(liquid_ent)
            is_new_report = (
                not bool(detector.get("initialized")) or
                token != detector.get("liquid_token")
            )
            if is_new_report:
                detector["liquid_token"] = token
                required = max(
                    1, int(globals().get("DEFROST_DEBOUNCE_SAMPLES", 1))
                )
                exit_margin = max(
                    0.0,
                    float(globals().get("DEFROST_EXIT_HYSTERESIS_C", 2.0)),
                )
                threshold = float(DEFROST_LIQUID_THRESHOLD)
                if detector.get("defrosting") is True:
                    threshold += exit_margin
                low = float(liquid) < threshold
                if low:
                    detector["low_count"] = (
                        int(detector.get("low_count") or 0) + 1
                    )
                    if detector["low_count"] >= required:
                        detector["defrosting"] = True
                else:
                    detector["low_count"] = 0
                    detector["defrosting"] = False
                detector["initialized"] = True
            detector["source"] = "liquid"
        else:
            detector["initialized"] = True
            detector["defrosting"] = None
            detector["source"] = "unavailable"

    new_state = detector.get("defrosting")
    if old_state is not new_state and old_state != new_state:
        detector["last_transition_at"] = time.time()
    detector["liquid"] = float(liquid) if liquid_valid else None
    detector["liquid_fresh"] = bool(
        liquid_info and liquid_info.get("fresh")
    )
    detector["heat_mode"] = heat_mode
    detector["actual_mode"] = actual_mode
    return dict(detector)


def _system_defrost_snapshot():
    """Read the authoritative per-unit detector without evaluation debounce."""
    cache = _TickCache()
    _requested, effective_mode, _summer, _reason = _effective_hvac_mode(cache)
    snap = {}

    for u in DAIKINS:
        name = _unit_name(u)
        info = _unit_hvac_state(u, cache)
        detector = _defrost_detector_update(u, cache=cache)
        defrosting = detector.get("defrosting")
        heat_mode = bool(detector.get("heat_mode"))
        reported_heating = _unit_reported_heating(u, cache) if heat_mode else False
        heating = bool(reported_heating)
        if defrosting is True and heat_mode:
            # During defrost hvac_action may be idle; physical heat mode and
            # the detector preserve participation evidence across reload.
            heating = True

        snap[name] = {
            "unit": u,
            "liquid": detector.get("liquid"),
            "defrosting": defrosting,
            "heating": heating,
            "running": bool(info.get("running")) if info.get("running") is not None else heating,
            "mode": info.get("mode"),
            "hvac_action": info.get("action"),
            "system_mode": effective_mode,
            "detector_source": detector.get("source"),
            "distinct_low_reports": int(detector.get("low_count") or 0),
            "last_transition_at": detector.get("last_transition_at"),
        }
    return snap


def _start_post_defrost_hold(u, now=None):
    """Start the configured dual-mode recovery hold; zero minutes disables it."""
    now = time.time() if now is None else float(now)
    name = _unit_name(u)
    hold_s = float(_post_defrost_hold_s(u))
    if hold_s <= 0.0:
        _dual_post_defrost_holds.pop(name, None)
        return None
    cache = _TickCache()
    configured = float(_post_defrost_demand_pct(u))
    current = float(_dual_current_effective_demand(u, cache))
    step_ent = u.get("STEP_LIMIT_HELPER")
    step = (
        cache.get_float(step_ent, default=AUTO_STEP_BASE)
        if step_ent else float(AUTO_STEP_BASE)
    )
    step = _clip(step, float(AUTO_STEP_MIN), float(AUTO_STEP_MAX))
    applied = _clip(configured, current - step, current + step)
    select_ent = u.get("SELECT")
    option = (
        _snap_to_select(select_ent, min(100.0, applied), 0)
        if select_ent else None
    )
    quiet = (
        not (applied > 100.0 + 1e-6)
        if u.get("QUIET_OUTDOOR_SWITCH") else None
    )
    hold = {
        "started_at": now,
        "until": now + hold_s,
        "configured_demand": configured,
        "demand": applied,
        "select_option": option,
    }
    _dual_post_defrost_holds[name] = hold
    _desired_update_unit(
        u,
        demand_option=option,
        quiet=quiet,
        reason="post_defrost_hold",
    )
    return dict(hold)


def _publish_system_defrost_freeze(active, snap, timed_out=False, duration_s=0.0):
    try:
        defrosting_units = sorted([
            name for name, item in snap.items()
            if item.get("defrosting") is True
        ])
        state.set(
            SYSTEM_DEFROST_FREEZE_ENTITY,
            value="on" if active else "off",
            active=bool(active),
            participating_units=sorted(list(_system_defrost_freeze_units)),
            defrosting_units=defrosting_units,
            defrosted_units=sorted(list(_system_defrosted_units)),
            held_select=dict(_system_frozen_select),
            held_quiet=dict(_system_frozen_quiet),
            started_at=round(float(_system_defrost_freeze_started), 1),
            duration_s=round(float(max(0.0, duration_s)), 1),
            timed_out=bool(timed_out),
            post_defrost_holds=dict(_dual_post_defrost_holds),
            system_mode=next(iter(snap.values())).get("system_mode") if snap else _requested_hvac_mode(),
            mode_gated=True,
        )
    except Exception as e:
        log.debug("Daikin ML: failed to publish system defrost freeze state: %s", e)


def _enforce_system_frozen_demands():
    """Re-apply captured demand and Quiet states to every frozen unit."""
    for u in DAIKINS:
        name = u.get("name", "daikin?")
        if name not in _system_defrost_freeze_units:
            continue

        select_ent = u.get("SELECT")
        held_select = _system_frozen_select.get(name)
        if select_ent and held_select:
            _set_select_if_needed(select_ent, held_select)

        quiet_ent = u.get("QUIET_OUTDOOR_SWITCH")
        held_quiet = _system_frozen_quiet.get(name)
        if quiet_ent and held_quiet is not None:
            _set_switch_if_needed(quiet_ent, bool(held_quiet))


def _update_system_defrost_freeze():
    """Update and enforce the multi-unit defrost demand freeze.

    A freeze starts only on a real false->true defrost transition. With two or
    more configured units, at least two must have been heating on the previous
    monitor tick. Single-unit fallback freezes its own validated defrost, which
    preserves the original per-unit hold behavior. This avoids treating a cold,
    inactive outdoor unit as defrost merely because liquid is below 20 C.

    Returns True while all per-unit demand control must remain paused.
    """
    global _system_defrost_freeze_active, _system_defrost_freeze_started
    global _system_defrost_freeze_units, _system_defrosted_units
    global _system_frozen_select, _system_frozen_quiet
    global _system_prev_defrost, _system_prev_heating

    now = time.time()
    snap = _system_defrost_snapshot()

    # Defrost is physically relevant only in heating mode.  Leaving heat
    # releases any freeze immediately and clears all legacy hold/cooldown state
    # so a cooling liquid temperature can never keep heating logic alive.
    system_mode = next(iter(snap.values())).get("system_mode") if snap else _requested_hvac_mode()
    if system_mode != "heat":
        if _system_defrost_freeze_active:
            log.warning("Daikin ML: SYSTEM DEFROST FREEZE cancelled because mode changed to %s", system_mode)
        _system_defrost_freeze_active = False
        _system_defrost_freeze_started = 0.0
        _system_defrost_freeze_units = set()
        _system_defrosted_units = set()
        _system_frozen_select = {}
        _system_frozen_quiet = {}
        _system_prev_defrost = {name: False for name in snap}
        _system_prev_heating = {name: False for name in snap}
        _dual_post_defrost_holds.clear()
        _publish_system_defrost_freeze(False, snap)
        return False

    # Reconstruct an active defrost after startup/reload. Explicit True and a
    # debounced fresh liquid state are already mode-gated, so treating the
    # initial snapshot as a false->true edge is both safe and restart-correct.
    if not _system_prev_defrost:
        _system_prev_defrost = {name: False for name in snap}
        _system_prev_heating = {
            name: bool(
                item.get("heating") or item.get("defrosting") is True
            )
            for name, item in snap.items()
        }

    newly_defrosting = set()
    for name, item in snap.items():
        if item.get("defrosting") is True and _system_prev_defrost.get(name) is False:
            newly_defrosting.add(name)

    if not _system_defrost_freeze_active and newly_defrosting:
        heating_before = set([
            name for name, was_heating in _system_prev_heating.items()
            if bool(was_heating)
        ])

        # With two or more configured units, freeze only if at least two were
        # heating as originally requested. In single-unit fallback, preserve
        # the legacy behavior by freezing that unit's own demand during its
        # validated defrost.
        entered_from_heat = bool(newly_defrosting.intersection(heating_before))
        enough_participants = (
            (len(DAIKINS) == 1 and len(heating_before) >= 1) or
            (len(DAIKINS) >= 2 and len(heating_before) >= 2)
        )
        if enough_participants and entered_from_heat:
            _system_defrost_freeze_active = True
            _system_defrost_freeze_started = now
            _system_defrost_freeze_units = set(heating_before)
            _system_defrosted_units = set(newly_defrosting)
            _system_frozen_select = {}
            _system_frozen_quiet = {}

            capture_cache = _TickCache()
            for name in _system_defrost_freeze_units:
                item = snap.get(name) or {}
                u = item.get("unit")
                if not u:
                    continue

                select_ent = u.get("SELECT")
                current_select = capture_cache.get_str(select_ent, default="")
                if current_select:
                    _system_frozen_select[name] = current_select

                quiet_ent = u.get("QUIET_OUTDOOR_SWITCH")
                if quiet_ent:
                    quiet_state = capture_cache.get_str(quiet_ent, default="").lower()
                    if quiet_state in ("on", "off"):
                        _system_frozen_quiet[name] = (quiet_state == "on")

            # Discard shared-zone rate samples affected by the defrost event.
            _dual_tin_hist[:] = []
            _dual_fast_tin_hist[:] = []

            log.warning(
                "Daikin ML: SYSTEM DEFROST FREEZE started; defrost=%s, units=%s, held_select=%s",
                str(sorted(list(newly_defrosting))),
                str(sorted(list(_system_defrost_freeze_units))),
                str(_system_frozen_select),
            )

    if _system_defrost_freeze_active:
        for name in _system_defrost_freeze_units:
            item = snap.get(name) or {}
            if item.get("defrosting") is True:
                _system_defrosted_units.add(name)

        duration_s = max(0.0, now - float(_system_defrost_freeze_started))
        timed_out = duration_s >= float(SYSTEM_DEFROST_FREEZE_MAX_S)
        any_defrosting = any([
            (snap.get(name) or {}).get("defrosting") is True
            for name in _system_defrost_freeze_units
        ])
        unknown_defrost_state = any([
            (snap.get(name) or {}).get("defrosting") is None
            for name in _system_defrosted_units
        ])

        if (any_defrosting or unknown_defrost_state) and not timed_out:
            _enforce_system_frozen_demands()
            _publish_system_defrost_freeze(True, snap, duration_s=duration_s)

            _system_prev_defrost = {
                name: item.get("defrosting") for name, item in snap.items()
            }
            _system_prev_heating = {
                name: bool(item.get("heating")) for name, item in snap.items()
            }
            return True

        # No participating unit is defrosting any longer (or the safety timeout
        # elapsed). Start the optional per-unit post-defrost recovery hold.
        frozen_units = set(_system_defrost_freeze_units)
        defrosted_units = set(_system_defrosted_units)
        for name in frozen_units:
            if name in defrosted_units:
                unit = (snap.get(name) or {}).get("unit")
                if unit:
                    _start_post_defrost_hold(unit, now)
        _dual_tin_hist[:] = []
        _dual_fast_tin_hist[:] = []

        log.warning(
            "Daikin ML: SYSTEM DEFROST FREEZE ended after %.0fs; defrosted=%s, timed_out=%s",
            float(duration_s), str(sorted(list(defrosted_units))), str(bool(timed_out)),
        )
        _publish_system_defrost_freeze(False, snap, timed_out=timed_out, duration_s=duration_s)

        _system_defrost_freeze_active = False
        _system_defrost_freeze_started = 0.0
        _system_defrost_freeze_units = set()
        _system_defrosted_units = set()
        _system_frozen_select = {}
        _system_frozen_quiet = {}

    _system_prev_defrost = {
        name: item.get("defrosting") for name, item in snap.items()
    }
    _system_prev_heating = {
        name: bool(item.get("heating")) for name, item in snap.items()
    }
    return False


# ============================================================
# SHARED DUAL-ZONE CONTROLLER + SLOW ENERGY OPTIMIZER
# ============================================================

def _timestamp_seconds(value):
    if value is None:
        return None
    try:
        if hasattr(value, "timestamp"):
            return float(value.timestamp())
    except Exception:
        pass
    try:
        result = float(value)
        return result if isfinite(result) else None
    except Exception:
        return None


def _entity_exists(entity_id):
    if not entity_id:
        return False
    try:
        return bool(state.exist(entity_id))
    except Exception:
        try:
            return state.get(entity_id) is not None
        except Exception:
            return False


def _entity_reported_timestamp(entity_id):
    """Return HA's integration-report timestamp, falling back to update time."""
    if not entity_id:
        return None
    for suffix in (".last_reported", ".last_updated"):
        try:
            ts = _timestamp_seconds(state.get(str(entity_id) + suffix))
            if ts is not None:
                _sensor_timestamp_sources[str(entity_id)] = suffix[1:]
                return ts
        except Exception:
            pass
    _sensor_timestamp_sources.pop(str(entity_id), None)
    return None


def _sensor_report_monitored_entities():
    """Return entities whose reports can change controller health or safety.

    Home Assistant updates ``last_reported`` even when an integration publishes
    exactly the same state and attributes. Monitoring that timestamp avoids
    treating a stable temperature value as a sensor that has stopped reporting.
    Optional power/COP inputs are intentionally excluded: their freshness is
    diagnostic/optimizer-only and never activates the anomaly command hold.
    """
    entities = []

    def _append(entity_id):
        if entity_id and isinstance(entity_id, str) and entity_id not in entities:
            entities.append(entity_id)

    indoor_entities = list(globals().get("DUAL_ZONE_INDOOR_SENSORS", []) or [])
    if not indoor_entities:
        primary = _dual_primary_unit()
        if primary:
            _append(primary.get("INDOOR"))
            _append(primary.get("INDOOR2"))
    else:
        for entity_id in indoor_entities:
            _append(entity_id)

    _append(globals().get("DUAL_HUMIDITY_SENSOR"))
    primary = _dual_primary_unit()
    if primary:
        _append(primary.get("OUTDOOR"))

    for unit in DAIKINS:
        _append(unit.get("CLIMATE"))
        _append(unit.get("SEPARATE_INDOOR"))
        _append(unit.get("DEFROST_ENTITY"))
        if not unit.get("DEFROST_ENTITY"):
            _append(unit.get("LIQUID"))
    return entities


def _sensor_report_poll(prime_only=False):
    """Detect new HA reports, including unchanged-value sensor reports.

    This is deliberately a timestamp poll instead of a Pyscript state trigger:
    Home Assistant emits unchanged writes as the high-volume ``state_reported``
    event, while ordinary state triggers only see value/attribute changes.
    Pyscript cannot safely subscribe to that event with Home Assistant's
    required entity-level event filter. Polling the small configured entity set
    gives equivalent report detection without subscribing to the global event.
    """
    global _sensor_report_sequence, _sensor_report_last_entities
    global _sensor_report_last_at
    changed = []
    newest_at = 0.0
    for entity_id in _sensor_report_monitored_entities():
        reported_at = _entity_reported_timestamp(entity_id)
        if reported_at is None:
            continue
        had_previous = entity_id in _sensor_report_tokens
        previous = _sensor_report_tokens.get(entity_id)
        _sensor_report_tokens[entity_id] = float(reported_at)
        if prime_only:
            continue
        if (not had_previous) or float(reported_at) > float(previous) + 1e-6:
            changed.append(entity_id)
            newest_at = max(newest_at, float(reported_at))

    if changed:
        _sensor_report_sequence += 1
        _sensor_report_last_entities = list(changed)
        _sensor_report_last_at = float(newest_at or time.time())
    return changed


def _sensor_freshness(
    role, entity_id, max_age_s, cache=None, numeric=True,
    allow_stale_value=False, accepted_text_values=None,
):
    cache = cache or _TickCache()
    now = time.time()
    raw = cache.get(entity_id, None) if entity_id else None
    text = str(raw).strip().lower() if raw is not None else "unavailable"
    available = raw is not None and text not in ("unknown", "unavailable", "none", "")
    value = None
    valid = bool(available)
    if numeric:
        try:
            value = float(raw)
            valid = bool(available and isfinite(value))
        except Exception:
            valid = False
            value = None
    else:
        value = str(raw) if available else None
        if valid and accepted_text_values is not None:
            valid = (
                str(value).strip().lower() in accepted_text_values
            )
    reported_at = _entity_reported_timestamp(entity_id)
    age_s = max(0.0, now - float(reported_at)) if reported_at is not None else None
    freshness_known = age_s is not None
    stale = bool(valid and freshness_known and age_s > float(max_age_s))
    fresh = bool(valid and not stale)
    usable_for_control = bool(
        valid and (fresh or bool(allow_stale_value))
    )
    if usable_for_control:
        _sensor_last_good[str(entity_id)] = {
            "value": value,
            "observed_at": now,
            "reported_at": reported_at,
        }
    info = {
        "role": str(role),
        "entity_id": entity_id,
        "value": value,
        "valid": bool(valid),
        "fresh": bool(fresh),
        "stale": bool(stale),
        "usable_for_control": bool(usable_for_control),
        "stale_is_warning_only": bool(
            allow_stale_value and valid and stale
        ),
        "freshness_known": bool(freshness_known),
        "timestamp_source": _sensor_timestamp_sources.get(str(entity_id)),
        "reported_at": round(float(reported_at), 3) if reported_at is not None else None,
        "age_seconds": round(float(age_s), 1) if age_s is not None else None,
        "max_age_seconds": float(max_age_s),
        "last_good": _sensor_last_good.get(str(entity_id)),
    }
    _sensor_status[str(role) + "|" + str(entity_id)] = info
    return info


def _critical_command_failures():
    critical = []
    noncritical = []
    for cmd in _command_failures.values():
        kind = str(cmd.get("kind"))
        if kind in ("off", "mode", "target", "select"):
            critical.append(dict(cmd))
        else:
            noncritical.append(dict(cmd))
    return critical, noncritical


def _safety_evaluate(cache=None):
    """Evaluate command holds separately from feature-only degradation."""
    global _safety_health, _safety_reasons, _safety_degraded_active, _safety_valid_since
    global _safety_valid_readings, _safety_critical_active, _safety_recovery_active
    global _safety_recovery_remaining_s, _safety_original_trigger
    global _safety_original_trigger_at, _safety_last_critical_trigger
    global _safety_last_critical_trigger_at, _safety_feature_degradations
    global _safety_diagnostic_warnings
    global _sensor_status
    cache = cache or _TickCache()
    now = time.time()
    _sensor_status = {}
    feature_reasons = []
    diagnostic_warnings = []
    hold_reasons = []
    fault_reasons = []

    indoor_entities = list(globals().get("DUAL_ZONE_INDOOR_SENSORS", []) or [])
    if not indoor_entities:
        primary = _dual_primary_unit()
        if primary and primary.get("INDOOR"):
            indoor_entities.append(primary.get("INDOOR"))
        if primary and primary.get("INDOOR2"):
            indoor_entities.append(primary.get("INDOOR2"))
    indoor_ok = False
    for ent in indoor_entities:
        info = _sensor_freshness(
            "indoor", ent,
            globals().get("DUAL_INDOOR_MAX_AGE_S", 300.0),
            cache, True, True,
        )
        if info.get("usable_for_control"):
            indoor_ok = True
        if info.get("stale_is_warning_only"):
            warning = "indoor_sensor_report_stale_value_usable"
            if warning not in diagnostic_warnings:
                diagnostic_warnings.append(warning)
    if not indoor_ok:
        hold_reasons.append("indoor_sensor_invalid")

    humidity_ent = globals().get("DUAL_HUMIDITY_SENSOR", "")
    humidity_info = _sensor_freshness(
        "humidity", humidity_ent, globals().get("DUAL_HUMIDITY_MAX_AGE_S", 600.0), cache, True
    ) if humidity_ent else None
    requested = _requested_hvac_mode(cache)
    humidity_needed = bool(
        requested == "dry" or
        (_dual_zone_active and str(_dual_active_mode) == "dry") or
        (requested in ("auto", "auto_fan") and str(_dual_auto_selected_mode) == "dry")
    )
    if humidity_needed and (humidity_info is None or not humidity_info.get("fresh")):
        # Active drying holds its current physical state until humidity is
        # fresh again. Heat and temperature-driven cool remain available
        # because logical cool is not humidity-driven.
        feature_reasons.append("humidity_driven_drying_unavailable")

    primary = _dual_primary_unit()
    outdoor_ent = primary.get("OUTDOOR") if primary else None
    if outdoor_ent:
        info = _sensor_freshness("outdoor", outdoor_ent, globals().get("DUAL_OUTDOOR_MAX_AGE_S", 900.0), cache, True)
        if not info.get("fresh"):
            feature_reasons.append("outdoor_sensor_invalid_or_stale_conservative_control")

    for u in DAIKINS:
        name = str(u.get("name", "unit"))
        climate_ent = u.get("CLIMATE")
        if not climate_ent or not _entity_exists(climate_ent):
            fault_reasons.append("%s_climate_missing" % name)
        else:
            c_info = _sensor_freshness("climate_%s" % name, climate_ent, 600.0, cache, False)
            if not c_info.get("valid"):
                fault_reasons.append("%s_climate_unavailable" % name)
        liquid_ent = u.get("LIQUID")
        explicit_defrost = u.get("DEFROST_ENTITY")
        if explicit_defrost:
            d_info = _explicit_defrost_info(u, cache)
            if d_info and d_info.get("usable_for_control"):
                if d_info.get("stale_is_warning_only"):
                    diagnostic_warnings.append(
                        "%s_explicit_defrost_report_stale_state_usable" % name
                    )
            else:
                # Missing, unavailable, unknown and unrecognized explicit
                # states fall back to the age-limited liquid detector.
                l_info = _sensor_freshness(
                    "liquid_%s" % name, liquid_ent,
                    globals().get("DUAL_LIQUID_MAX_AGE_S", 180.0), cache, True,
                ) if liquid_ent else None
                if l_info and l_info.get("fresh"):
                    diagnostic_warnings.append(
                        "%s_explicit_defrost_invalid_liquid_fallback_active" % name
                    )
                else:
                    feature_reasons.append(
                        "%s_controller_defrost_detection_unavailable" % name
                    )
        elif liquid_ent:
            l_info = _sensor_freshness(
                "liquid_%s" % name, liquid_ent, globals().get("DUAL_LIQUID_MAX_AGE_S", 180.0), cache, True
            )
            if not l_info.get("fresh"):
                feature_reasons.append("%s_controller_defrost_detection_unavailable" % name)
        power_ent = u.get("POWER_SENSOR")
        if power_ent:
            _sensor_freshness(
                "power_%s" % name, power_ent, globals().get("DUAL_POWER_MAX_AGE_S", 120.0), cache, True
            )
        cop_ent = u.get("COP_SENSOR")
        if cop_ent:
            # COP is an optional, optimizer-only measurement. A missing or
            # stale value falls back to the electrical-energy score and must
            # never stop comfort control or mark controller health degraded.
            _sensor_freshness(
                "cop_optional_%s" % name, cop_ent,
                globals().get("DUAL_COP_MAX_AGE_S", 120.0), cache, True,
            )

    critical_cmds, noncritical_cmds = _critical_command_failures()
    if critical_cmds:
        # A failed physical command is diagnostic and unit-specific. It must
        # never cascade into a whole-system shutdown or global command hold.
        feature_reasons.append("physical_command_failed_no_shutdown")
    if noncritical_cmds:
        feature_reasons.append("demand_or_quiet_command_failed")
    if _controller_shadow_enabled(cache):
        feature_reasons.append("controller_shadow_mode")

    # Indoor/climate anomalies freeze new commands and preserve the equipment's
    # current operating state. They never request HVAC off. Preserve the first
    # trigger throughout recovery so the wait state never erases root cause.
    current_critical = list(hold_reasons) + list(fault_reasons)
    recovery_s = max(
        0.0,
        float(globals().get(
            "DUAL_CRITICAL_RECOVERY_SECONDS",
            float(globals().get("DUAL_DEGRADED_RECOVERY_MINUTES", 1.0)) * 60.0,
        )),
    )
    readings_required = max(
        1, int(globals().get("DUAL_CRITICAL_RECOVERY_VALID_READINGS", 4))
    )
    if current_critical:
        _cancel_pending_commands_for_anomaly(
            "critical_anomaly:" + ",".join([str(v) for v in current_critical])
        )
        if not _safety_critical_active and not _safety_recovery_active:
            _safety_original_trigger = list(current_critical)
            _safety_original_trigger_at = now
        else:
            for reason in current_critical:
                if reason not in _safety_original_trigger:
                    _safety_original_trigger.append(reason)
        _safety_last_critical_trigger = list(_safety_original_trigger)
        _safety_last_critical_trigger_at = float(_safety_original_trigger_at or now)
        _safety_critical_active = True
        _safety_recovery_active = False
        _safety_valid_since = 0.0
        _safety_valid_readings = 0
        _safety_recovery_remaining_s = recovery_s
        _safety_degraded_active = True
        _safety_health = "fault" if fault_reasons else "degraded"
        _safety_reasons = list(current_critical) + list(feature_reasons)
    else:
        if _safety_critical_active:
            _safety_critical_active = False
            _safety_recovery_active = True
            _safety_valid_since = now
            _safety_valid_readings = 1
        elif _safety_recovery_active:
            _safety_valid_readings += 1

        if _safety_recovery_active:
            elapsed = max(0.0, now - float(_safety_valid_since or now))
            _safety_recovery_remaining_s = max(0.0, recovery_s - elapsed)
            recovery_complete = bool(
                elapsed >= recovery_s and
                _safety_valid_readings >= readings_required
            )
            if recovery_complete:
                _safety_recovery_active = False
                _safety_valid_since = 0.0
                _safety_valid_readings = 0
                _safety_recovery_remaining_s = 0.0
                _safety_original_trigger = []
                _safety_original_trigger_at = 0.0
                _safety_degraded_active = False
                _safety_health = "degraded" if feature_reasons else "ok"
                _safety_reasons = list(feature_reasons)
            else:
                _safety_degraded_active = True
                _safety_health = "degraded"
                _safety_reasons = (
                    list(_safety_original_trigger) +
                    ["stable_data_recovery_wait"] +
                    list(feature_reasons)
                )
        else:
            _safety_recovery_remaining_s = 0.0
            _safety_degraded_active = False
            _safety_health = "degraded" if feature_reasons else "ok"
            _safety_reasons = list(feature_reasons)

    _safety_feature_degradations = list(feature_reasons)
    _safety_diagnostic_warnings = list(diagnostic_warnings)
    return {
        "health": _safety_health,
        "reasons": list(_safety_reasons),
        "degraded_active": bool(_safety_degraded_active),
        "control_hold_active": bool(_safety_degraded_active or _safety_health == "fault"),
        "commands_frozen": bool(_safety_degraded_active or _safety_health == "fault"),
        "shutdown_required": False,
        "anomaly_shutdown_policy": "never",
        "anomaly_shutdown_suppressed": bool(
            _safety_degraded_active or _safety_health == "fault"
        ),
        "feature_degradations": list(_safety_feature_degradations),
        "diagnostic_warnings": list(_safety_diagnostic_warnings),
        "original_trigger": list(_safety_original_trigger),
        "original_trigger_at": (
            round(float(_safety_original_trigger_at), 3)
            if _safety_original_trigger_at else None
        ),
        "last_critical_trigger": list(_safety_last_critical_trigger),
        "last_critical_trigger_at": (
            round(float(_safety_last_critical_trigger_at), 3)
            if _safety_last_critical_trigger_at else None
        ),
        "recovery_remaining_seconds": round(float(_safety_recovery_remaining_s), 1),
        "recovery_valid_readings": int(_safety_valid_readings),
        "recovery_valid_readings_required": int(readings_required),
        "sensor_status": dict(_sensor_status),
        "critical_command_failures": critical_cmds,
        "command_failures": noncritical_cmds,
    }


def _normal_optimizer_inputs_available():
    if _safety_health != "ok" or _controller_shadow_enabled():
        return False
    for key, info in _sensor_status.items():
        if str(info.get("role", "")) == "indoor" and not bool(info.get("fresh")):
            return False
        if str(info.get("role", "")).startswith("optimizer_block"):
            return False
    return True


def _daily_stats_empty(date_text):
    return {
        "date": str(date_text),
        "runtime_minutes": {},
        "starts": {},
        "energy_kwh": 0.0,
        "energy_samples_valid": 0,
        "defrost_minutes": 0.0,
        "comfort_deviation_minutes": 0.0,
        "comfort_degree_minutes": 0.0,
    }


def _today_text(now=None):
    now = time.time() if now is None else float(now)
    try:
        return time.strftime("%Y-%m-%d", time.localtime(now))
    except Exception:
        return "unknown"


def _runtime_payload():
    return {
        "schema": 4,
        "saved_at": time.time(),
        "zone_active": bool(_dual_zone_active),
        "active_mode": str(_dual_active_mode),
        "zone_started_at": float(_dual_zone_started_at or 0.0),
        "zone_stopped_at": float(_dual_zone_stopped_at or 0.0),
        "mode_changed_at": float(_dual_mode_changed_at or 0.0),
        "last_control_mode": str(_dual_last_control_mode or "off"),
        "auto_selected_mode": str(_dual_auto_selected_mode or "off"),
        "dry_reheat_active": bool(_dual_dry_reheat_active),
        "dry_reheat_started_at": float(_dual_dry_reheat_started_at or 0.0),
        "dry_reheat_stopped_at": float(_dual_dry_reheat_stopped_at or 0.0),
        "dry_reheat_integral": float(_dual_dry_reheat_integral or 0.0),
        "dry_reheat_last_demand": _dual_dry_reheat_last_demand,
        "dry_lead_paused": bool(_dual_dry_lead_paused),
        "dry_lead_paused_at": float(_dual_dry_lead_paused_at or 0.0),
        "dry_lead_resumed_at": float(_dual_dry_lead_resumed_at or 0.0),
        "coil_dry_active": bool(_dual_coil_dry_active),
        "coil_dry_started_at": float(_dual_coil_dry_started_at or 0.0),
        "coil_dry_until": float(_dual_coil_dry_until or 0.0),
        "coil_dry_source_mode": _dual_coil_dry_source_mode,
        "coil_dry_units": list(_dual_coil_dry_units),
        "coil_dry_reason": _dual_coil_dry_reason,
        "daily_stats_date": str(_daily_stats_date),
        "daily_stats": dict(_daily_stats),
        "stats_last_ts": float(_stats_last_ts or 0.0),
        "stats_prev_running": dict(_stats_prev_running),
        "stats_prev_power_w": _stats_prev_power_w,
        "unit_runtime": dict(_unit_runtime),
        "desired_units": dict(_desired_units),
        "defrost_detectors": dict(_defrost_detectors),
        "post_defrost_holds": dict(_dual_post_defrost_holds),
        # The replay ring is intentionally memory-only. Persisting a growing
        # 24-hour list on every statistics update would create excessive HA
        # recorder traffic. Safety timers and daily totals remain persistent.
    }


def _runtime_store_save(force=False):
    global _runtime_store_last_payload
    # ``force`` bypasses throttling only. It must never bypass load-before-save.
    if not _runtime_store_loaded:
        return False
    payload = _runtime_payload()
    compare_payload = dict(payload)
    compare_payload.pop("saved_at", None)
    encoded = json.dumps(compare_payload, sort_keys=True, separators=(",", ":"))
    if not force and encoded == _runtime_store_last_payload:
        return False
    entity_id = globals().get("DUAL_RUNTIME_STORE_ENTITY", "pyscript.daikin_dual_runtime_store")
    try:
        state.set(entity_id, value=round(time.time(), 3), **payload)
        _runtime_store_last_payload = encoded
        return True
    except Exception as e:
        log.error("Daikin dual: runtime-state persistence failed: %s", e)
        return False


def _runtime_store_load():
    global _runtime_store_loaded, _runtime_store_last_payload
    global _dual_zone_active, _dual_active_mode, _dual_zone_started_at, _dual_zone_stopped_at
    global _dual_mode_changed_at, _dual_last_control_mode, _dual_auto_selected_mode
    global _dual_dry_reheat_active, _dual_dry_reheat_started_at
    global _dual_dry_reheat_stopped_at, _dual_dry_reheat_integral
    global _dual_dry_reheat_last_demand, _dual_dry_lead_paused
    global _dual_dry_lead_paused_at, _dual_dry_lead_resumed_at
    global _dual_coil_dry_active, _dual_coil_dry_started_at, _dual_coil_dry_until
    global _dual_coil_dry_source_mode, _dual_coil_dry_units, _dual_coil_dry_reason
    global _daily_stats_date, _daily_stats, _stats_last_ts, _stats_prev_running
    global _stats_prev_power_w, _replay_samples, _replay_last_sample_ts
    global _unit_runtime, _desired_units, _desired_generation
    global _entity_generation
    global _defrost_detectors, _dual_post_defrost_holds
    entity_id = globals().get("DUAL_RUNTIME_STORE_ENTITY", "pyscript.daikin_dual_runtime_store")
    try:
        state.persist(entity_id)
    except Exception as e:
        log.error("Daikin dual: state.persist failed for runtime store %s: %s", entity_id, e)
    try:
        attrs = state.getattr(entity_id) or {}
    except Exception:
        attrs = {}
    now = time.time()
    if int(attrs.get("schema") or 0) >= 1:
        _dual_zone_active = bool(attrs.get("zone_active"))
        _dual_active_mode = str(attrs.get("active_mode") or "off")
        _dual_zone_started_at = float(attrs.get("zone_started_at") or 0.0)
        _dual_zone_stopped_at = float(attrs.get("zone_stopped_at") or 0.0)
        _dual_mode_changed_at = float(attrs.get("mode_changed_at") or 0.0)
        _dual_last_control_mode = str(attrs.get("last_control_mode") or "off")
        _dual_auto_selected_mode = str(attrs.get("auto_selected_mode") or "off")
        _dual_dry_reheat_active = bool(attrs.get("dry_reheat_active"))
        _dual_dry_reheat_started_at = float(attrs.get("dry_reheat_started_at") or 0.0)
        _dual_dry_reheat_stopped_at = float(attrs.get("dry_reheat_stopped_at") or 0.0)
        _dual_dry_reheat_integral = float(attrs.get("dry_reheat_integral") or 0.0)
        _dual_dry_reheat_last_demand = attrs.get("dry_reheat_last_demand")
        _dual_dry_lead_paused = bool(attrs.get("dry_lead_paused"))
        _dual_dry_lead_paused_at = float(attrs.get("dry_lead_paused_at") or 0.0)
        _dual_dry_lead_resumed_at = float(attrs.get("dry_lead_resumed_at") or 0.0)
        _dual_coil_dry_active = bool(attrs.get("coil_dry_active"))
        _dual_coil_dry_started_at = float(attrs.get("coil_dry_started_at") or 0.0)
        _dual_coil_dry_until = float(attrs.get("coil_dry_until") or 0.0)
        _dual_coil_dry_source_mode = attrs.get("coil_dry_source_mode")
        _dual_coil_dry_units = set(attrs.get("coil_dry_units") or [])
        _dual_coil_dry_reason = attrs.get("coil_dry_reason")
        _daily_stats_date = str(attrs.get("daily_stats_date") or "")
        _daily_stats = dict(attrs.get("daily_stats") or {})
        _stats_last_ts = float(attrs.get("stats_last_ts") or 0.0)
        _stats_prev_running = dict(attrs.get("stats_prev_running") or {})
        _stats_prev_power_w = attrs.get("stats_prev_power_w")
        loaded_runtime = attrs.get("unit_runtime")
        if isinstance(loaded_runtime, dict):
            _unit_runtime = {}
            for name, raw in loaded_runtime.items():
                if isinstance(raw, dict):
                    _unit_runtime[str(name)] = dict(raw)
        loaded_desired = attrs.get("desired_units")
        if isinstance(loaded_desired, dict):
            configured = set([_unit_name(u) for u in DAIKINS])
            _desired_units = {}
            for name, raw in loaded_desired.items():
                if str(name) in configured and isinstance(raw, dict):
                    _desired_units[str(name)] = dict(raw)
        _desired_generation = 0
        for raw in _desired_units.values():
            _desired_generation = max(
                _desired_generation, int(raw.get("generation") or 0)
            )
        _entity_generation = {}
        for u in DAIKINS:
            raw = _desired_units.get(_unit_name(u)) or {}
            generation = int(raw.get("generation") or 0)
            for entity_id in _desired_entities(u):
                _entity_generation[entity_id] = generation
        loaded_detectors = attrs.get("defrost_detectors")
        if isinstance(loaded_detectors, dict):
            _defrost_detectors = {}
            for name, raw in loaded_detectors.items():
                if isinstance(raw, dict):
                    _defrost_detectors[str(name)] = dict(raw)
        loaded_holds = attrs.get("post_defrost_holds")
        if isinstance(loaded_holds, dict):
            _dual_post_defrost_holds = {}
            for name, raw in loaded_holds.items():
                if (
                    isinstance(raw, dict) and
                    float(raw.get("until") or 0.0) > now
                ):
                    _dual_post_defrost_holds[str(name)] = dict(raw)
        _replay_samples = []
        _replay_last_sample_ts = 0.0
        # Reject impossible future values while preserving legitimate elapsed
        # timers over a normal restart/reload.
        if _dual_zone_started_at > now + 60.0:
            _dual_zone_started_at = now
        if _dual_zone_stopped_at > now + 60.0:
            _dual_zone_stopped_at = now
        if _dual_dry_reheat_started_at > now + 60.0:
            _dual_dry_reheat_started_at = now
        if _dual_dry_reheat_stopped_at > now + 60.0:
            _dual_dry_reheat_stopped_at = now
        if _dual_dry_lead_paused_at > now + 60.0:
            _dual_dry_lead_paused_at = now
        if _dual_dry_lead_resumed_at > now + 60.0:
            _dual_dry_lead_resumed_at = now
        if _dual_coil_dry_until > now + 24.0 * 3600.0:
            _dual_coil_dry_until = now
        for runtime in _unit_runtime.values():
            for key in ("started_at", "stopped_at", "last_transition_at"):
                try:
                    value = float(runtime.get(key) or 0.0)
                except Exception:
                    value = 0.0
                if value > now + 60.0:
                    runtime[key] = now
    if _daily_stats_date != _today_text(now) or not _daily_stats:
        _daily_stats_date = _today_text(now)
        _daily_stats = _daily_stats_empty(_daily_stats_date)
    _runtime_store_loaded = True
    _dual_sanitize_loaded_summer_role_desires()
    compare_payload = _runtime_payload()
    compare_payload.pop("saved_at", None)
    _runtime_store_last_payload = json.dumps(compare_payload, sort_keys=True, separators=(",", ":"))
    return bool(attrs)


def _publish_daily_stats():
    sensor = globals().get("DUAL_DAILY_STATS_SENSOR", "sensor.daikin_dual_daily_stats")
    try:
        state.set(
            sensor,
            value=round(float(_daily_stats.get("energy_kwh", 0.0)), 3),
            unit_of_measurement="kWh",
            date=_daily_stats.get("date"),
            runtime_minutes=dict(_daily_stats.get("runtime_minutes") or {}),
            starts=dict(_daily_stats.get("starts") or {}),
            defrost_minutes=round(float(_daily_stats.get("defrost_minutes", 0.0)), 1),
            comfort_deviation_minutes=round(float(_daily_stats.get("comfort_deviation_minutes", 0.0)), 1),
            comfort_degree_minutes=round(float(_daily_stats.get("comfort_degree_minutes", 0.0)), 2),
            energy_samples_valid=int(_daily_stats.get("energy_samples_valid", 0)),
        )
    except Exception as e:
        log.debug("Daikin dual: failed to publish daily statistics: %s", e)


def _stats_tick(cache=None, system_freeze_active=False):
    global _daily_stats_date, _daily_stats, _stats_last_ts, _stats_prev_running
    global _stats_prev_power_w, _replay_last_sample_ts, _replay_samples
    global _runtime_stats_save_bucket
    cache = cache or _TickCache()
    now = time.time()
    today = _today_text(now)
    if _daily_stats_date != today or not _daily_stats:
        _daily_stats_date = today
        _daily_stats = _daily_stats_empty(today)
        _stats_last_ts = now
        _stats_prev_power_w = None
    dt_s = max(0.0, now - float(_stats_last_ts or now))
    dt_s = min(dt_s, float(globals().get("DUAL_STATS_MAX_SAMPLE_GAP_S", 120.0)))
    runtime = _daily_stats.setdefault("runtime_minutes", {})
    starts = _daily_stats.setdefault("starts", {})
    for u in DAIKINS:
        name = str(u.get("name", "unit"))
        info = _unit_hvac_state(u, cache)
        running = str(info.get("action")) in ("heating", "cooling", "drying")
        if running:
            runtime[name] = float(runtime.get(name, 0.0)) + dt_s / 60.0
        if running and not bool(_stats_prev_running.get(name, False)):
            starts[name] = int(starts.get(name, 0)) + 1
        _stats_prev_running[name] = bool(running)
    stats_action = _dual_action_by_id(_dual_active_action_id) if _dual_zone_active else None
    power_w, power_valid, _per_unit = _dual_total_power(cache, stats_action)
    if power_valid and power_w is not None and isfinite(power_w):
        if _stats_prev_power_w is None:
            avg_power = float(power_w)
        else:
            avg_power = 0.5 * (float(_stats_prev_power_w) + float(power_w))
        _daily_stats["energy_kwh"] = float(_daily_stats.get("energy_kwh", 0.0)) + avg_power * dt_s / 3600000.0
        _daily_stats["energy_samples_valid"] = int(_daily_stats.get("energy_samples_valid", 0)) + 1
        _stats_prev_power_w = float(power_w)
    else:
        _stats_prev_power_w = None
    if system_freeze_active:
        _daily_stats["defrost_minutes"] = float(_daily_stats.get("defrost_minutes", 0.0)) + dt_s / 60.0
    tin = _dual_zone_temperature(cache)
    sp, _mn, _mx, _db = _dual_effective_setpoint(cache, write_helpers=False)
    if isfinite(tin) and isfinite(sp):
        deviation = abs(float(tin) - float(sp))
        if deviation > float(globals().get("DUAL_COMFORT_DEVIATION_C", 0.40)):
            _daily_stats["comfort_deviation_minutes"] = float(_daily_stats.get("comfort_deviation_minutes", 0.0)) + dt_s / 60.0
            _daily_stats["comfort_degree_minutes"] = float(_daily_stats.get("comfort_degree_minutes", 0.0)) + deviation * dt_s / 60.0

    sample_interval = max(60.0, float(globals().get("DUAL_REPLAY_SAMPLE_INTERVAL_S", 300.0)))
    if now - float(_replay_last_sample_ts or 0.0) >= sample_interval and isfinite(tin) and isfinite(sp):
        humidity = _dual_humidity(cache, filtered=False)
        _replay_samples.append({
            "ts": round(now, 1),
            "indoor": round(float(tin), 3),
            "setpoint": round(float(sp), 3),
            "humidity": round(float(humidity), 3) if isfinite(humidity) else None,
            "summer": bool(_summer_mode_enabled(cache)),
            "actual_mode": str(_dual_active_mode),
            "power_w": round(float(power_w), 1) if power_valid and power_w is not None else None,
        })
        max_samples = int(max(12.0, float(globals().get("DUAL_REPLAY_RETENTION_HOURS", 24.0)) * 3600.0 / sample_interval))
        if len(_replay_samples) > max_samples:
            _replay_samples = _replay_samples[-max_samples:]
        _replay_last_sample_ts = now
    _stats_last_ts = now
    _publish_daily_stats()
    save_bucket = int(now // 300.0)
    if _runtime_stats_save_bucket is None or save_bucket != _runtime_stats_save_bucket:
        _runtime_stats_save_bucket = save_bucket
        _runtime_store_save(force=False)


def _publish_health():
    sensor = globals().get("DUAL_HEALTH_SENSOR", "sensor.daikin_dual_health")
    now = time.time()
    min_on_remaining = 0.0
    min_off_remaining = 0.0
    limits = _dual_supervisor_limits(_TickCache()) if _dual_enabled() else {"min_on_min": 0.0, "min_off_min": 0.0}
    if _dual_zone_active and _dual_zone_started_at:
        min_on_remaining = max(0.0, float(limits.get("min_on_min", 0.0)) * 60.0 - (now - float(_dual_zone_started_at)))
    if not _dual_zone_active and _dual_zone_stopped_at:
        min_off_remaining = max(0.0, float(limits.get("min_off_min", 0.0)) * 60.0 - (now - float(_dual_zone_stopped_at)))
    next_control = (int(now // 300.0) + 1) * 300.0
    outdoor_fallback_active = any([
        str(info.get("role")) == "outdoor" and not bool(info.get("fresh"))
        for info in _sensor_status.values()
    ])
    outdoor_control = (
        _dual_outdoor_temperature(_TickCache(), allow_conservative_fallback=True)
        if _dual_enabled() else float("nan")
    )
    try:
        state.set(
            sensor,
            value=str(_safety_health),
            controller_version=globals().get("DUAL_CONTROLLER_VERSION"),
            reasons=list(_safety_reasons),
            degraded_mode_active=bool(_safety_degraded_active),
            control_hold_active=bool(
                _safety_degraded_active or _safety_health == "fault"
            ),
            commands_frozen=bool(
                _safety_degraded_active or _safety_health == "fault"
            ),
            shutdown_required=False,
            anomaly_shutdown_policy="never",
            anomaly_shutdown_suppressed=bool(
                _safety_degraded_active or _safety_health == "fault"
            ),
            feature_degradations=list(_safety_feature_degradations),
            diagnostic_warnings=list(_safety_diagnostic_warnings),
            original_trigger=list(_safety_original_trigger),
            original_trigger_at=(
                round(float(_safety_original_trigger_at), 3)
                if _safety_original_trigger_at else None
            ),
            last_critical_trigger=list(_safety_last_critical_trigger),
            last_critical_trigger_at=(
                round(float(_safety_last_critical_trigger_at), 3)
                if _safety_last_critical_trigger_at else None
            ),
            recovery_remaining_seconds=round(float(_safety_recovery_remaining_s), 1),
            recovery_valid_readings=int(_safety_valid_readings),
            recovery_valid_readings_required=max(
                1, int(globals().get("DUAL_CRITICAL_RECOVERY_VALID_READINGS", 4))
            ),
            sensor_report_monitor_active=True,
            sensor_report_poll_seconds=float(
                globals().get(
                    "_SENSOR_REPORT_POLL_INTERVAL_S",
                    globals().get("DUAL_SENSOR_REPORT_POLL_SECONDS", 1.0),
                )
            ),
            sensor_report_sequence=int(_sensor_report_sequence),
            last_sensor_report_entities=list(_sensor_report_last_entities),
            last_sensor_report_at=(
                round(float(_sensor_report_last_at), 3)
                if _sensor_report_last_at else None
            ),
            sensor_report_timestamp_sources=dict(_sensor_timestamp_sources),
            optimizer_suspended=not _normal_optimizer_inputs_available(),
            outdoor_fallback_active=bool(outdoor_fallback_active),
            outdoor_control_temperature=(
                round(float(outdoor_control), 2) if isfinite(outdoor_control) else None
            ),
            controller_shadow_mode=_controller_shadow_enabled(),
            sensor_status=dict(_sensor_status),
            last_good_values=dict(_sensor_last_good),
            heartbeat_age_seconds=round(max(0.0, now - float(_heartbeat_last_success)), 1) if _heartbeat_last_success else None,
            last_controller_error=_heartbeat_last_error,
            controller_error_count=int(_controller_error_count),
            command_pending=len(_command_pending),
            command_recovering=_command_recovering_count(),
            command_verifying=_command_verifying_count(),
            command_failures=len(_command_failures),
            command_suppressed_count=int(_command_suppressed_count),
            command_cancelled_count=int(_command_cancelled_count),
            minimum_on_remaining_seconds=round(min_on_remaining, 1),
            minimum_off_remaining_seconds=round(min_off_remaining, 1),
            next_control_epoch=round(next_control, 1),
            next_control_in_seconds=round(max(0.0, next_control - now), 1),
            runtime_store_loaded=bool(_runtime_store_loaded),
        )
    except Exception as e:
        log.debug("Daikin dual: failed to publish health: %s", e)


def _heartbeat_success(started_at, note="controller_tick"):
    global _heartbeat_sequence, _heartbeat_last_success, _heartbeat_last_error
    _heartbeat_sequence += 1
    _heartbeat_last_success = time.time()
    _heartbeat_last_error = None
    sensor = globals().get("DUAL_HEARTBEAT_SENSOR", "sensor.daikin_dual_controller_heartbeat")
    try:
        state.set(
            sensor,
            value=round(float(_heartbeat_last_success), 3),
            sequence=int(_heartbeat_sequence),
            status="ok",
            note=str(note),
            controller_version=globals().get("DUAL_CONTROLLER_VERSION"),
            execution_ms=round(max(0.0, _heartbeat_last_success - float(started_at)) * 1000.0, 1),
            health=str(_safety_health),
        )
    except Exception:
        pass
    _publish_health()


def _heartbeat_error(started_at, error):
    global _heartbeat_last_error, _controller_error_count
    _heartbeat_last_error = str(error)
    _controller_error_count += 1
    sensor = globals().get("DUAL_HEARTBEAT_SENSOR", "sensor.daikin_dual_controller_heartbeat")
    try:
        state.set(
            sensor,
            value=round(float(_heartbeat_last_success or 0.0), 3),
            sequence=int(_heartbeat_sequence),
            status="error",
            error=str(error),
            controller_version=globals().get("DUAL_CONTROLLER_VERSION"),
            execution_ms=round(max(0.0, time.time() - float(started_at)) * 1000.0, 1),
            health="fault",
        )
    except Exception:
        pass
    _publish_health()


def _initialize_diagnostic_entities():
    """Create every diagnostic entity even while the HVAC is idle."""
    entities = [
        (globals().get("DUAL_MODE_STATUS_SENSOR", "sensor.daikin_dual_mode_status"), "startup", {
            "active": False, "fan_only_active": False, "reason": "startup",
            "configured_units": len(DAIKINS), "single_unit_fallback": len(DAIKINS) == 1,
            "coordinated_reheat_drying": False,
            "dry_reheat_active": False,
            "dry_lead_paused": False,
            "dry_coordination": dict(_dual_dry_coordination_status),
        }),
        (globals().get("DUAL_TOTAL_DEMAND_SENSOR", "sensor.daikin_dual_total_demand"), 0.0, {
            "unit_of_measurement": "% points", "operating_mode": "off", "active": False,
            "note": "created_at_startup_before_first_active_control_pass",
        }),
        (globals().get("DUAL_OPTIMIZER_SENSOR", "sensor.daikin_dual_efficiency_optimizer"), "inactive", {
            "optimizer_mode": _dual_optimizer_mode(_TickCache()), "operating_mode": "off",
            "note": "created_at_startup",
        }),
        (globals().get("DUAL_COP_EFFICIENCY_SENSOR", "sensor.daikin_dual_cop_efficiency"), "unavailable", {
            "valid": False, "reason": "startup", "operating_mode": "off",
            "combined_cop": None, "context": None, "configured_units": len(DAIKINS),
            "note": "created_at_startup",
        }),
        (globals().get("DUAL_REPLAY_SENSOR", "sensor.daikin_dual_history_replay"), "not_run", {
            "sample_count": len(_replay_samples), "source": "internal_24h",
        }),
        (globals().get("DUAL_LANDING_STATUS_SENSOR", "sensor.daikin_dual_landing_status"), "inactive", {
            "operating_mode": "off", "reason": "startup",
            "landing_cap": None, "projected_error": None,
            "rapid_approach": False,
            "control_interval_seconds": float(globals().get("DUAL_FAST_TAPER_INTERVAL_S", 60.0)),
            "fast_rate_cph": None, "long_rate_cph": None,
            "rate_source": "none",
            "sustain_probe": _dual_probe_snapshot(),
        }),
        (globals().get("DUAL_COMMISSIONING_SENSOR", "sensor.daikin_dual_commissioning"), "not_run", {
            "issues": [], "warnings": [], "checked_at": None,
        }),
    ]
    for entity_id, value, attrs in entities:
        try:
            state.set(entity_id, value=value, **attrs)
        except Exception:
            pass
    _dual_publish_learning_status("off")
    _dual_publish_fan_calibration()
    _dual_publish_cdp_curve()
    _publish_separate_areas_status(_TickCache(), _separate_area_configuration_issues())
    _publish_command_status()
    _publish_daily_stats()
    _publish_health()


def _dual_enabled():
    """Return True when the shared-zone controller can own one or more units.

    The historical name is retained for compatibility. With one configured
    unit the controller uses a synthetic single-unit allocation. Adding a
    second DAIKINS entry automatically exposes the normal dual-unit policies.
    """
    return bool(globals().get("DUAL_ZONE_ENABLED", False)) and len(DAIKINS) >= 1


def _dual_primary_unit():
    wanted = str(globals().get("DUAL_PRIMARY_UNIT", ""))
    for u in DAIKINS:
        if str(u.get("name")) == wanted:
            return u
    for u in DAIKINS:
        if str(u.get("ROLE", "")).lower() == "lead":
            return u
    return DAIKINS[0] if DAIKINS else None


def _dual_heating_lead_unit(cache=None):
    """Return the UI-selected first heating unit without changing other roles."""
    cache = cache or _TickCache()
    helper = globals().get(
        "DUAL_DAIKIN1_HEATING_LEAD_HELPER",
        "input_boolean.daikin1_heating_lead",
    )
    use_daikin1 = bool(
        helper and cache.get_str(helper, default="off").strip().lower() == "on"
    )
    if use_daikin1:
        for u in DAIKINS:
            if str(u.get("name", "")) == "daikin1":
                return u
    return _dual_primary_unit()


def _dual_coordinated_dry_units():
    """Return the fixed (drying, reheat) pair when mixed control is usable."""
    if not bool(globals().get("DUAL_DRY_REHEAT_ENABLED", True)) or len(DAIKINS) < 2:
        return None, None
    lead = _dual_drying_unit()
    assist = _dual_dry_reheat_unit()
    if not (lead and assist):
        return None, None
    for u in (lead, assist):
        if not (
            u.get("CLIMATE") and u.get("SELECT") and
            bool(u.get("ALLOW_HVAC_MODE_CONTROL", u.get("ALLOW_HVAC_OFF")))
        ):
            return None, None
    try:
        lead_modes = list((state.getattr(lead.get("CLIMATE")) or {}).get("hvac_modes") or [])
        assist_modes = list((state.getattr(assist.get("CLIMATE")) or {}).get("hvac_modes") or [])
    except Exception:
        lead_modes = []
        assist_modes = []
    dry_physical_mode = _dual_physical_hvac_mode("dry", lead)
    if lead_modes and dry_physical_mode not in [str(v).lower() for v in lead_modes]:
        return None, None
    if assist_modes and "heat" not in [str(v).lower() for v in assist_modes]:
        return None, None
    return lead, assist


def _dual_coordinated_dry_available():
    lead, assist = _dual_coordinated_dry_units()
    return bool(lead and assist)


def _dual_is_dry_reheat_assist(u):
    if not (_dual_zone_active and str(_dual_active_mode) == "dry"):
        return False
    _lead, assist = _dual_coordinated_dry_units()
    return bool(assist is not None and u is assist and _dual_dry_reheat_active)


def _dual_float_helper(entity_id, default, lo=None, hi=None, cache=None):
    try:
        if cache is not None:
            value = float(cache.get_float(entity_id, default=default))
        else:
            value = float(state.get(entity_id)) if entity_id else float(default)
    except Exception:
        value = float(default)
    if not isfinite(value):
        value = float(default)
    if lo is not None:
        value = max(float(lo), value)
    if hi is not None:
        value = min(float(hi), value)
    return value


def _dual_optimizer_mode(cache=None):
    ent = globals().get("DUAL_OPTIMIZER_MODE_HELPER", "")
    try:
        value = cache.get_str(ent, default="shadow") if cache is not None else str(state.get(ent) or "shadow")
    except Exception:
        value = "shadow"
    value = str(value).strip().lower()
    return value if value in ("disabled", "shadow", "auto") else "shadow"


def _dual_manual_override(cache=None):
    ent = globals().get("DUAL_MANUAL_OVERRIDE_HELPER", "")
    try:
        value = cache.get_str(ent, default="off") if cache is not None else str(state.get(ent) or "off")
    except Exception:
        value = "off"
    return str(value).lower() == "on"


def _dual_zone_temperature(cache=None):
    cache = cache or _TickCache()
    vals = []
    sensors = globals().get("DUAL_ZONE_INDOOR_SENSORS", []) or []
    for ent in sensors:
        info = _sensor_freshness(
            "indoor", ent,
            globals().get("DUAL_INDOOR_MAX_AGE_S", 300.0),
            cache, True, True,
        )
        value = info.get("value")
        if info.get("usable_for_control") and value is not None and isfinite(value):
            vals.append(float(value))

    if not vals:
        primary = _dual_primary_unit()
        if primary:
            for ent in (primary.get("INDOOR"), primary.get("INDOOR2")):
                if not ent:
                    continue
                info = _sensor_freshness(
                    "indoor", ent,
                    globals().get("DUAL_INDOOR_MAX_AGE_S", 300.0),
                    cache, True, True,
                )
                value = info.get("value")
                if info.get("usable_for_control") and value is not None and isfinite(value):
                    vals.append(float(value))

    if not vals:
        return float("nan")
    return sum(vals) / float(len(vals))


def _dual_outdoor_temperature(cache=None, allow_conservative_fallback=False):
    cache = cache or _TickCache()
    primary = _dual_primary_unit()
    if not primary:
        return float("nan")
    ent = primary.get("OUTDOOR")
    info = _sensor_freshness(
        "outdoor", ent, globals().get("DUAL_OUTDOOR_MAX_AGE_S", 900.0), cache, True
    )
    value = info.get("value")
    if info.get("fresh") and value is not None and isfinite(value):
        return float(value)
    if not allow_conservative_fallback:
        return float("nan")

    fallback = float(globals().get("DUAL_OUTDOOR_STALE_FALLBACK_C", -5.0))
    if not isfinite(fallback):
        fallback = -5.0
    last_good = _sensor_last_good.get(str(ent)) or {}
    last_value = last_good.get("value")
    try:
        last_value = float(last_value)
    except Exception:
        last_value = float("nan")
    # Colder is conservative for the heating Demand floors/caps and dual-assist
    # eligibility. A stale warm last-good value therefore cannot weaken the
    # configured fallback, while a genuinely colder last-good value is retained.
    if isfinite(last_value):
        return min(float(last_value), float(fallback))
    return float(fallback)


def _dual_add_humidity_sample(now, humidity):
    global _dual_humidity_hist
    if not (isfinite(now) and isfinite(humidity) and 0.0 <= float(humidity) <= 100.0):
        _dual_publish_cdp_curve(cache)
        return
    _dual_humidity_hist.append((float(now), float(humidity)))
    window_s = max(60.0, float(globals().get("DUAL_HUMIDITY_FILTER_MINUTES", 10.0)) * 60.0)
    cutoff = float(now) - window_s - 60.0
    _dual_humidity_hist = [
        (t, v) for t, v in _dual_humidity_hist
        if t >= cutoff and isfinite(t) and isfinite(v)
    ][-200:]


def _dual_humidity(cache=None, filtered=True):
    cache = cache or _TickCache()
    ent = globals().get("DUAL_HUMIDITY_SENSOR", "")
    info = _sensor_freshness(
        "humidity", ent, globals().get("DUAL_HUMIDITY_MAX_AGE_S", 600.0), cache, True
    ) if ent else None
    raw = info.get("value") if info and info.get("fresh") else float("nan")
    if not isfinite(raw) or not (0.0 <= float(raw) <= 100.0):
        return float("nan")
    if not filtered:
        return float(raw)
    now = time.time()
    window_s = max(60.0, float(globals().get("DUAL_HUMIDITY_FILTER_MINUTES", 10.0)) * 60.0)
    vals = [v for t, v in _dual_humidity_hist if t >= now - window_s and isfinite(v)]
    return sum(vals) / float(len(vals)) if vals else float(raw)


def _dual_humidity_rate(now):
    """Return actual-running-only RH slope in percentage points/hour."""
    window_s = max(15.0 * 60.0, float(globals().get("DUAL_DRY_RESPONSE_MINUTES", 45.0)) * 60.0)
    pts = [(t, v) for t, v in _dual_learning_humidity_hist if t >= float(now) - window_s]
    if len(pts) < 3:
        return 0.0, False
    span = float(pts[-1][0] - pts[0][0])
    if span < min(15.0 * 60.0, window_s * 0.5):
        return 0.0, False
    mt = sum([p[0] for p in pts]) / float(len(pts))
    mv = sum([p[1] for p in pts]) / float(len(pts))
    den = sum([(t - mt) * (t - mt) for t, _v in pts])
    if den <= 1e-9:
        return 0.0, False
    rate = sum([(t - mt) * (v - mv) for t, v in pts]) / den * 3600.0
    return (float(rate), True) if isfinite(rate) else (0.0, False)


def _dual_supervisor_limits(cache=None):
    cache = cache or _TickCache()
    return {
        "heat_on": _dual_float_helper(globals().get("DUAL_HEAT_ON_DELTA_HELPER", ""), globals().get("DUAL_HEAT_ON_DELTA_C", 0.15), 0.05, 2.0, cache),
        "heat_off": _dual_float_helper(globals().get("DUAL_HEAT_OFF_DELTA_HELPER", ""), globals().get("DUAL_HEAT_OFF_DELTA_C", 0.10), 0.0, 2.0, cache),
        "cool_on": _dual_float_helper(globals().get("DUAL_COOL_ON_DELTA_HELPER", ""), globals().get("DUAL_COOL_ON_DELTA_C", 0.20), 0.05, 2.0, cache),
        "cool_off": _dual_float_helper(globals().get("DUAL_COOL_OFF_DELTA_HELPER", ""), globals().get("DUAL_COOL_OFF_DELTA_C", 0.10), 0.0, 2.0, cache),
        "humidity_target": _dual_float_helper(globals().get("DUAL_HUMIDITY_TARGET_HELPER", ""), globals().get("DUAL_HUMIDITY_TARGET_DEFAULT", 55.0), 35.0, 75.0, cache),
        "humidity_on_delta": _dual_float_helper(globals().get("DUAL_HUMIDITY_ON_DELTA_HELPER", ""), globals().get("DUAL_HUMIDITY_ON_DELTA_DEFAULT", 3.0), 1.0, 15.0, cache),
        "dry_temp_margin": _dual_float_helper(globals().get("DUAL_DRY_TEMP_MARGIN_HELPER", ""), globals().get("DUAL_DRY_TEMP_MARGIN_C", 0.20), 0.0, 2.0, cache),
        "min_on_min": _dual_float_helper(globals().get("DUAL_MIN_ON_MINUTES_HELPER", ""), globals().get("DUAL_MIN_ON_MINUTES_DEFAULT", 20.0), 0.0, 120.0, cache),
        "min_off_min": _dual_float_helper(globals().get("DUAL_MIN_OFF_MINUTES_HELPER", ""), globals().get("DUAL_MIN_OFF_MINUTES_DEFAULT", 10.0), 0.0, 120.0, cache),
        "auto_heat_on": _dual_float_helper(globals().get("DUAL_AUTO_HEAT_ON_DELTA_HELPER", ""), globals().get("DUAL_AUTO_HEAT_ON_DELTA_C", 0.40), 0.10, 3.0, cache),
        "auto_cool_on": _dual_float_helper(globals().get("DUAL_AUTO_COOL_ON_DELTA_HELPER", ""), globals().get("DUAL_AUTO_COOL_ON_DELTA_C", 0.40), 0.10, 3.0, cache),
        "auto_confirm_min": _dual_float_helper(globals().get("DUAL_AUTO_CONFIRMATION_HELPER", ""), globals().get("DUAL_AUTO_MODE_CONFIRMATION_MINUTES", 3.0), 0.0, 30.0, cache),
        "coil_dry_min": _dual_float_helper(globals().get("DUAL_COIL_DRY_MINUTES_HELPER", ""), globals().get("DUAL_COIL_DRY_MINUTES", 15.0), 0.0, 60.0, cache),
    }


def _dual_mode_demand(mode, tin, sp, humidity, limits):
    """Return (demand_on, demand_off, safety_stop, reason) for one mode."""
    if mode not in ("heat", "cool", "dry"):
        return False, False, True, "requested_off"
    if not (isfinite(tin) and isfinite(sp)):
        # The safety layer freezes commands for an invalid indoor input. This
        # secondary guard must also preserve the current physical state.
        return False, False, False, "invalid_temperature_or_setpoint_hold"
    if mode == "heat":
        demand_on = float(tin) <= float(sp) - float(limits["heat_on"])
        demand_off = float(tin) >= float(sp) + float(limits["heat_off"])
        reason = "heat_demand" if demand_on else "heat_satisfied" if demand_off else "heat_hysteresis"
        return bool(demand_on), bool(demand_off), False, reason
    if mode == "cool":
        demand_on = float(tin) >= float(sp) + float(limits["cool_on"])
        demand_off = float(tin) <= float(sp) - float(limits["cool_off"])
        reason = "cool_demand" if demand_on else "cool_satisfied" if demand_off else "cool_hysteresis"
        return bool(demand_on), bool(demand_off), False, reason

    if not isfinite(humidity):
        return False, False, False, "invalid_humidity_hold"
    humidity_satisfied = float(humidity) <= float(limits["humidity_target"])
    if _dual_coordinated_dry_available():
        pause_below = max(
            0.05,
            float(globals().get("DUAL_DRY_LEAD_PAUSE_BELOW_SETPOINT_C", 0.30)),
        )
        temperature_permits_start = float(tin) >= float(sp) - pause_below
        demand_on = bool(
            float(humidity) >= float(limits["humidity_target"]) + float(limits["humidity_on_delta"]) and
            temperature_permits_start
        )
        demand_off = bool(humidity_satisfied)
        reason = (
            "dry_reheat_demand" if demand_on else
            "dry_humidity_satisfied" if humidity_satisfied else
            "dry_reheat_temperature_guard" if not temperature_permits_start else
            "dry_reheat_hysteresis"
        )
        # Temperature is protected by the dedicated assist-heating and
        # lead-pause state machine; it must not stop both units at setpoint.
        return demand_on, demand_off, False, reason

    demand_on = bool(
        float(humidity) >= float(limits["humidity_target"]) + float(limits["humidity_on_delta"]) and
        float(tin) >= float(sp) + float(limits["dry_temp_margin"])
    )
    temp_floor = float(tin) <= float(sp)
    demand_off = bool(humidity_satisfied or temp_floor)
    reason = (
        "dry_demand" if demand_on else
        "dry_temperature_floor" if temp_floor else
        "dry_humidity_satisfied" if humidity_satisfied else
        "dry_hysteresis"
    )
    return bool(demand_on), demand_off, bool(temp_floor), reason


def _dual_auto_desired_mode(tin, sp, humidity, limits, summer):
    """Return the current unconfirmed automatic-mode candidate and reason."""
    if not (isfinite(tin) and isfinite(sp)):
        return "off", "auto_invalid_temperature_or_setpoint"
    if float(tin) <= float(sp) - float(limits["auto_heat_on"]):
        return "heat", "auto_heat_temperature_low"
    if summer and float(tin) >= float(sp) + float(limits["auto_cool_on"]):
        return "cool", "auto_cool_temperature_high"
    if summer and isfinite(humidity):
        humidity_high = float(humidity) >= float(limits["humidity_target"]) + float(limits["humidity_on_delta"])
        if _dual_coordinated_dry_available():
            pause_below = max(
                0.05,
                float(globals().get("DUAL_DRY_LEAD_PAUSE_BELOW_SETPOINT_C", 0.30)),
            )
            temp_headroom = float(tin) >= float(sp) - pause_below
        else:
            temp_headroom = float(tin) >= float(sp) + float(limits["dry_temp_margin"])
        if humidity_high and temp_headroom:
            return (
                "dry",
                "auto_dry_reheat_humidity_high"
                if _dual_coordinated_dry_available()
                else "auto_dry_humidity_high",
            )
        if humidity_high and not temp_headroom:
            return "off", "auto_dry_blocked_temperature"
    if summer and not isfinite(humidity):
        return "off", "auto_dry_blocked_invalid_humidity"
    return "off", "auto_idle_comfort_satisfied"


def _dual_auto_select_mode(now, tin, sp, humidity, limits, summer):
    """Update the persistent auto selection with latching and confirmation."""
    global _dual_auto_selected_mode, _dual_auto_candidate_mode
    global _dual_auto_candidate_since, _dual_auto_selection_reason

    # Keep an active compressor mode latched. Dry may request a confirmed cool
    # changeover if the room becomes clearly warm; all other normal exits are
    # handled by the mode-specific stop thermostat.
    if _dual_zone_active and _dual_active_mode in ("heat", "cool", "dry"):
        if (
            _dual_active_mode == "dry" and summer and isfinite(tin) and isfinite(sp) and
            float(tin) >= float(sp) + float(globals().get("DUAL_AUTO_DRY_TO_COOL_DELTA_C", 0.60))
        ):
            desired = "cool"
            desired_reason = "auto_dry_to_cool_temperature_high"
        else:
            _dual_auto_selected_mode = _dual_active_mode
            _dual_auto_candidate_mode = _dual_active_mode
            _dual_auto_candidate_since = float(now)
            _dual_auto_selection_reason = "auto_active_mode_latched"
            return _dual_auto_selected_mode
    else:
        desired, desired_reason = _dual_auto_desired_mode(tin, sp, humidity, limits, summer)

    if desired == "off":
        _dual_auto_selected_mode = "off"
        _dual_auto_candidate_mode = "off"
        _dual_auto_candidate_since = 0.0
        _dual_auto_selection_reason = desired_reason
        return "off"

    emergency_heat = (
        desired == "heat" and isfinite(tin) and isfinite(sp) and
        float(tin) <= float(sp) - float(globals().get("DUAL_AUTO_EMERGENCY_HEAT_DELTA_C", 1.0))
    )
    if desired != _dual_auto_candidate_mode:
        _dual_auto_candidate_mode = desired
        _dual_auto_candidate_since = float(now)
        _dual_auto_selection_reason = "auto_mode_confirmation"
    candidate_s = max(0.0, float(now) - float(_dual_auto_candidate_since or now))
    confirm_s = float(limits["auto_confirm_min"]) * 60.0
    if emergency_heat or candidate_s >= confirm_s:
        _dual_auto_selected_mode = desired
        _dual_auto_selection_reason = "auto_emergency_heat" if emergency_heat else desired_reason
    else:
        # Keep an active dry mode until a confirmed dry-to-cool changeover.
        if not _dual_zone_active:
            _dual_auto_selected_mode = "off"
    return _dual_auto_selected_mode


def _dual_unit_supports_fan_only(u, cache=None):
    if not bool(globals().get("DUAL_COIL_DRY_ENABLED", True)):
        return False
    if not bool(u.get("ALLOW_FAN_ONLY_COIL_DRY", True)):
        return False
    climate_ent = u.get("CLIMATE")
    if not climate_ent:
        return False
    cache = cache or _TickCache()
    attrs = cache.getattr(climate_ent, {}) or {}
    modes = attrs.get("hvac_modes") or []
    for value in modes:
        if str(value).strip().lower() == str(globals().get("DUAL_COIL_DRY_HVAC_MODE", "fan_only")):
            return True
    return False


def _dual_learning_required_seconds():
    """Continuous confirmed runtime required before any learned value changes."""
    configured = float(globals().get("DUAL_LEARNING_MIN_ACTIVE_MINUTES", 12.0)) * 60.0
    return max(float(TIN_SLOPE_MIN_SPAN_S), max(0.0, configured))


def _dual_reset_learning_window(reason, discard_episode=True):
    """Remove all response history which could span non-conditioning time."""
    global _dual_tin_hist, _dual_learning_humidity_hist
    global _dual_learning_evidence_ok, _dual_learning_allowed
    global _dual_learning_started_at, _dual_learning_signature
    global _dual_learning_block_reason, _dual_learning_actual_running
    global _dual_learning_expected_units, _dual_learning_actual_units
    global _dual_learning_unit_states, _dual_learning_last_reset_reason
    global _dual_learning_last_sample_ts
    global _dual_optimizer_prev_power_w, _dual_optimizer_prev_cop_power_w
    global _dual_optimizer_prev_thermal_w, _dual_optimizer_prev_heating

    _dual_tin_hist[:] = []
    _dual_learning_humidity_hist[:] = []
    _dual_learning_evidence_ok = False
    _dual_learning_allowed = False
    _dual_learning_started_at = 0.0
    _dual_learning_signature = None
    _dual_learning_block_reason = str(reason)
    _dual_learning_actual_running = False
    _dual_learning_expected_units = []
    _dual_learning_actual_units = []
    _dual_learning_unit_states = {}
    _dual_learning_last_reset_reason = str(reason)
    _dual_learning_last_sample_ts = 0.0
    # Never integrate power across an off/idle/fan-only or mode-change gap.
    _dual_optimizer_prev_power_w = None
    _dual_optimizer_prev_cop_power_w = None
    _dual_optimizer_prev_thermal_w = None
    _dual_optimizer_prev_heating = {}
    if str(_dual_sustain_probe.get("state")) == "testing":
        _dual_reset_sustain_probe("learning_reset_%s" % str(reason))
    if discard_episode and _dual_episode is not None:
        _dual_discard_episode("learning_reset_%s" % str(reason))


def _dual_expected_learning_units(action=None):
    """Return units which are meant to contribute output for this action."""
    expected = []
    if action is not None:
        shares = _dual_action_shares(action)
        for idx, u in enumerate(DAIKINS):
            if idx < len(shares) and float(shares[idx]) > 1e-6:
                expected.append(u)
    if not expected:
        primary = _dual_primary_unit()
        if primary:
            expected.append(primary)
    return expected


def _dual_hvac_action_confirms(mode, actual_mode, hvac_action, running=None):
    """Confirm physical conditioning evidence for learning."""
    mode = str(mode or "").strip().lower()
    actual_mode = str(actual_mode or "unknown").strip().lower()
    hvac_action = str(hvac_action or "unknown").strip().lower()
    physical_mode = _dual_physical_hvac_mode(mode)
    mode_ok = actual_mode in (physical_mode, "unknown", "unavailable", "")
    if mode == "heat":
        # Heating requires an explicitly acknowledged physical heat mode. The
        # climate action is preferred, but Faikin/MQTT installations may omit
        # it; an explicitly active configured compressor/running sensor is
        # equivalent positive evidence. Explicitly off running evidence does
        # not confirm idle/unknown actions.
        return bool(
            actual_mode == physical_mode and
            (
                hvac_action in ("heat", "heating") or
                running is True
            )
        )
    if mode == "cool":
        # Temperature cooling must be acknowledged as physical cool with an
        # explicitly cooling compressor action.
        return bool(mode_ok and hvac_action == "cooling")
    if mode == "dry":
        # Native physical dry may still be reported with either a cooling or
        # drying compressor action by different integrations.
        return bool(mode_ok and hvac_action in ("cooling", "drying"))
    return False


def _dual_actual_operation_status(mode, cache=None, action=None):
    """Describe whether the commanded allocation is actually conditioning."""
    cache = cache or _TickCache()
    expected = _dual_expected_learning_units(action)
    expected_names = []
    actual_names = []
    unit_states = {}
    expected_lookup = {}
    for u in expected:
        name = str(u.get("name", "daikin?"))
        expected_names.append(name)
        expected_lookup[name] = True

    block_reason = None
    if not expected_names:
        block_reason = "no_expected_units"

    for u in DAIKINS:
        name = str(u.get("name", "daikin?"))
        info = _unit_hvac_state(u, cache)
        actual_mode = str(info.get("mode") or "unknown").strip().lower()
        hvac_action = str(info.get("action") or "unknown").strip().lower()
        physical_mode = _dual_physical_hvac_mode(mode)
        is_expected = bool(expected_lookup.get(name))
        confirmed = bool(
            is_expected and _dual_hvac_action_confirms(
                mode, actual_mode, hvac_action, info.get("running")
            )
        )
        confirmed_by_running = bool(
            confirmed and str(mode).lower() == "heat" and
            hvac_action not in ("heat", "heating") and
            info.get("running") is True
        )
        unit_reason = (
            "confirmed_by_running_entity"
            if confirmed_by_running else "confirmed" if confirmed else "standby"
        )
        if is_expected and not confirmed:
            if str(mode).lower() == "heat" and actual_mode != str(physical_mode):
                unit_reason = "hvac_mode_mismatch"
            elif (
                str(mode).lower() == "heat" and
                info.get("running") is False and
                hvac_action not in ("heat", "heating")
            ):
                unit_reason = "hvac_not_running"
            elif hvac_action in ("unknown", "unavailable", "", "none"):
                unit_reason = "hvac_action_unknown"
            elif hvac_action in ("idle", "off", "fan"):
                unit_reason = "hvac_not_running"
            elif hvac_action in ("defrosting", "preheating"):
                unit_reason = hvac_action
            elif actual_mode not in (str(physical_mode), "unknown", "unavailable", ""):
                unit_reason = "hvac_mode_mismatch"
            else:
                unit_reason = "hvac_action_mismatch"
            if block_reason is None:
                block_reason = unit_reason
        elif confirmed:
            actual_names.append(name)
        else:
            intentional_fan = bool(
                _dual_heat_standby_fan_required(u, mode, action) and
                actual_mode == "fan_only" and
                hvac_action in ("fan", "idle", "unknown", "", "none")
            )
            # Daikin1 circulation is an intentional part of Daikin2-only
            # heating. Any other zero-share output invalidates comparisons.
            if intentional_fan:
                unit_reason = "intentional_heat_standby_fan"
            elif hvac_action in ("heating", "cooling", "drying", "fan", "defrosting", "preheating"):
                unit_reason = "unexpected_unit_running"
                if block_reason is None:
                    block_reason = unit_reason

        unit_states[name] = {
            "expected": is_expected,
            "logical_mode": str(mode),
            "expected_physical_mode": str(physical_mode),
            "climate_mode": actual_mode,
            "hvac_action": hvac_action,
            "running_evidence": info.get("running"),
            "confirmed_by_running_entity": confirmed_by_running,
            "confirmed_running": confirmed,
            "intentional_standby_fan": bool(
                _dual_heat_standby_fan_required(u, mode, action)
            ),
            "reason": unit_reason,
        }

    actual_running = bool(expected_names and len(actual_names) == len(expected_names) and block_reason is None)
    return {
        "actual_running": actual_running,
        "block_reason": block_reason,
        "expected_units": expected_names,
        "actual_units": actual_names,
        "unit_states": unit_states,
    }


def _dual_add_learning_humidity_sample(now, humidity):
    global _dual_learning_humidity_hist
    if not (isfinite(now) and isfinite(humidity) and 0.0 <= float(humidity) <= 100.0):
        return
    _dual_learning_humidity_hist.append((float(now), float(humidity)))
    window_s = max(15.0 * 60.0, float(globals().get("DUAL_DRY_RESPONSE_MINUTES", 45.0)) * 60.0)
    cutoff = float(now) - window_s - 60.0
    _dual_learning_humidity_hist = [
        (t, v) for t, v in _dual_learning_humidity_hist
        if t >= cutoff and isfinite(t) and isfinite(v)
    ][-200:]


def _dual_update_learning_gate(mode, cache=None, action=None, now=None, system_freeze_active=False, collect_sample=True):
    """Update continuous actual-operation evidence and learning readiness."""
    global _dual_learning_evidence_ok, _dual_learning_allowed
    global _dual_learning_started_at, _dual_learning_signature
    global _dual_learning_block_reason, _dual_learning_actual_running
    global _dual_learning_expected_units, _dual_learning_actual_units
    global _dual_learning_unit_states, _dual_learning_last_sample_ts

    cache = cache or _TickCache()
    now = time.time() if now is None else float(now)
    mode = str(mode or "off").strip().lower()
    status = _dual_actual_operation_status(mode, cache, action)
    block_reason = status.get("block_reason")
    if system_freeze_active:
        block_reason = "system_defrost_freeze"
    elif _controller_shadow_enabled(cache):
        block_reason = "controller_shadow_mode"
    elif _safety_health != "ok":
        block_reason = "safety_%s" % str(_safety_health)
    elif not _normal_optimizer_inputs_available():
        # A finite but stale indoor value is safe for closed-loop comfort
        # control, but it must not create temperature-response learning samples
        # or efficiency comparisons until the sensor reports again.
        block_reason = "indoor_sensor_report_stale_for_learning"
    elif _dual_manual_override(cache):
        block_reason = "manual_override"
    elif _dual_coil_dry_active:
        block_reason = "fan_only"
    elif not _dual_zone_active:
        block_reason = "plant_inactive"
    elif mode not in ("heat", "cool", "dry"):
        block_reason = "mode_not_conditioning"
    elif str(_dual_active_mode) != mode:
        block_reason = "controller_mode_mismatch"
    elif not bool(status.get("actual_running")) and block_reason is None:
        block_reason = "hvac_not_running"

    if block_reason is not None:
        if (
            _dual_learning_evidence_ok or _dual_learning_started_at or
            _dual_tin_hist or _dual_learning_humidity_hist or _dual_episode is not None
        ):
            _dual_reset_learning_window(block_reason, discard_episode=True)
        _dual_learning_block_reason = str(block_reason)
        _dual_learning_actual_running = bool(status.get("actual_running"))
        _dual_learning_expected_units = list(status.get("expected_units") or [])
        _dual_learning_actual_units = list(status.get("actual_units") or [])
        _dual_learning_unit_states = dict(status.get("unit_states") or {})
        return _dual_learning_status_snapshot(now)

    action_id = str((action or {}).get("id") or _dual_active_action_id or "startup_primary")
    signature = "%s|%s|%s" % (
        mode,
        action_id,
        ",".join(list(status.get("expected_units") or [])),
    )
    if signature != _dual_learning_signature:
        _dual_reset_learning_window("conditioning_context_changed", discard_episode=True)
        _dual_learning_signature = signature
        _dual_learning_started_at = now
    elif not _dual_learning_evidence_ok or _dual_learning_started_at <= 0.0:
        _dual_reset_learning_window("actual_operation_resumed", discard_episode=True)
        _dual_learning_signature = signature
        _dual_learning_started_at = now

    _dual_learning_evidence_ok = True
    _dual_learning_actual_running = True
    _dual_learning_expected_units = list(status.get("expected_units") or [])
    _dual_learning_actual_units = list(status.get("actual_units") or [])
    _dual_learning_unit_states = dict(status.get("unit_states") or {})

    sample_interval = max(10.0, float(globals().get("DUAL_LEARNING_SAMPLE_MIN_INTERVAL_S", 30.0)))
    if collect_sample and now - float(_dual_learning_last_sample_ts or 0.0) >= sample_interval:
        tin = _dual_zone_temperature(cache)
        # Use the current raw RH reading here. The supervisor's filtered RH
        # history intentionally spans idle periods, so reusing it would leak
        # pre-start humidity into the dry-mode response learner.
        humidity = _dual_humidity(cache, filtered=False)
        if isfinite(tin):
            _dual_add_temperature_sample(now, tin)
        if isfinite(humidity):
            _dual_add_learning_humidity_sample(now, humidity)
        _dual_learning_last_sample_ts = now

    active_span = max(0.0, now - float(_dual_learning_started_at or now))
    required_s = _dual_learning_required_seconds()
    _dual_learning_allowed = bool(active_span >= required_s)
    _dual_learning_block_reason = None if _dual_learning_allowed else "active_learning_warmup"
    return _dual_learning_status_snapshot(now)


def _dual_learning_status_snapshot(now=None):
    now = time.time() if now is None else float(now)
    active_span = 0.0
    if _dual_learning_evidence_ok and _dual_learning_started_at > 0.0:
        active_span = max(0.0, now - float(_dual_learning_started_at))
    return {
        "actual_running": bool(_dual_learning_actual_running),
        "learning_evidence_ok": bool(_dual_learning_evidence_ok),
        "learning_allowed": bool(_dual_learning_allowed),
        "learning_block_reason": _dual_learning_block_reason,
        "active_learning_span_seconds": active_span,
        "learning_required_seconds": _dual_learning_required_seconds(),
        "learning_signature": _dual_learning_signature,
        "expected_running_units": list(_dual_learning_expected_units),
        "actual_running_units": list(_dual_learning_actual_units),
        "learning_unit_states": dict(_dual_learning_unit_states),
        "temperature_learning_samples": len(_dual_tin_hist),
        "humidity_learning_samples": len(_dual_learning_humidity_hist),
        "learning_last_reset_reason": _dual_learning_last_reset_reason,
    }


def _dual_publish_learning_status(mode=None):
    sensor = globals().get("DUAL_LEARNING_STATUS_SENSOR", "sensor.daikin_dual_learning_status")
    info = _dual_learning_status_snapshot()
    value = "learning" if info.get("learning_allowed") else "warming_up" if info.get("learning_evidence_ok") else "blocked"
    try:
        state.set(
            sensor,
            value=value,
            operating_mode=str(mode or _dual_active_mode),
            actual_running=bool(info.get("actual_running")),
            learning_evidence_ok=bool(info.get("learning_evidence_ok")),
            learning_allowed=bool(info.get("learning_allowed")),
            learning_block_reason=info.get("learning_block_reason"),
            active_learning_span_seconds=round(float(info.get("active_learning_span_seconds", 0.0)), 1),
            learning_required_seconds=round(float(info.get("learning_required_seconds", 0.0)), 1),
            learning_signature=info.get("learning_signature"),
            expected_running_units=info.get("expected_running_units"),
            actual_running_units=info.get("actual_running_units"),
            unit_states=info.get("learning_unit_states"),
            temperature_samples=int(info.get("temperature_learning_samples", 0)),
            humidity_samples=int(info.get("humidity_learning_samples", 0)),
            last_reset_reason=info.get("learning_last_reset_reason"),
        )
    except Exception as e:
        log.debug("Daikin dual: failed to publish learning state: %s", e)


def _dual_start_coil_dry(source_mode, reason, limits, cache=None):
    """Start fan_only only on units whose coil was just active."""
    global _dual_coil_dry_active, _dual_coil_dry_started_at
    global _dual_coil_dry_until, _dual_coil_dry_source_mode
    global _dual_coil_dry_units, _dual_coil_dry_reason

    cache = cache or _TickCache()
    now = time.time()
    duration_s = max(0.0, float(limits.get("coil_dry_min", 15.0)) * 60.0)
    eligible_names = set()
    for u in DAIKINS:
        climate_ent = u.get("CLIMATE")
        actual = cache.get_str(climate_ent, default="off").strip().lower() if climate_ent else "off"
        attrs = cache.getattr(climate_ent, {}) if climate_ent else {}
        action = str((attrs or {}).get("hvac_action") or "unknown").strip().lower()
        used = actual == _dual_physical_hvac_mode(source_mode) or action in ("cooling", "drying")
        if used and _dual_unit_supports_fan_only(u, cache):
            eligible_names.add(str(u.get("name")))

    # The primary has definitely been requested while the shared plant is
    # active. Some integrations update climate state after the service returns,
    # so retain this conservative fallback when no active unit was observable.
    if not eligible_names:
        primary = _dual_primary_unit()
        if primary and _dual_unit_supports_fan_only(primary, cache):
            eligible_names.add(str(primary.get("name")))

    if duration_s <= 0.0 or not eligible_names:
        _dual_coil_dry_active = False
        _dual_coil_dry_units = set()
        _dual_shutdown_all("coil_dry_unavailable")
        return False

    for u in DAIKINS:
        name = str(u.get("name"))
        if name in eligible_names:
            _dual_set_hvac_active(u, True, str(globals().get("DUAL_COIL_DRY_HVAC_MODE", "fan_only")), None)
        else:
            _dual_set_hvac_active(u, False, "off", None)
    _dual_reset_dry_coordination("coil_dry_started")
    _dual_reset_learning_window("coil_dry_started", discard_episode=True)
    _dual_coil_dry_active = True
    _dual_coil_dry_started_at = now
    _dual_coil_dry_until = now + duration_s
    _dual_coil_dry_source_mode = source_mode
    _dual_coil_dry_units = eligible_names
    _dual_coil_dry_reason = reason
    _runtime_store_save(force=False)
    return True


def _dual_stop_coil_dry(reason="coil_dry_complete"):
    global _dual_coil_dry_active, _dual_coil_dry_started_at
    global _dual_coil_dry_until, _dual_coil_dry_source_mode
    global _dual_coil_dry_units, _dual_coil_dry_reason
    for u in DAIKINS:
        _dual_set_hvac_active(u, False, "off", None)
    _dual_reset_dry_coordination(reason)
    _dual_reset_learning_window(reason, discard_episode=True)
    _dual_coil_dry_active = False
    _dual_coil_dry_started_at = 0.0
    _dual_coil_dry_until = 0.0
    _dual_coil_dry_source_mode = None
    _dual_coil_dry_units = set()
    _dual_coil_dry_reason = reason
    _runtime_store_save(force=False)


def _dual_publish_supervisor(info):
    sensor = globals().get("DUAL_MODE_STATUS_SENSOR", "sensor.daikin_dual_mode_status")
    try:
        learning = _dual_learning_status_snapshot()
        if info.get("auto_fan_idle_ventilation"):
            state_value = "auto_fan_idle_ventilation"
        elif info.get("shared_fan_mode_active"):
            state_value = str(info.get("effective_mode"))
        elif info.get("fan_only_active"):
            state_value = "fan_only_coil_dry"
        elif (
            info.get("effective_mode") == "dry" and info.get("active") and
            _dual_coordinated_dry_available()
        ):
            state_value = (
                "dry_reheat_active"
                if _dual_dry_reheat_active
                else "dry_lead_active"
            )
        else:
            state_value = "%s_%s" % (
                str(info.get("effective_mode", "off")),
                "active" if info.get("active") else "waiting",
            )
        if info.get("effective_mode") == "off" and not info.get("fan_only_active"):
            state_value = "auto_idle" if info.get("requested_mode") in ("auto", "auto_fan") else "off"
        candidate_seconds = 0.0
        if _dual_auto_candidate_since:
            candidate_seconds = max(0.0, time.time() - float(_dual_auto_candidate_since))
        coil_remaining = 0.0
        if _dual_coil_dry_active:
            coil_remaining = max(0.0, float(_dual_coil_dry_until) - time.time())
        now = time.time()
        off_remaining = max(0.0, float(info.get("minimum_off_seconds", 0.0)) - float(info.get("offtime_seconds", 0.0)))
        min_on_s = float(_dual_supervisor_limits(_TickCache()).get("min_on_min", 0.0)) * 60.0
        on_remaining = max(0.0, min_on_s - (now - float(_dual_zone_started_at))) if _dual_zone_active and _dual_zone_started_at else 0.0
        next_control = (int(now // 300.0) + 1) * 300.0
        state.set(
            sensor,
            value=state_value,
            requested_mode=info.get("requested_mode"),
            selected_mode=info.get("selected_mode"),
            effective_mode=info.get("effective_mode"),
            physical_hvac_mode=(
                "fan_only"
                if info.get("auto_fan_idle_ventilation")
                else info.get("physical_mode")
                if info.get("shared_fan_mode_active")
                else
                "mixed_dry_heat"
                if (
                    info.get("effective_mode") == "dry" and info.get("active") and
                    _dual_dry_reheat_active
                )
                else _dual_physical_hvac_mode(info.get("effective_mode"))
                if info.get("effective_mode") in ("heat", "cool", "dry", "fan_only")
                else None
            ),
            cooling_via_dry=False,
            cooling_via_cool=bool(info.get("effective_mode") == "cool"),
            coordinated_reheat_drying=bool(
                info.get("effective_mode") == "dry" and
                _dual_coordinated_dry_available()
            ),
            dry_lead_unit=_dual_dry_coordination_status.get("lead_unit"),
            dry_lead_mode=_dual_dry_coordination_status.get("lead_mode"),
            dry_lead_paused=bool(_dual_dry_lead_paused),
            dry_assist_unit=_dual_dry_coordination_status.get("assist_unit"),
            dry_assist_mode=_dual_dry_coordination_status.get("assist_mode"),
            dry_reheat_active=bool(_dual_dry_reheat_active),
            dry_coordination=dict(_dual_dry_coordination_status),
            summer_mode=bool(info.get("summer_mode")),
            heating_lead_unit=str(
                (_dual_heating_lead_unit() or {}).get("name", "unavailable")
            ),
            daikin1_heating_lead=bool(
                str((_dual_heating_lead_unit() or {}).get("name", "")) == "daikin1"
            ),
            summer_blocked=info.get("block_reason") == "summer_mode_off",
            active=bool(info.get("active")),
            fan_only_active=bool(info.get("fan_only_active")),
            daikin1_heat_circulation=bool(
                info.get("effective_mode") == "heat" and info.get("active") and
                bool((_desired_units.get("daikin1") or {}).get("active")) and
                str((_desired_units.get("daikin1") or {}).get("mode", "")) == "fan_only"
            ),
            auto_fan_idle_ventilation=bool(info.get("auto_fan_idle_ventilation")),
            shared_fan_mode_active=bool(info.get("shared_fan_mode_active")),
            coil_dry_source_mode=_dual_coil_dry_source_mode,
            coil_dry_units=list(_dual_coil_dry_units),
            coil_dry_reason=_dual_coil_dry_reason,
            coil_dry_remaining_seconds=round(coil_remaining, 1),
            configured_units=len(DAIKINS),
            single_unit_fallback=(len(DAIKINS) == 1),
            reason=info.get("reason"),
            block_reason=info.get("block_reason"),
            selection_reason=info.get("selection_reason"),
            candidate_mode=_dual_auto_candidate_mode if info.get("requested_mode") in ("auto", "auto_fan") else None,
            candidate_seconds=round(candidate_seconds, 1) if info.get("requested_mode") in ("auto", "auto_fan") else 0.0,
            minimum_on_remaining_seconds=round(on_remaining, 1),
            minimum_off_remaining_seconds=round(off_remaining, 1),
            next_control_epoch=round(next_control, 1),
            next_control_in_seconds=round(max(0.0, next_control - now), 1),
            indoor=round(float(info.get("indoor")), 2) if isfinite(info.get("indoor", float("nan"))) else None,
            effective_setpoint=round(float(info.get("setpoint")), 2) if isfinite(info.get("setpoint", float("nan"))) else None,
            humidity=round(float(info.get("humidity")), 2) if isfinite(info.get("humidity", float("nan"))) else None,
            humidity_target=round(float(info.get("humidity_target")), 1) if isfinite(info.get("humidity_target", float("nan"))) else None,
            hvac_target_temperature=(
                round(float(_dual_climate_target_temperature(info.get("effective_mode"))), 1)
                if info.get("active") and _dual_climate_target_temperature(info.get("effective_mode")) is not None
                else None
            ),
            dry_reheat_hvac_target=(
                round(float(_dual_climate_target_temperature("heat")), 1)
                if _dual_dry_reheat_active else None
            ),
            hvac_target_source=(
                "lead_dry_no_target_assist_fixed_heat_target"
                if (
                    info.get("active") and info.get("effective_mode") == "dry" and
                    _dual_coordinated_dry_available()
                )
                else
                "fixed_mode_target"
                if info.get("active") and info.get("effective_mode") in ("heat", "cool")
                else "no_temperature_command"
            ),
            heat_hvac_target=_dual_climate_target_temperature("heat"),
            cool_hvac_target=_dual_climate_target_temperature("cool"),
            degraded_mode_active=bool(_safety_degraded_active),
            degraded_target=info.get("degraded_target"),
            health=str(_safety_health),
            health_reasons=list(_safety_reasons),
            control_hold_active=bool(
                _safety_degraded_active or _safety_health == "fault" or
                info.get("commands_frozen")
            ),
            commands_frozen=bool(info.get("commands_frozen")),
            shutdown_required=False,
            anomaly_shutdown_policy="never",
            anomaly_shutdown_suppressed=bool(info.get("commands_frozen")),
            original_trigger=list(_safety_original_trigger),
            recovery_remaining_seconds=round(float(_safety_recovery_remaining_s), 1),
            controller_shadow_mode=_controller_shadow_enabled(),
            command_status=_command_status_value(),
            command_pending=len(_command_pending),
            command_recovering=_command_recovering_count(),
            command_verifying=_command_verifying_count(),
            command_failures=len(_command_failures),
            desired_generation=int(_desired_generation),
            desired_units=dict(_desired_units),
            unit_runtime=dict(_unit_runtime),
            startup_demand=_dual_last_startup_demand,
            active_since=round(float(_dual_zone_started_at), 1) if _dual_zone_active else None,
            stopped_since=round(float(_dual_zone_stopped_at), 1) if not _dual_zone_active and _dual_zone_stopped_at else None,
            assist_active=bool(_dual_assist_by_mode.get(str(info.get("effective_mode")), False)),
            actual_running=bool(learning.get("actual_running")),
            learning_allowed=bool(learning.get("learning_allowed")),
            learning_block_reason=learning.get("learning_block_reason"),
            active_learning_span_seconds=round(float(learning.get("active_learning_span_seconds", 0.0)), 1),
            learning_required_seconds=round(float(learning.get("learning_required_seconds", 0.0)), 1),
            expected_running_units=learning.get("expected_running_units"),
            actual_running_units=learning.get("actual_running_units"),
            learning_unit_states=learning.get("learning_unit_states"),
        )
    except Exception as e:
        log.debug("Daikin dual: failed to publish mode supervisor: %s", e)


def _dual_shutdown_all(
    reason="supervisor_off", allow_during_anomaly=False
):
    """Turn off every controller-owned climate unit and discard active learning."""
    force_stop = bool(
        allow_during_anomaly or
        str(reason) in (
            "requested_off",
            "summer_mode_off",
            "heat_hard_comfort_stop",
            "cool_hard_comfort_stop",
            "dry_temperature_floor",
        )
    )
    for u in DAIKINS:
        _dual_set_hvac_active(
            u, False, "off", None,
            allow_during_anomaly=force_stop,
        )
    _dual_reset_dry_coordination(reason)
    _dual_reset_learning_window(reason, discard_episode=True)
    _runtime_store_save(force=False)


def _dual_explicit_stop_supervisor(
    cache, requested, summer, limits, reason="requested_off"
):
    """Honor an explicit stop even while anomaly-driven commands are frozen."""
    global _dual_zone_active, _dual_active_mode, _dual_zone_started_at
    global _dual_zone_stopped_at, _dual_last_control_mode
    global _dual_last_supervisor_reason
    now = time.time()
    stop_reason = str(reason)
    _dual_shutdown_all(reason, allow_during_anomaly=True)
    _dual_zone_active = False
    _dual_active_mode = "off"
    _dual_zone_started_at = 0.0
    _dual_zone_stopped_at = now
    _dual_last_control_mode = "off"
    auto_fan_idle_ventilation = bool(
        requested == "auto_fan" and reason == "summer_mode_off"
    )
    if auto_fan_idle_ventilation:
        for u in DAIKINS:
            _dual_set_hvac_active(u, True, "off_fan", None)
        reason = "auto_fan_idle_ventilation"
    _dual_last_supervisor_reason = str(reason)
    info = {
        "requested_mode": requested,
        "selected_mode": (
            str(_dual_auto_selected_mode)
            if requested in ("auto", "auto_fan") else str(requested)
        ),
        "effective_mode": "off",
        "summer_mode": summer,
        "block_reason": stop_reason,
        "active": False,
        "fan_only_active": bool(auto_fan_idle_ventilation),
        "auto_fan_idle_ventilation": bool(auto_fan_idle_ventilation),
        "shared_fan_mode_active": False,
        "reason": reason,
        "selection_reason": "explicit_stop_overrides_anomaly_hold",
        "indoor": _dual_zone_temperature(cache),
        "setpoint": float("nan"),
        "humidity": _dual_humidity(cache, filtered=True),
        "humidity_target": limits.get("humidity_target"),
        "minimum_off_seconds": float(limits.get("min_off_min", 0.0)) * 60.0,
        "offtime_seconds": 0.0,
        "commands_frozen": False,
        "anomaly_shutdown_suppressed": False,
    }
    _dual_publish_supervisor(info)
    _runtime_store_save(force=False)
    return info


def _dual_shared_fan_supervisor(cache, requested, summer, limits):
    """Continuously own both units in either shared manual fan mode."""
    global _dual_zone_active, _dual_active_mode, _dual_zone_started_at
    global _dual_zone_stopped_at, _dual_last_control_mode
    global _dual_last_supervisor_reason
    global _dual_coil_dry_active, _dual_coil_dry_started_at
    global _dual_coil_dry_until, _dual_coil_dry_source_mode
    global _dual_coil_dry_units, _dual_coil_dry_reason

    now = time.time()
    physical_mode = _dual_physical_hvac_mode(requested)
    reason = "shared_%s" % requested
    _dual_coil_dry_active = False
    _dual_coil_dry_started_at = 0.0
    _dual_coil_dry_until = 0.0
    _dual_coil_dry_source_mode = None
    _dual_coil_dry_units = set()
    _dual_coil_dry_reason = reason
    _dual_reset_dry_coordination(reason)
    _dual_reset_learning_window(reason, discard_episode=True)

    for u in DAIKINS:
        _dual_set_hvac_active(u, True, requested, None)

    compressor_capable = requested == "auto_fan"
    _dual_zone_active = bool(compressor_capable)
    _dual_active_mode = requested
    _dual_zone_started_at = now if compressor_capable else 0.0
    if not compressor_capable:
        _dual_zone_stopped_at = _dual_zone_stopped_at or now
    _dual_last_control_mode = requested
    _dual_last_supervisor_reason = reason
    info = {
        "requested_mode": requested,
        "selected_mode": requested,
        "effective_mode": requested,
        "physical_mode": physical_mode,
        "summer_mode": summer,
        "block_reason": None,
        "active": bool(compressor_capable),
        "fan_only_active": requested == "off_fan",
        "shared_fan_mode_active": True,
        "reason": reason,
        "selection_reason": "manual_shared_fan_mode",
        "indoor": _dual_zone_temperature(cache),
        "setpoint": float("nan"),
        "humidity": _dual_humidity(cache, filtered=True),
        "humidity_target": limits.get("humidity_target"),
        "minimum_off_seconds": 0.0,
        "offtime_seconds": 0.0,
    }
    _dual_publish_supervisor(info)
    _runtime_store_save(force=False)
    return info


def _dual_anomaly_hold_supervisor(
    cache, requested, effective, summer, sp, humidity, limits,
    block_reason="anomaly_control_hold",
):
    """Freeze controller outputs while preserving current HVAC operation."""
    global _dual_last_supervisor_reason
    now = time.time()
    selected = str(_dual_auto_selected_mode) if requested in ("auto", "auto_fan") else str(requested)
    active = bool(_dual_zone_active)
    held_mode = (
        str(_dual_active_mode)
        if active and str(_dual_active_mode) in ("heat", "cool", "dry")
        else str(effective)
    )
    fan_only_active = bool(_dual_coil_dry_active)
    reason = "anomaly_control_hold"
    if block_reason == "missing_climate_mode_control":
        reason = "configuration_control_hold"
    _dual_reset_learning_window(reason, discard_episode=True)
    _dual_last_supervisor_reason = reason
    info = {
        "requested_mode": requested,
        "selected_mode": selected,
        "effective_mode": held_mode,
        "summer_mode": summer,
        "block_reason": block_reason,
        "active": active,
        "fan_only_active": fan_only_active,
        "reason": reason,
        "selection_reason": "anomaly_detected_commands_frozen",
        "indoor": _dual_zone_temperature(cache),
        "setpoint": sp,
        "humidity": humidity,
        "humidity_target": limits.get("humidity_target"),
        "minimum_off_seconds": float(limits.get("min_off_min", 0.0)) * 60.0,
        "offtime_seconds": max(0.0, now - float(_dual_zone_stopped_at)) if _dual_zone_stopped_at else 0.0,
        "degraded_target": None,
        "commands_frozen": True,
        "anomaly_shutdown_suppressed": True,
    }
    _dual_publish_supervisor(info)
    return info


def _dual_degraded_supervisor(cache, requested, effective, summer, sp, humidity, limits):
    """Compatibility wrapper for anomaly-tolerant command hold."""
    return _dual_anomaly_hold_supervisor(
        cache, requested, effective, summer, sp, humidity, limits,
        block_reason="anomaly_control_hold",
    )


def _dual_fault_supervisor(cache, requested, summer, limits):
    """Compatibility wrapper for fault-level anomaly command hold."""
    effective = (
        str(_dual_active_mode)
        if _dual_zone_active and str(_dual_active_mode) in ("heat", "cool", "dry")
        else str(_dual_auto_selected_mode) if requested in ("auto", "auto_fan") else str(requested)
    )
    sp_mode = effective if effective in ("heat", "cool", "dry") else "auto"
    sp, _min_guard, _max_guard, _deadband = _dual_effective_setpoint(
        cache, write_helpers=False, mode=sp_mode
    )
    return _dual_anomaly_hold_supervisor(
        cache, requested, effective, summer, sp,
        _dual_humidity(cache, filtered=True), limits,
        block_reason="anomaly_control_hold",
    )


def _dual_supervisor_tick(cache=None):
    """Automatic/manual mode selection plus plant and coil-dry state machine."""
    global _dual_supervisor_initialized, _dual_zone_active, _dual_active_mode
    global _dual_zone_started_at, _dual_zone_stopped_at, _dual_mode_changed_at
    global _dual_last_control_mode, _dual_err_int, _dual_last_ctrl_ts
    global _dual_last_total_target, _dual_last_supervisor_reason
    global _dual_auto_selected_mode, _dual_auto_candidate_mode
    global _dual_auto_candidate_since, _dual_auto_selection_reason
    global _dual_coil_dry_active, _dual_coil_dry_started_at
    global _dual_coil_dry_until, _dual_coil_dry_source_mode
    global _dual_coil_dry_units, _dual_coil_dry_reason
    global _dual_dry_reheat_active, _dual_dry_reheat_started_at
    global _dual_dry_reheat_stopped_at, _dual_auto_fan_override_active

    cache = cache or _TickCache()
    now = time.time()
    requested = _requested_hvac_mode(cache)
    _dual_auto_fan_override_active = requested == "auto_fan"
    _dual_apply_auto_fan_override()
    summer = _summer_mode_enabled(cache)
    limits = _dual_supervisor_limits(cache)
    mode_control_ready = bool(DAIKINS)
    if mode_control_ready:
        for u in DAIKINS:
            if not (
                bool(u.get("CLIMATE")) and
                bool(u.get("ALLOW_HVAC_MODE_CONTROL", u.get("ALLOW_HVAC_OFF")))
            ):
                mode_control_ready = False
                break

    tin = _dual_zone_temperature(cache)
    humidity = _dual_humidity(cache, filtered=True)

    # Explicit operator/system stop intent remains authoritative. This is not
    # an anomaly-induced shutdown: the anomaly path itself still sends no off
    # command. Disabling summer mode is an explicit stop for cool/dry only.
    summer_stop = bool(
        not summer and (
            str(_dual_active_mode) in ("cool", "dry") or
            str(requested) in ("cool", "dry") or
            (
                str(requested) in ("auto", "auto_fan") and
                str(_dual_auto_selected_mode) in ("cool", "dry")
            )
        )
    )
    if str(requested) == "off" or summer_stop:
        return _dual_explicit_stop_supervisor(
            cache, requested, summer, limits,
            reason="requested_off" if str(requested) == "off" else "summer_mode_off",
        )

    # Apply anomaly/configuration holds before startup reconciliation. The
    # reconciliation path can legitimately finish an expired fan-only cycle;
    # it must not issue that or any other physical command while an anomaly is
    # active. Current HVAC state and Demand are therefore left untouched.
    if _anomaly_control_hold_active() or not mode_control_ready:
        requested_now, effective, summer, _effective_block = _effective_hvac_mode(cache)
        held_mode = (
            str(_dual_active_mode)
            if _dual_zone_active and str(_dual_active_mode) in ("heat", "cool", "dry")
            else str(effective)
        )
        sp_mode = (
            held_mode if held_mode in ("heat", "cool", "dry")
            else "auto" if requested in ("auto", "auto_fan") else str(requested)
        )
        sp, _min_guard, _max_guard, _deadband = _dual_effective_setpoint(
            cache, write_helpers=False, mode=sp_mode
        )
        return _dual_anomaly_hold_supervisor(
            cache, requested_now, held_mode, summer, sp, humidity, limits,
            block_reason=(
                "anomaly_control_hold"
                if _anomaly_control_hold_active()
                else "missing_climate_mode_control"
            ),
        )

    # These are explicit, continuous operator modes. They are deliberately
    # handled before startup recovery so fan_only is never mistaken for an
    # expired post-cooling coil-dry cycle.
    if requested == "off_fan":
        return _dual_shared_fan_supervisor(cache, requested, summer, limits)

    if not _dual_supervisor_initialized:
        stored_active = bool(_dual_zone_active)
        stored_mode = str(_dual_active_mode or "off")
        stored_started = float(_dual_zone_started_at or 0.0)
        stored_stopped = float(_dual_zone_stopped_at or 0.0)
        stored_coil_active = bool(_dual_coil_dry_active)
        stored_coil_until = float(_dual_coil_dry_until or 0.0)
        stored_coil_started = float(_dual_coil_dry_started_at or 0.0)
        stored_coil_units = set(_dual_coil_dry_units)
        any_running = False
        running_mode = "off"
        running_modes = []
        fan_units = set()
        for u in DAIKINS:
            climate_ent = u.get("CLIMATE")
            actual = cache.get_str(climate_ent, default="off").strip().lower() if climate_ent else "off"
            if actual in ("heat", "cool", "dry"):
                any_running = True
                preferred_mode = (
                    stored_mode if stored_active else
                    _dual_auto_selected_mode if requested in ("auto", "auto_fan") else
                    requested
                )
                running_mode = _dual_logical_mode_from_physical(actual, preferred_mode)
                running_modes.append(running_mode)
            elif actual == str(globals().get("DUAL_COIL_DRY_HVAC_MODE", "fan_only")):
                fan_units.add(str(u.get("name")))
        # A coordinated dry restart presents one physical dry unit and one
        # physical heat unit. Mapping the dry unit through the stored logical
        # preference recovers both as one dry cycle instead of
        # letting the later DAIKINS entry overwrite the zone mode to heat.
        preferred_running_mode = (
            stored_mode if stored_active else
            _dual_auto_selected_mode if requested in ("auto", "auto_fan") else requested
        )
        if (
            "dry" in running_modes and "heat" in running_modes and
            str(preferred_running_mode) == "dry" and
            _dual_coordinated_dry_available()
        ):
            running_mode = "dry"
            _dual_dry_reheat_active = True
            if not _dual_dry_reheat_started_at:
                _dual_dry_reheat_started_at = stored_started or now
            _dual_dry_reheat_stopped_at = 0.0
        _dual_zone_active = bool(any_running)
        _dual_active_mode = running_mode if any_running else "off"
        if any_running:
            if stored_active and stored_mode == running_mode and 0.0 < stored_started <= now:
                _dual_zone_started_at = stored_started
            else:
                _dual_zone_started_at = now
            _dual_zone_stopped_at = stored_stopped
        else:
            _dual_zone_started_at = 0.0
            # If the store thought the compressor was active but HA now says
            # off, use restart time as a conservative new compressor stop.
            _dual_zone_stopped_at = now if stored_active else stored_stopped
        if requested in ("auto", "auto_fan") and any_running:
            _dual_auto_selected_mode = running_mode
        if fan_units:
            if stored_coil_active and stored_coil_until > now:
                _dual_coil_dry_active = True
                _dual_coil_dry_started_at = stored_coil_started or now
                _dual_coil_dry_until = stored_coil_until
                _dual_coil_dry_units = fan_units if fan_units else stored_coil_units
                _dual_coil_dry_reason = "restored_remaining_deadline"
                _dual_zone_stopped_at = stored_stopped or _dual_coil_dry_started_at
                _dual_reset_learning_window("fan_only_restored_after_restart", discard_episode=True)
            else:
                # Never restart a full 15-minute fan cycle when its original
                # absolute deadline is unavailable or already expired.
                _dual_coil_dry_active = False
                _dual_coil_dry_started_at = 0.0
                _dual_coil_dry_until = 0.0
                _dual_coil_dry_units = set()
                _dual_coil_dry_reason = "expired_or_untracked_after_restart"
                for u in DAIKINS:
                    _dual_set_hvac_active(u, False, "off", None)
                _dual_zone_stopped_at = stored_stopped or now
                _dual_reset_learning_window("fan_only_expired_after_restart", discard_episode=True)
        elif any_running:
            _dual_coil_dry_active = False
            _dual_reset_learning_window("conditioning_recovered_after_restart", discard_episode=True)
        else:
            _dual_coil_dry_active = False
            _dual_coil_dry_started_at = 0.0
            _dual_coil_dry_until = 0.0
            _dual_coil_dry_units = set()
            _dual_reset_learning_window("startup_inactive", discard_episode=True)
        _dual_last_control_mode = running_mode if any_running else "off"
        _dual_supervisor_initialized = True
        _runtime_store_save(force=False)

    # Auto selects against a neutral, non-price-biased setpoint. After a mode
    # is selected, its own mode-aware setpoint (including price direction) is
    # used by the thermostat and demand controller.
    selection_sp, _sg_min, _sg_max, _sg_db = _dual_effective_setpoint(
        cache, write_helpers=False, mode="auto" if requested in ("auto", "auto_fan") else requested
    )
    if requested in ("auto", "auto_fan") and not _safety_degraded_active and _safety_health != "fault":
        _dual_auto_select_mode(now, tin, selection_sp, humidity, limits, summer)
    requested_now, effective, summer, block_reason = _effective_hvac_mode(cache)
    selected_mode = _dual_auto_selected_mode if requested in ("auto", "auto_fan") else requested
    sp_mode = effective if effective in ("heat", "cool", "dry") else "auto" if requested in ("auto", "auto_fan") else selected_mode
    sp, _min_guard, _max_guard, _deadband = _dual_effective_setpoint(cache, write_helpers=True, mode=sp_mode)

    if _safety_health == "fault":
        return _dual_fault_supervisor(cache, requested_now, summer, limits)
    if _safety_degraded_active:
        return _dual_degraded_supervisor(cache, requested_now, effective, summer, sp, humidity, limits)

    if effective == "off":
        demand_on = False
        demand_off = False
        safety_stop = True
        if block_reason:
            reason = block_reason
        elif requested in ("auto", "auto_fan"):
            reason = _dual_auto_selection_reason
        else:
            reason = "requested_off"
    else:
        demand_on, demand_off, safety_stop, reason = _dual_mode_demand(
            effective, tin, sp, humidity, limits
        )

    runtime_s = now - float(_dual_zone_started_at or now)
    offtime_s = now - float(_dual_zone_stopped_at or 0.0) if _dual_zone_stopped_at else 1e9
    min_on_s = float(limits["min_on_min"]) * 60.0
    min_off_s = float(limits["min_off_min"]) * 60.0
    new_mode_needed = effective in ("heat", "cool", "dry") and bool(demand_on)
    heat_hard_stop_c = max(0.0, float(globals().get("DUAL_HEAT_HARD_STOP_ABOVE_C", 0.25)))
    hard_comfort_stop = bool(
        _dual_zone_active and str(_dual_active_mode) == "heat" and
        isfinite(tin) and isfinite(sp) and float(tin) >= float(sp) + heat_hard_stop_c
    )
    if hard_comfort_stop:
        reason = "heat_hard_comfort_stop"

    # fan_only is an interruptible post-run state, not a compressor state.
    if _dual_coil_dry_active:
        if not mode_control_ready:
            _dual_stop_coil_dry("missing_climate_mode_control")
            reason = "missing_climate_mode_control"
        elif new_mode_needed:
            interrupted_for = effective
            _dual_stop_coil_dry("interrupted_for_%s" % interrupted_for)
            reason = "coil_dry_interrupted_for_%s" % interrupted_for
        elif now >= float(_dual_coil_dry_until or 0.0):
            _dual_stop_coil_dry("coil_dry_complete")
            reason = "coil_dry_complete"
        else:
            for u in DAIKINS:
                if str(u.get("name")) in _dual_coil_dry_units:
                    _dual_set_hvac_active(u, True, str(globals().get("DUAL_COIL_DRY_HVAC_MODE", "fan_only")), None)
                else:
                    _dual_set_hvac_active(u, False, "off", None)
            reason = "coil_dry_fan_only"
            info = {
                "requested_mode": requested_now,
                "selected_mode": selected_mode,
                "effective_mode": effective,
                "summer_mode": summer,
                "block_reason": block_reason,
                "active": False,
                "fan_only_active": True,
                "reason": reason,
                "selection_reason": _dual_auto_selection_reason if requested in ("auto", "auto_fan") else "manual_mode",
                "indoor": tin,
                "setpoint": sp,
                "humidity": humidity,
                "humidity_target": limits["humidity_target"],
                "minimum_off_seconds": min_off_s,
                "offtime_seconds": offtime_s,
            }
            _dual_last_supervisor_reason = reason
            _dual_publish_supervisor(info)
            return info

    # A mode transition discards incomparable learning, clears shared PI state,
    # and enforces a real compressor changeover off interval. A completed cool
    # or dry cycle enters fan_only unless another conditioning mode is needed.
    if effective != _dual_last_control_mode:
        old_mode = _dual_last_control_mode
        _dual_mode_changed_at = now
        _dual_err_int = 0.0
        _dual_last_ctrl_ts = 0.0
        _dual_last_total_target = float("nan")
        _dual_reset_learning_window("mode_changed", discard_episode=True)
        if _dual_zone_active:
            conditioning_mode = _dual_active_mode
            coil_started = False
            if (
                conditioning_mode in ("cool", "dry") and not new_mode_needed and
                block_reason != "missing_climate_mode_control"
            ):
                coil_started = _dual_start_coil_dry(conditioning_mode, "mode_changed", limits, cache)
            if not coil_started:
                _dual_shutdown_all("mode_changed")
            _dual_zone_active = False
            _dual_active_mode = "off"
            _dual_zone_stopped_at = now
            offtime_s = 0.0
            if requested in ("auto", "auto_fan") and not new_mode_needed:
                _dual_auto_selected_mode = "off"
            if coil_started:
                reason = "coil_dry_fan_only"
                selected_mode = _dual_auto_selected_mode if requested in ("auto", "auto_fan") else requested
        _dual_last_control_mode = effective

    if _dual_zone_active:
        if safety_stop or hard_comfort_stop or (demand_off and runtime_s >= min_on_s):
            finished_mode = _dual_active_mode
            coil_started = False
            if finished_mode in ("cool", "dry"):
                coil_started = _dual_start_coil_dry(finished_mode, reason, limits, cache)
            if not coil_started:
                _dual_shutdown_all(reason)
            _dual_zone_active = False
            _dual_active_mode = "off"
            _dual_zone_stopped_at = now
            offtime_s = 0.0
            if requested in ("auto", "auto_fan"):
                _dual_auto_selected_mode = "off"
                _dual_auto_candidate_mode = "off"
                _dual_auto_candidate_since = 0.0
                _dual_auto_selection_reason = "auto_%s_cycle_complete" % finished_mode
                selected_mode = "off"
            if coil_started:
                reason = "coil_dry_fan_only"
        elif demand_off and runtime_s < min_on_s:
            reason = "minimum_on_predictive_taper"
    elif _dual_coil_dry_active:
        # A mode transition may have started fan_only above. Keep it running;
        # the next one-minute tick handles interruption or expiry.
        reason = "coil_dry_fan_only"
    elif safety_stop and requested != "auto_fan":
        # Enforce the summer/off/sensor safety gate even if HA or a user turned
        # a climate entity on while the internal supervisor latch was idle.
        _dual_shutdown_all(reason)
    elif effective in ("heat", "cool", "dry") and demand_on:
        if offtime_s >= min_off_s:
            _dual_reset_learning_window("%s_started" % effective, discard_episode=True)
            _dual_zone_active = True
            _dual_active_mode = effective
            _dual_zone_started_at = now
            start_unit = (
                _dual_summer_conditioning_unit(effective)
                if effective in ("cool", "dry")
                else _dual_heating_lead_unit(cache)
            )
            if effective in ("cool", "dry"):
                for u in DAIKINS:
                    if u is not start_unit:
                        _dual_set_hvac_active(u, False, "off", None)
            if start_unit:
                _dual_set_hvac_active(start_unit, True, effective, sp)
                _dual_apply_startup_demand(start_unit, effective, cache)
            if (
                effective == "heat" and
                str((start_unit or {}).get("name", "")) == "daikin2"
            ):
                for u in DAIKINS:
                    if str(u.get("name", "")) == "daikin1":
                        _dual_set_heat_standby_fan(u)
            reason = "%s_started" % effective
        else:
            reason = "minimum_off_time"

    # ``auto + fan`` is normal controller auto with continuous idle
    # ventilation.  A fan preset on an off climate entity does not run the
    # blower, so explicitly own both units in physical fan_only whenever no
    # conditioning or timed coil-dry cycle is active.  This also keeps
    # ventilation running while a new compressor cycle waits for minimum-off
    # time.  A confirmed heat/cool/dry start above overwrites these desires in
    # the same supervisor tick, so ventilation never delays conditioning.
    auto_fan_idle_ventilation = bool(
        requested == "auto_fan" and
        not _dual_zone_active and
        not _dual_coil_dry_active
    )
    if auto_fan_idle_ventilation:
        for u in DAIKINS:
            _dual_set_hvac_active(u, True, "off_fan", None)
        reason = "auto_fan_idle_ventilation"

    if (
        _dual_zone_active and str(_dual_active_mode) == "dry" and
        _dual_coordinated_dry_available()
    ):
        _dual_run_coordinated_dry(cache)
    if (
        _dual_zone_active and
        str(_dual_active_mode) in ("cool", "dry")
    ):
        # Enforce the fixed Daikin2 summer role on every supervisor pass,
        # including restart reconciliation from an older controller release.
        _dual_reassert_active_hvac_targets()

    _dual_last_supervisor_reason = reason
    info = {
        "requested_mode": requested_now,
        "selected_mode": selected_mode,
        "effective_mode": effective,
        "summer_mode": summer,
        "block_reason": block_reason,
        "active": bool(_dual_zone_active),
        "fan_only_active": bool(_dual_coil_dry_active or auto_fan_idle_ventilation),
        "auto_fan_idle_ventilation": bool(auto_fan_idle_ventilation),
        "shared_fan_mode_active": False,
        "reason": reason,
        "selection_reason": _dual_auto_selection_reason if requested in ("auto", "auto_fan") else "manual_mode",
        "indoor": tin,
        "setpoint": sp,
        "humidity": humidity,
        "humidity_target": limits["humidity_target"],
        "minimum_off_seconds": min_off_s,
        "offtime_seconds": offtime_s,
    }
    _dual_publish_supervisor(info)
    _runtime_store_save(force=False)
    return info


def _dual_effective_setpoint(cache=None, write_helpers=True, mode=None):
    """Calculate one mode-aware effective setpoint for the shared zone."""
    cache = cache or _TickCache()
    if mode not in ("auto", "heat", "cool", "dry"):
        _requested, effective, _summer, _reason = _effective_hvac_mode(cache)
        mode = effective if effective in ("heat", "cool", "dry") else _requested
    primary = _dual_primary_unit()
    if not primary:
        return float(SP_BASE_DEFAULT), float(MIN_GUARD_DEFAULT), float(MAX_GUARD_DEFAULT), 0.1

    base_ent = primary.get("SP_BASE_HELPER", "input_number.daikin_setpoint_base")
    sp_base = cache.get_float(base_ent, default=SP_BASE_DEFAULT)
    if not isfinite(sp_base):
        sp_base = float(SP_BASE_DEFAULT)

    bias_enabled_ent = primary.get("PRICE_BIAS_ENABLED", "input_boolean.nordpool_bias_enabled")
    bias_enabled = cache.get_str(bias_enabled_ent, default="off").lower() == "on"
    bias_ent = primary.get("PRICE_BIAS_HELPER")
    bias_points = cache.get_float(bias_ent, default=0.0) if bias_enabled and bias_ent else 0.0
    if not isfinite(bias_points):
        bias_points = 0.0

    avg_window_h = cache.get_float(NORDPOOL_AVG_WINDOW_HELPER, default=AVG_WINDOW_REF_H)
    if (not isfinite(avg_window_h)) or avg_window_h <= 0.0:
        avg_window_h = float(AVG_WINDOW_REF_H)
    factor = _clip(
        float(AVG_WINDOW_REF_H) / float(avg_window_h),
        float(AVG_WINDOW_FACTOR_MIN),
        float(AVG_WINDOW_FACTOR_MAX),
    )
    bias_c = _clip(
        float(bias_points) * float(SP_BIAS_DEGC_PER_POINT) * factor,
        float(SP_BIAS_CLAMP_MIN),
        float(SP_BIAS_CLAMP_MAX),
    )
    # Auto chooses the operating mode against the neutral comfort setpoint.
    # Applying a heating-oriented price shift before the mode is known could
    # itself bias the selector toward heating or away from cooling.
    if mode == "auto":
        bias_c = 0.0
    # A positive heating price-bias raises the heating setpoint.  The same
    # energy-price signal must be inverted for refrigeration: lowering the
    # cooling setpoint pre-cools when heating would pre-heat.
    if mode in ("cool", "dry") and bool(globals().get("DUAL_INVERT_COOLING_PRICE_BIAS", True)):
        bias_c = -float(bias_c)

    min_ent = primary.get("MIN_TEMP_GUARD_HELPER")
    max_ent = primary.get("MAX_TEMP_GUARD_HELPER")
    min_guard = cache.get_float(min_ent, default=MIN_GUARD_DEFAULT) if min_ent else float(MIN_GUARD_DEFAULT)
    max_guard = cache.get_float(max_ent, default=MAX_GUARD_DEFAULT) if max_ent else float(MAX_GUARD_DEFAULT)
    if not isfinite(min_guard):
        min_guard = float(MIN_GUARD_DEFAULT)
    if not isfinite(max_guard):
        max_guard = float(MAX_GUARD_DEFAULT)
    if min_guard > max_guard:
        min_guard, max_guard = max_guard, min_guard

    sp = _clip(float(sp_base) + float(bias_c), float(min_guard), float(max_guard))
    deadband_ent = primary.get("DEADBAND_HELPER")
    deadband = cache.get_float(deadband_ent, default=AUTO_DB_BASE) if deadband_ent else float(AUTO_DB_BASE)
    deadband = _clip(deadband, float(AUTO_DB_MIN), float(AUTO_DB_MAX))

    if write_helpers:
        for u in DAIKINS:
            sp_ent = u.get("SP_HELPER")
            if sp_ent:
                _set_input_number_if_needed(sp_ent, round(float(sp), 2), eps=SP_WRITE_EPS)

    return float(sp), float(min_guard), float(max_guard), float(deadband)


def _dual_add_temperature_sample(now, tin):
    global _dual_tin_hist
    if not (isfinite(now) and isfinite(tin)):
        return
    _dual_tin_hist.append((float(now), float(tin)))
    cutoff = float(now) - float(TIN_SLOPE_WINDOW_S) - 30.0
    _dual_tin_hist = [(t, v) for t, v in _dual_tin_hist if t >= cutoff and isfinite(t) and isfinite(v)]
    if len(_dual_tin_hist) > 200:
        _dual_tin_hist = _dual_tin_hist[-200:]


def _dual_temperature_rate(now):
    pts = [(t, v) for t, v in _dual_tin_hist if t >= float(now) - float(TIN_SLOPE_WINDOW_S)]
    if len(pts) < int(TIN_SLOPE_MIN_SAMPLES):
        return 0.0, False, 0.0, len(pts)
    span = float(pts[-1][0] - pts[0][0])
    if span < float(TIN_SLOPE_MIN_SPAN_S):
        return 0.0, False, span, len(pts)
    mt = sum([p[0] for p in pts]) / float(len(pts))
    mv = sum([p[1] for p in pts]) / float(len(pts))
    num = sum([(t - mt) * (v - mv) for t, v in pts])
    den = sum([(t - mt) * (t - mt) for t, _v in pts])
    if den <= 1e-9 or not isfinite(num) or not isfinite(den):
        return 0.0, False, span, len(pts)
    rate = (num / den) * 3600.0
    if not isfinite(rate):
        return 0.0, False, span, len(pts)
    return float(rate), True, span, len(pts)


def _dual_add_fast_temperature_sample(now, tin):
    """Keep a short, independent Tin history for rapid landing decisions."""
    global _dual_fast_tin_hist
    if not (isfinite(now) and isfinite(tin)):
        return
    _dual_fast_tin_hist.append((float(now), float(tin)))
    window_s = max(60.0, float(globals().get("DUAL_FAST_RATE_WINDOW_S", 180.0)))
    cutoff = float(now) - window_s - 30.0
    _dual_fast_tin_hist = [
        (t, v) for t, v in _dual_fast_tin_hist
        if t >= cutoff and isfinite(t) and isfinite(v)
    ]
    if len(_dual_fast_tin_hist) > 200:
        _dual_fast_tin_hist = _dual_fast_tin_hist[-200:]


def _dual_fast_temperature_rate(now):
    """Return a short-window regression rate without affecting ML learning."""
    window_s = max(60.0, float(globals().get("DUAL_FAST_RATE_WINDOW_S", 180.0)))
    min_span_s = _clip(
        float(globals().get("DUAL_FAST_RATE_MIN_SPAN_S", 60.0)),
        20.0, window_s,
    )
    min_samples = int(max(3.0, float(globals().get("DUAL_FAST_RATE_MIN_SAMPLES", 4))))
    pts = [
        (t, v) for t, v in _dual_fast_tin_hist
        if t >= float(now) - window_s and isfinite(t) and isfinite(v)
    ]
    if len(pts) < min_samples:
        return 0.0, False, 0.0, len(pts)
    span = float(pts[-1][0] - pts[0][0])
    if span < min_span_s:
        return 0.0, False, span, len(pts)
    mt = sum([p[0] for p in pts]) / float(len(pts))
    mv = sum([p[1] for p in pts]) / float(len(pts))
    num = sum([(t - mt) * (v - mv) for t, v in pts])
    den = sum([(t - mt) * (t - mt) for t, _v in pts])
    if den <= 1e-9 or not isfinite(num) or not isfinite(den):
        return 0.0, False, span, len(pts)
    rate = (num / den) * 3600.0
    if not isfinite(rate):
        return 0.0, False, span, len(pts)
    return float(rate), True, span, len(pts)


def _dual_current_effective_demand(u, cache=None):
    cache = cache or _TickCache()
    select_ent = u.get("SELECT")
    raw = cache.get_str(select_ent, default="")
    base = _parse_select_numeric(select_ent, raw, default=MIN_DEM)
    quiet_ent = u.get("QUIET_OUTDOOR_SWITCH")
    if quiet_ent and base >= 100.0 - 1e-6:
        quiet = cache.get_str(quiet_ent, default="on").lower()
        if quiet == "off":
            return float(MAX_DEM_LAYER)
    return float(base)


def _dual_current_total_demand(cache=None, action=None):
    cache = cache or _TickCache()
    total = 0.0
    shares = _dual_action_shares(action) if action is not None else None
    for idx, u in enumerate(DAIKINS):
        if shares is not None and shares[idx] <= 1e-6:
            continue
        weight = _dual_float_helper(None, u.get("CAPACITY_WEIGHT", 1.0), 0.05, 20.0)
        total += weight * _dual_current_effective_demand(u, cache)
    return float(total)


def _dual_unit_bounds(u, tout, tin, min_guard, cache=None, mode="heat"):
    cache = cache or _TickCache()
    if mode in ("cool", "dry"):
        lower_default = globals().get("DUAL_COOL_MIN_DEMAND", MIN_DEM) if mode == "cool" else globals().get("DUAL_DRY_MIN_DEMAND", MIN_DEM)
        upper_default = globals().get("DUAL_COOL_MAX_DEMAND", MAX_DEM) if mode == "cool" else globals().get("DUAL_DRY_MAX_DEMAND", 70.0)
        lower = _clip(float(lower_default), 0.0, 100.0)
        upper = _clip(float(upper_default), lower, 100.0)
        return float(lower), float(upper), False

    bucket = int(round(float(tout)))
    lower = float(_min_demand_floor_for_outdoor(u, bucket))

    global_upper = float(MAX_DEM) if bucket <= -5 else float(GLOBAL_MILD_MAX)
    icing_ent = u.get("ICING_CAP_HELPER")
    icing_cap = cache.get_float(icing_ent, default=ICING_BAND_CAP_DEFAULT) if icing_ent else float(ICING_BAND_CAP_DEFAULT)
    icing_cap = _clip(icing_cap, float(MIN_DEM), float(MAX_DEM))
    in_icing = float(ICING_BAND_MIN) <= float(tout) <= float(ICING_BAND_MAX)
    upper = min(global_upper, icing_cap) if in_icing else global_upper
    upper = min(upper, _max_demand_cap_for_outdoor(u, bucket, upper))

    # The virtual 105 layer is an emergency layer only. It becomes available
    # after the measured zone temperature breaches the configured minimum guard.
    if (
        u.get("QUIET_OUTDOOR_SWITCH") and
        isfinite(tin) and isfinite(min_guard) and float(tin) < float(min_guard)
    ):
        upper = float(MAX_DEM_LAYER)

    if lower > upper:
        upper = lower
    return float(lower), float(upper), bool(in_icing)


def _dual_action_by_id(action_id):
    if len(DAIKINS) == 1:
        if str(action_id) == "single_unit":
            return {"id": "single_unit", "shares": [1.0]}
        return None
    for action in globals().get("DUAL_ACTIONS", []) or []:
        if str(action.get("id")) == str(action_id):
            return action
    return None


def _dual_action_shares(action):
    shares = list((action or {}).get("shares") or [])
    while len(shares) < len(DAIKINS):
        shares.append(0.0)
    shares = shares[:len(DAIKINS)]
    clean = []
    for value in shares:
        try:
            fv = max(0.0, float(value))
        except Exception:
            fv = 0.0
        clean.append(fv if isfinite(fv) else 0.0)
    total = sum(clean)
    if total <= 1e-9:
        return [1.0 / float(len(clean)) for _ in clean] if clean else []
    return [v / total for v in clean]


def _dual_heat_standby_fan_required(u, mode, action):
    """True only when Daikin2 alone supplies heat and Daikin1 circulates air."""
    if str(mode or "").strip().lower() != "heat" or action is None:
        return False
    shares = _dual_action_shares(action)
    active_names = [
        str(DAIKINS[idx].get("name", ""))
        for idx, share in enumerate(shares)
        if idx < len(DAIKINS) and float(share) > 1e-6
    ]
    return bool(
        str((u or {}).get("name", "")) == "daikin1" and
        active_names == ["daikin2"]
    )


def _dual_eligible_actions(mode=None, assist_allowed=True):
    operating_mode = str(mode or "").lower()
    if len(DAIKINS) == 1:
        # Do not normalize the two-unit action table down to one element: that
        # would create several identical policies and corrupt comparisons.
        fixed_unit = _dual_summer_conditioning_unit(operating_mode)
        if operating_mode in ("cool", "dry") and DAIKINS[0] is not fixed_unit:
            return []
        return [{"id": "single_unit", "shares": [1.0]}]

    # Cooling and drying are role-fixed, not optimizer-allocated. Even when
    # coordinated reheat is unavailable, never fall back to a policy that puts
    # Daikin1 (or any other unit) into a refrigeration role.
    if operating_mode in ("cool", "dry"):
        conditioning = _dual_summer_conditioning_unit(operating_mode)
        if conditioning not in DAIKINS:
            return []
        conditioning_idx = DAIKINS.index(conditioning)
        fixed_actions = []
        for action in globals().get("DUAL_ACTIONS", []) or []:
            shares = _dual_action_shares(action)
            active_indices = [
                idx for idx, value in enumerate(shares) if value > 1e-6
            ]
            if active_indices == [conditioning_idx]:
                fixed_actions.append(action)
        return fixed_actions

    eligible = []
    allow_singles = bool(globals().get("DUAL_ALLOW_SINGLE_UNIT_ACTIONS", True))
    primary = (
        _dual_heating_lead_unit()
        if operating_mode == "heat"
        else _dual_primary_unit()
    )
    primary_idx = DAIKINS.index(primary) if primary in DAIKINS else 0
    for action in globals().get("DUAL_ACTIONS", []) or []:
        shares = _dual_action_shares(action)
        active_count = sum([1 for v in shares if v > 1e-6])
        if active_count < 2 and not allow_singles:
            continue
        if not assist_allowed:
            active_indices = [idx for idx, value in enumerate(shares) if value > 1e-6]
            if active_indices != [primary_idx]:
                continue

        valid = True
        for idx, share in enumerate(shares):
            if share > 1e-6:
                continue
            u = DAIKINS[idx]
            # A zero-share action is real only if the unit can be put in standby.
            if not (bool(u.get("ALLOW_HVAC_MODE_CONTROL", u.get("ALLOW_HVAC_OFF"))) and bool(u.get("CLIMATE"))):
                valid = False
                break
        if valid:
            eligible.append(action)

    if not eligible and assist_allowed:
        # All-positive actions are always safe because they never power-cycle a unit.
        eligible = [
            a for a in (globals().get("DUAL_ACTIONS", []) or [])
            if all([v > 1e-6 for v in _dual_action_shares(a)])
        ]
    return eligible


def _dual_default_action(eligible=None):
    eligible = eligible if eligible is not None else _dual_eligible_actions()
    if len(DAIKINS) == 1:
        return eligible[0] if eligible else None
    wanted = str(globals().get("DUAL_DEFAULT_ACTION", "balanced"))
    for action in eligible:
        if str(action.get("id")) == wanted:
            return action
    return eligible[0] if eligible else None


def _dual_context(tout, total_target, total_max, mode="heat", humidity=None):
    step = max(0.5, float(globals().get("DUAL_OUTDOOR_CONTEXT_STEP_C", 2.0)))
    bucket = round(float(tout) / step) * step
    frac = float(total_target) / float(total_max) if total_max > 1e-9 else 0.0
    if frac < 0.40:
        load = "low"
    elif frac < 0.70:
        load = "medium"
    else:
        load = "high"
    humidity_bucket = "na"
    if mode == "dry" and humidity is not None and isfinite(humidity):
        humidity_bucket = str(int(round(float(humidity) / 5.0) * 5))
    return "%s|%+.1f|%s|rh%s" % (str(mode), float(bucket), load, humidity_bucket)


def _dual_allocate(total_target, action, bounds):
    """Capacity-aware water-filling allocation. Returns demand or None per unit."""
    shares = _dual_action_shares(action)
    active = [idx for idx, share in enumerate(shares) if share > 1e-6]
    if not active:
        return [None for _u in DAIKINS]

    cap_weights = [
        _dual_float_helper(None, u.get("CAPACITY_WEIGHT", 1.0), 0.05, 20.0)
        for u in DAIKINS
    ]
    q_min = [0.0 for _u in DAIKINS]
    q_max = [0.0 for _u in DAIKINS]
    q = [0.0 for _u in DAIKINS]
    for idx in active:
        lo, hi, _icing = bounds[idx]
        q_min[idx] = cap_weights[idx] * float(lo)
        q_max[idx] = cap_weights[idx] * float(hi)
        q[idx] = _clip(float(total_target) * shares[idx], q_min[idx], q_max[idx])

    target = _clip(
        float(total_target),
        sum([q_min[idx] for idx in active]),
        sum([q_max[idx] for idx in active]),
    )

    # Redistribute clamp residual while retaining the selected policy ratio as
    # far as unit limits allow.
    for _pass in range(4):
        residual = target - sum([q[idx] for idx in active])
        if abs(residual) <= 0.01:
            break
        if residual > 0.0:
            candidates = [idx for idx in active if q[idx] < q_max[idx] - 0.01]
            rooms = {idx: q_max[idx] - q[idx] for idx in candidates}
        else:
            candidates = [idx for idx in active if q[idx] > q_min[idx] + 0.01]
            rooms = {idx: q[idx] - q_min[idx] for idx in candidates}
        if not candidates:
            break
        denom = sum([max(0.001, shares[idx]) for idx in candidates])
        for idx in candidates:
            portion = abs(residual) * max(0.001, shares[idx]) / denom
            change = min(portion, rooms[idx])
            q[idx] += change if residual > 0.0 else -change

    demands = []
    for idx in range(len(DAIKINS)):
        if idx not in active:
            demands.append(None)
        else:
            demands.append(q[idx] / cap_weights[idx])
    return demands


def _dual_heating_landing_snapshot(tin, sp, directed_rate, rate_valid, total_min, total_max):
    """Calculate a rate-aware, reduction-only heating Demand ceiling."""
    start_c = max(0.05, float(globals().get("DUAL_HEAT_LANDING_START_C", 0.40)))
    prediction_min = max(0.0, float(globals().get("DUAL_HEAT_PREDICTION_MINUTES", 10.0)))
    err = float(sp) - float(tin)
    approach_rate = max(0.0, float(directed_rate)) if rate_valid and isfinite(directed_rate) else 0.0
    projected_error = float(err) - approach_rate * prediction_min / 60.0
    projected_temperature = float(sp) - projected_error
    fraction = _clip(projected_error / start_c, 0.0, 1.0)
    landing_cap = float(total_min) + fraction * (float(total_max) - float(total_min))
    eta = None
    if err <= 0.0:
        eta = 0.0
    elif approach_rate > 0.001:
        eta = max(0.0, err / approach_rate * 60.0)
    active = bool(projected_error < start_c - 1e-6)
    reason = "projected_setpoint_approach" if active else "outside_landing_band"
    if active and projected_error <= 0.0:
        reason = "projected_at_or_above_setpoint"
    return {
        "active": active,
        "reason": reason,
        "error": float(err),
        "projected_error": float(projected_error),
        "projected_temperature": float(projected_temperature),
        "eta_to_setpoint_minutes": eta,
        "landing_fraction": float(fraction),
        "landing_cap": _clip(float(landing_cap), float(total_min), float(total_max)),
        "total_min": float(total_min),
        "total_max": float(total_max),
        "prediction_minutes": float(prediction_min),
        "directed_rate": float(directed_rate) if isfinite(directed_rate) else 0.0,
        "rate_valid": bool(rate_valid),
    }


def _dual_probe_snapshot():
    return {
        "state": str(_dual_sustain_probe.get("state") or "idle"),
        "context": _dual_sustain_probe.get("context"),
        "candidate": _dual_sustain_probe.get("candidate"),
        "baseline": _dual_sustain_probe.get("baseline"),
        "started_at": float(_dual_sustain_probe.get("started_at") or 0.0),
        "settled_at": float(_dual_sustain_probe.get("settled_at") or 0.0),
        "cooldown_until": float(_dual_sustain_probe.get("cooldown_until") or 0.0),
        "reason": str(_dual_sustain_probe.get("reason") or "idle"),
    }


def _dual_reset_sustain_probe(reason, keep_result=False):
    """Abort an in-flight sustain test without changing its learned baseline."""
    global _dual_sustain_probe
    if keep_result and str(_dual_sustain_probe.get("state")) in ("accepted", "failed"):
        _dual_sustain_probe["reason"] = str(reason)
        return
    cooldown_until = float(_dual_sustain_probe.get("cooldown_until") or 0.0)
    _dual_sustain_probe = {
        "state": "idle",
        "context": None,
        "candidate": None,
        "baseline": None,
        "started_at": 0.0,
        "settled_at": 0.0,
        "cooldown_until": cooldown_until,
        "reason": str(reason),
    }


def _dual_sustain_probe_target(ctx, sustain, total_min, total_max, err, directed_rate,
                               rate_valid, observed_total, learning_allowed, now):
    """Return an exact lower Demand test target, or None when no test is active."""
    global _dual_sustain_probe
    enabled = bool(globals().get("DUAL_HEAT_SUSTAIN_PROBE_ENABLED", True))
    if not enabled:
        _dual_reset_sustain_probe("disabled")
        return None

    state_name = str(_dual_sustain_probe.get("state") or "idle")
    current_ctx = _dual_sustain_probe.get("context")
    cooldown_until = float(_dual_sustain_probe.get("cooldown_until") or 0.0)
    if state_name == "testing" and str(current_ctx) != str(ctx):
        _dual_reset_sustain_probe("context_changed")
        state_name = "idle"
    if state_name in ("accepted", "failed") and now >= cooldown_until:
        _dual_reset_sustain_probe("cooldown_complete")
        state_name = "idle"

    if state_name == "testing":
        if not learning_allowed:
            _dual_reset_sustain_probe("learning_not_allowed")
            return None
        candidate = float(_dual_sustain_probe.get("candidate"))
        fail_err = max(0.01, float(globals().get("DUAL_HEAT_SUSTAIN_PROBE_FAIL_ERR_C", 0.20)))
        max_fall = max(0.0, float(globals().get("DUAL_HEAT_SUSTAIN_PROBE_MAX_FALL_CPH", 0.05)))
        settled_at = float(_dual_sustain_probe.get("settled_at") or 0.0)
        if float(err) > fail_err or (settled_at > 0.0 and rate_valid and float(directed_rate) < -max_fall):
            cooldown_min = max(0.0, float(globals().get("DUAL_HEAT_SUSTAIN_PROBE_COOLDOWN_MINUTES", 30.0)))
            _dual_sustain_probe["state"] = "failed"
            _dual_sustain_probe["cooldown_until"] = float(now) + cooldown_min * 60.0
            _dual_sustain_probe["reason"] = "temperature_falling_away"
            return None
        tolerance = max(0.5, float(globals().get("DUAL_HEAT_SUSTAIN_PROBE_SETTLE_TOLERANCE", 2.6)))
        if observed_total is not None and abs(float(observed_total) - candidate) <= tolerance:
            if settled_at <= 0.0:
                settled_at = float(now)
                _dual_sustain_probe["settled_at"] = settled_at
                _dual_sustain_probe["reason"] = "candidate_settled"
        else:
            _dual_sustain_probe["settled_at"] = 0.0
            _dual_sustain_probe["reason"] = "ramping_to_candidate"
            settled_at = 0.0
        hold_s = max(60.0, float(globals().get("DUAL_HEAT_SUSTAIN_PROBE_HOLD_MINUTES", 15.0)) * 60.0)
        if settled_at > 0.0 and float(now) - settled_at >= hold_s:
            accepted = _clip(candidate, float(total_min), float(total_max))
            _dual_sustain_by_ctx[str(ctx)] = float(accepted)
            cooldown_min = max(0.0, float(globals().get("DUAL_HEAT_SUSTAIN_PROBE_COOLDOWN_MINUTES", 30.0)))
            _dual_sustain_probe["state"] = "accepted"
            _dual_sustain_probe["cooldown_until"] = float(now) + cooldown_min * 60.0
            _dual_sustain_probe["reason"] = "lower_sustain_accepted"
            _dual_save_store(force=False)
            return float(accepted)
        return float(candidate)

    if now < cooldown_until or not learning_allowed or not rate_valid:
        return None
    start_err = max(0.01, float(globals().get("DUAL_HEAT_SUSTAIN_PROBE_START_ERR_C", 0.10)))
    stable_rate = max(0.0, float(globals().get("DUAL_SUSTAIN_RATE_MAX_CPH", 0.08)))
    if abs(float(err)) > start_err or abs(float(directed_rate)) > stable_rate:
        return None
    step = max(1.0, float(globals().get("DUAL_HEAT_SUSTAIN_PROBE_STEP", 5.0)))
    candidate = _clip(float(sustain) - step, float(total_min), float(total_max))
    if observed_total is not None:
        # Never raise output merely to run a test. If the landing governor has
        # already reached a lower stable level, validate that observed level.
        candidate = min(candidate, _clip(float(observed_total), float(total_min), float(total_max)))
    if candidate >= float(sustain) - 0.5:
        return None
    _dual_sustain_probe = {
        "state": "testing",
        "context": str(ctx),
        "candidate": float(candidate),
        "baseline": float(sustain),
        "started_at": float(now),
        "settled_at": 0.0,
        "cooldown_until": 0.0,
        "reason": "lower_sustain_test_started",
    }
    return float(candidate)


def _dual_publish_landing_status(info, total_target=None):
    """Publish the one-minute landing governor and probe diagnostics."""
    sensor = globals().get("DUAL_LANDING_STATUS_SENSOR", "sensor.daikin_dual_landing_status")
    probe = _dual_probe_snapshot()
    active = bool((info or {}).get("active"))
    try:
        state.set(
            sensor,
            value="active" if active else "inactive",
            operating_mode=str(_dual_active_mode),
            reason=(info or {}).get("reason"),
            projected_error=round(float((info or {}).get("projected_error")), 4)
                if (info or {}).get("projected_error") is not None else None,
            projected_temperature=round(float((info or {}).get("projected_temperature")), 3)
                if (info or {}).get("projected_temperature") is not None else None,
            eta_to_setpoint_minutes=round(float((info or {}).get("eta_to_setpoint_minutes")), 2)
                if (info or {}).get("eta_to_setpoint_minutes") is not None else None,
            landing_fraction=round(float((info or {}).get("landing_fraction")), 4)
                if (info or {}).get("landing_fraction") is not None else None,
            landing_cap=round(float((info or {}).get("landing_cap")), 2)
                if (info or {}).get("landing_cap") is not None else None,
            step_limited_total=round(float(total_target), 2) if total_target is not None else None,
            prediction_minutes=(info or {}).get("prediction_minutes"),
            directed_rate=(info or {}).get("directed_rate"),
            rate_valid=bool((info or {}).get("rate_valid")),
            rapid_approach=bool((info or {}).get("rapid_approach")),
            control_interval_seconds=(info or {}).get("control_interval_seconds"),
            fast_rate_cph=(info or {}).get("fast_rate_cph"),
            fast_rate_valid=bool((info or {}).get("fast_rate_valid")),
            fast_rate_span_seconds=(info or {}).get("fast_rate_span_seconds"),
            fast_rate_samples=(info or {}).get("fast_rate_samples"),
            long_rate_cph=(info or {}).get("long_rate_cph"),
            long_rate_valid=bool((info or {}).get("long_rate_valid")),
            rate_source=(info or {}).get("rate_source"),
            sustain_probe=probe,
            updated_at=time.time(),
        )
    except Exception as e:
        log.debug("Daikin dual: failed to publish landing state: %s", e)


def _dual_fast_landing_tick(cache=None):
    """Adaptive reduction-only heat governor; never raises Demand."""
    global _dual_last_fast_taper_ts, _dual_last_total_target, _dual_landing_status
    global _dual_fast_tin_hist
    now = time.time()
    normal_interval_s = max(10.0, float(globals().get("DUAL_FAST_TAPER_INTERVAL_S", 60.0)))
    rapid_interval_s = _clip(
        float(globals().get("DUAL_RAPID_TAPER_INTERVAL_S", 20.0)),
        10.0, normal_interval_s,
    )
    elapsed_s = now - float(_dual_last_fast_taper_ts or 0.0)
    if (
        not _dual_zone_active or str(_dual_active_mode) != "heat" or
        _safety_health == "fault" or _safety_degraded_active or
        _system_defrost_freeze_active
    ):
        _dual_fast_tin_hist = []
        if elapsed_s < normal_interval_s - 1.0:
            return None
        _dual_last_fast_taper_ts = now
        _dual_landing_status = {
            "active": False, "reason": "heat_not_eligible",
            "projected_error": None, "projected_temperature": None,
            "eta_to_setpoint_minutes": None, "landing_cap": None,
            "rapid_approach": False,
            "control_interval_seconds": normal_interval_s,
            "fast_rate_cph": None, "long_rate_cph": None,
            "rate_source": "none",
        }
        _dual_publish_landing_status(_dual_landing_status)
        return None
    cache = cache or _TickCache()
    if _dual_manual_override(cache):
        _dual_fast_tin_hist = []
        if elapsed_s < normal_interval_s - 1.0:
            return None
        _dual_last_fast_taper_ts = now
        _dual_landing_status = {
            "active": False, "reason": "manual_override",
            "projected_error": None, "projected_temperature": None,
            "eta_to_setpoint_minutes": None, "landing_cap": None,
            "rapid_approach": False,
            "control_interval_seconds": normal_interval_s,
            "fast_rate_cph": None, "long_rate_cph": None,
            "rate_source": "none",
        }
        _dual_publish_landing_status(_dual_landing_status)
        return None
    tin = _dual_zone_temperature(cache)
    tout = _dual_outdoor_temperature(cache, allow_conservative_fallback=True)
    sp, min_guard, _max_guard, _deadband = _dual_effective_setpoint(cache, write_helpers=True, mode="heat")
    if not (isfinite(tin) and isfinite(tout) and isfinite(sp)):
        return None
    _dual_add_fast_temperature_sample(now, tin)
    long_rate, long_rate_valid, _long_span, _long_samples = _dual_temperature_rate(now)
    fast_rate, fast_rate_valid, fast_span, fast_samples = _dual_fast_temperature_rate(now)
    rate_clamp = float(globals().get("DUAL_RATE_CLAMP_CPH", 2.0))
    long_directed = _clip(float(long_rate), -rate_clamp, rate_clamp) if long_rate_valid else 0.0
    fast_directed = _clip(float(fast_rate), -rate_clamp, rate_clamp) if fast_rate_valid else 0.0
    rate_candidates = []
    if long_rate_valid:
        rate_candidates.append(("long", float(long_directed)))
    if fast_rate_valid:
        rate_candidates.append(("fast", float(fast_directed)))
    if rate_candidates:
        selected_rate = max(rate_candidates, key=lambda item: item[1])
        directed_rate = float(selected_rate[1])
        rate_source = str(selected_rate[0])
        rate_valid = True
    else:
        directed_rate = 0.0
        rate_source = "none"
        rate_valid = False
    startup_hold_s = max(0.0, float(globals().get("DUAL_STARTUP_DEMAND_HOLD_S", 60.0)))
    if _dual_zone_started_at and now - float(_dual_zone_started_at) < startup_hold_s:
        if elapsed_s < normal_interval_s - 1.0:
            return None
        _dual_last_fast_taper_ts = now
        _dual_landing_status = {
            "active": False, "reason": "startup_demand_hold",
            "projected_error": None, "projected_temperature": None,
            "eta_to_setpoint_minutes": None, "landing_cap": None,
            "rapid_approach": False,
            "control_interval_seconds": normal_interval_s,
            "fast_rate_cph": round(float(fast_directed), 4) if fast_rate_valid else None,
            "fast_rate_valid": bool(fast_rate_valid),
            "fast_rate_span_seconds": round(float(fast_span), 1),
            "fast_rate_samples": int(fast_samples),
            "long_rate_cph": round(float(long_directed), 4) if long_rate_valid else None,
            "long_rate_valid": bool(long_rate_valid),
            "rate_source": rate_source,
        }
        _dual_publish_landing_status(_dual_landing_status)
        return None
    bounds = [_dual_unit_bounds(u, tout, tin, min_guard, cache, mode="heat") for u in DAIKINS]
    action = _dual_action_by_id(_dual_active_action_id)
    if action is None:
        action = _dual_default_action(_dual_eligible_actions(mode="heat", assist_allowed=bool(_dual_assist_by_mode.get("heat"))))
    if action is None:
        return None
    shares = _dual_action_shares(action)
    weights = [_dual_float_helper(None, u.get("CAPACITY_WEIGHT", 1.0), 0.05, 20.0) for u in DAIKINS]
    active_indices = [idx for idx, share in enumerate(shares) if share > 1e-6]
    total_min = sum([weights[idx] * bounds[idx][0] for idx in active_indices])
    total_max = sum([weights[idx] * bounds[idx][1] for idx in active_indices])
    info = _dual_heating_landing_snapshot(tin, sp, directed_rate, rate_valid, total_min, total_max)
    rapid_rate_threshold = max(
        0.01, float(globals().get("DUAL_RAPID_APPROACH_RATE_CPH", 0.30))
    )
    rapid_approach = bool(
        info.get("active") and fast_rate_valid and
        float(fast_directed) >= rapid_rate_threshold
    )
    selected_interval_s = rapid_interval_s if rapid_approach else normal_interval_s
    info["rapid_approach"] = bool(rapid_approach)
    info["control_interval_seconds"] = float(selected_interval_s)
    info["fast_rate_cph"] = round(float(fast_directed), 4) if fast_rate_valid else None
    info["fast_rate_valid"] = bool(fast_rate_valid)
    info["fast_rate_span_seconds"] = round(float(fast_span), 1)
    info["fast_rate_samples"] = int(fast_samples)
    info["long_rate_cph"] = round(float(long_directed), 4) if long_rate_valid else None
    info["long_rate_valid"] = bool(long_rate_valid)
    info["rate_source"] = rate_source
    if elapsed_s < selected_interval_s - 1.0:
        _dual_landing_status = dict(info)
        return None
    _dual_last_fast_taper_ts = now
    if (
        str(_dual_sustain_probe.get("state")) == "failed" and
        str(_dual_sustain_probe.get("reason")) == "temperature_falling_away" and
        float(info.get("error")) > 0.0
    ):
        info["active"] = False
        info["reason"] = "sustain_probe_recovery"
        _dual_landing_status = dict(info)
        _dual_publish_landing_status(info)
        return None
    probe_target = None
    if str(_dual_sustain_probe.get("state")) == "testing":
        probe_target = _dual_sustain_probe.get("candidate")
    if not bool(info.get("active")) and probe_target is None:
        _dual_landing_status = dict(info)
        _dual_publish_landing_status(info)
        return None
    current_total = _dual_current_total_demand(cache, action)
    reference = float(_dual_last_total_target) if isfinite(_dual_last_total_target) else float(current_total)
    governor_cap = float(probe_target) if probe_target is not None else float(info.get("landing_cap"))
    desired = min(float(reference), governor_cap)
    total_step = _dual_float_helper(
        globals().get("DUAL_TOTAL_STEP_HELPER", ""),
        globals().get("DUAL_TOTAL_STEP_DEFAULT", 12.0), 1.0, 40.0, cache,
    )
    if rapid_approach:
        rapid_step_per_unit = _clip(
            float(globals().get("DUAL_RAPID_TAPER_STEP_PER_ACTIVE_UNIT", 5.0)),
            1.0, 20.0,
        )
        rapid_total_step = rapid_step_per_unit * max(1, len(active_indices))
        total_step = min(float(total_step), float(rapid_total_step))
    desired = max(float(reference) - total_step, desired)
    desired = _clip(desired, total_min, total_max)
    if desired >= float(reference) - 0.01:
        _dual_landing_status = dict(info)
        _dual_publish_landing_status(info, reference)
        return float(reference)
    demands = _dual_allocate(desired, action, bounds)
    for idx, u in enumerate(DAIKINS):
        lo, hi, _icing = bounds[idx]
        if _dual_heat_standby_fan_required(u, "heat", action):
            _dual_set_heat_standby_fan(u)
        else:
            _dual_apply_unit_demand(
                u, demands[idx], lo, hi, cache, mode="heat", setpoint=sp,
                downward_min_interval_s=rapid_interval_s if rapid_approach else None,
            )
    _dual_last_total_target = float(desired)
    _dual_discard_episode("predictive_heat_landing")
    _dual_landing_status = dict(info)
    _dual_landing_status["fast_taper_applied"] = True
    _dual_landing_status["step_limited_total"] = float(desired)
    _dual_publish_landing_status(_dual_landing_status, desired)
    demand_sensor = globals().get("DUAL_TOTAL_DEMAND_SENSOR", "sensor.daikin_dual_total_demand")
    try:
        attrs = state.getattr(demand_sensor) or {}
        attrs["landing_active"] = bool(info.get("active"))
        attrs["landing_cap"] = round(float(info.get("landing_cap")), 2)
        attrs["projected_error"] = round(float(info.get("projected_error")), 4)
        attrs["projected_temperature"] = round(float(info.get("projected_temperature")), 3)
        attrs["eta_to_setpoint_minutes"] = (
            round(float(info.get("eta_to_setpoint_minutes")), 2)
            if info.get("eta_to_setpoint_minutes") is not None else None
        )
        attrs["rapid_approach"] = bool(rapid_approach)
        attrs["landing_control_interval_seconds"] = float(selected_interval_s)
        attrs["landing_fast_rate_cph"] = info.get("fast_rate_cph")
        attrs["landing_long_rate_cph"] = info.get("long_rate_cph")
        attrs["landing_rate_source"] = info.get("rate_source")
        attrs["step_limited_total"] = round(float(desired), 2)
        attrs["sustain_probe"] = _dual_probe_snapshot()
        state.set(demand_sensor, value=round(float(desired), 1), **attrs)
    except Exception:
        pass
    return float(desired)


def _dual_load_store():
    global _dual_store_loaded, _dual_sustain_by_ctx, _dual_policy_stats
    global _dual_cop_stats, _dual_fan_rpm_profiles, _dual_cdp_learned_demand
    global _dual_cdp_temperature_curve
    global _separate_area_state, _separate_area_learning
    global _separate_area_learning
    global _dual_active_action_id
    if _dual_store_loaded:
        return
    entity = globals().get("DUAL_ZONE_STORE_ENTITY", "pyscript.daikin_dual_zone_store")
    attrs = state.getattr(entity) or {}

    sustain = attrs.get("sustain_by_ctx")
    if isinstance(sustain, dict):
        clean = {}
        for key, value in sustain.items():
            try:
                fv = float(value)
            except Exception:
                continue
            if isfinite(fv) and fv >= 0.0:
                clean[str(key)] = fv
        _dual_sustain_by_ctx = clean

    stats = attrs.get("policy_stats")
    if isinstance(stats, dict):
        clean_stats = {}
        for ctx, actions in stats.items():
            if not isinstance(actions, dict):
                continue
            clean_actions = {}
            for action_id, raw in actions.items():
                if not isinstance(raw, dict):
                    continue
                try:
                    n = max(0, int(raw.get("n", 0)))
                    mean = float(raw.get("mean", 0.0))
                    m2 = max(0.0, float(raw.get("m2", 0.0)))
                except Exception:
                    continue
                if not (isfinite(mean) and isfinite(m2)):
                    continue
                clean_actions[str(action_id)] = {
                    "n": n,
                    "mean": mean,
                    "m2": m2,
                    "last_score": float(raw.get("last_score", mean)),
                    "last_score_source": str(raw.get("last_score_source", "energy_norm")),
                    "energy_norm_mean": float(raw.get("energy_norm_mean", 0.0)),
                    "comfort_mean": float(raw.get("comfort_mean", 0.0)),
                    "defrost_fraction_mean": float(raw.get("defrost_fraction_mean", 0.0)),
                    "humidity_penalty_mean": float(raw.get("humidity_penalty_mean", 0.0)),
                    "energy_score_n": max(0, int(raw.get("energy_score_n", n))),
                    "energy_score_mean": float(raw.get("energy_score_mean", mean)),
                    "energy_score_m2": max(0.0, float(raw.get("energy_score_m2", m2))),
                    "cop_n": max(0, int(raw.get("cop_n", 0))),
                    "cop_score_n": max(0, int(raw.get("cop_score_n", raw.get("cop_n", 0)))),
                    "cop_score_mean": float(raw.get("cop_score_mean", 0.0)),
                    "cop_score_m2": max(0.0, float(raw.get("cop_score_m2", 0.0))),
                    "combined_cop_mean": float(raw.get("combined_cop_mean", 0.0)),
                    "combined_cop_m2": max(0.0, float(raw.get("combined_cop_m2", 0.0))),
                    "cop_input_kwh": max(0.0, float(raw.get("cop_input_kwh", 0.0))),
                    "estimated_thermal_kwh": max(0.0, float(raw.get("estimated_thermal_kwh", 0.0))),
                }
            if clean_actions:
                clean_stats[str(ctx)] = clean_actions
        _dual_policy_stats = clean_stats

    cop_stats = attrs.get("cop_stats")
    if isinstance(cop_stats, dict):
        clean_cop = {}
        for key, raw in cop_stats.items():
            if not isinstance(raw, dict):
                continue
            try:
                seconds = max(0.0, float(raw.get("seconds", 0.0)))
                input_kwh = max(0.0, float(raw.get("input_kwh", 0.0)))
                thermal_kwh = max(0.0, float(raw.get("thermal_kwh", 0.0)))
                combined_cop = thermal_kwh / input_kwh if input_kwh > 1e-9 else 0.0
                per_unit = {}
                for name, unit in (raw.get("per_unit") or {}).items():
                    unit_input = max(0.0, float(unit.get("input_kwh", 0.0)))
                    unit_thermal = max(0.0, float(unit.get("thermal_kwh", 0.0)))
                    unit_seconds = max(0.0, float(unit.get("seconds", 0.0)))
                    per_unit[str(name)] = {
                        "samples": max(0, int(unit.get("samples", 0))),
                        "seconds": unit_seconds,
                        "input_kwh": unit_input,
                        "thermal_kwh": unit_thermal,
                        "cop": unit_thermal / unit_input if unit_input > 1e-9 else 0.0,
                        "average_input_w": (
                            unit_input * 3600000.0 / unit_seconds
                            if unit_seconds > 0.0 else 0.0
                        ),
                    }
                clean_cop[str(key)] = {
                    "samples": max(0, int(raw.get("samples", 0))),
                    "seconds": seconds,
                    "input_kwh": input_kwh,
                    "thermal_kwh": thermal_kwh,
                    "combined_cop": combined_cop,
                    "average_input_w": (
                        input_kwh * 3600000.0 / seconds if seconds > 0.0 else 0.0
                    ),
                    "mode": str(raw.get("mode", "heat")),
                    "action": str(raw.get("action", "none")),
                    "outdoor_bucket": str(raw.get("outdoor_bucket", "na")),
                    "demands": dict(raw.get("demands") or {}),
                    "per_unit": per_unit,
                    "last_seen": max(0.0, float(raw.get("last_seen", 0.0))),
                }
            except Exception:
                continue
        _dual_cop_stats = clean_cop
        _dual_prune_cop_stats()

    profiles = attrs.get("fan_rpm_profiles")
    if isinstance(profiles, dict):
        clean_profiles = {}
        for name, profile in profiles.items():
            if not isinstance(profile, dict):
                continue
            clean_profile = {}
            for mode, record in profile.items():
                if not isinstance(record, dict):
                    continue
                try:
                    rpm = float(record.get("rpm"))
                except Exception:
                    continue
                if isfinite(rpm) and rpm > 0.0:
                    clean_record = dict(record)
                    clean_record["rpm"] = round(rpm, 2)
                    clean_profile[str(mode)] = clean_record
            if clean_profile:
                clean_profiles[str(name)] = clean_profile
        _dual_fan_rpm_profiles = clean_profiles

    learned_cdp = attrs.get("cdp_learned_release_demand")
    if isinstance(learned_cdp, dict):
        clean_cdp = {}
        for name, value in learned_cdp.items():
            try:
                demand = float(value)
            except Exception:
                continue
            if isfinite(demand) and 30.0 <= demand <= 60.0:
                clean_cdp[str(name)] = demand
        _dual_cdp_learned_demand = clean_cdp

    stored_curve = attrs.get("cdp_temperature_curve")
    if isinstance(stored_curve, dict):
        clean_curve = {}
        for name, fans in stored_curve.items():
            if not isinstance(fans, dict):
                continue
            clean_fans = {}
            for fan, buckets in fans.items():
                if not isinstance(buckets, dict):
                    continue
                clean_buckets = {}
                for bucket, raw in buckets.items():
                    if not isinstance(raw, dict):
                        continue
                    try:
                        edge = float(bucket)
                        sufficient = float(raw.get("sufficient_demand"))
                        insufficient = raw.get("insufficient_demand")
                        insufficient = float(insufficient) if insufficient is not None else None
                    except Exception:
                        continue
                    if not (isfinite(edge) and isfinite(sufficient)):
                        continue
                    record = dict(raw)
                    record["sufficient_demand"] = sufficient
                    record["insufficient_demand"] = insufficient
                    clean_buckets[_dual_cdp_bucket_key(edge)] = record
                if clean_buckets:
                    clean_fans[str(fan)] = clean_buckets
            if clean_fans:
                clean_curve[str(name)] = clean_fans
        _dual_cdp_temperature_curve = clean_curve

    stored_separate_learning = attrs.get("separate_area_learning")
    if isinstance(stored_separate_learning, dict):
        _separate_area_learning = {
            str(key): dict(value) for key, value in stored_separate_learning.items()
            if isinstance(value, dict)
        }

    stored_action = attrs.get("active_action")
    if stored_action and _dual_action_by_id(stored_action):
        _dual_active_action_id = str(stored_action)
    _dual_store_loaded = True


def _dual_save_store(force=False):
    global _dual_last_store_save_ts, _dual_last_store_payload
    now = time.time()
    min_interval = float(globals().get("DUAL_STORE_SAVE_MIN_INTERVAL_S", 1800.0))
    if not force and now - float(_dual_last_store_save_ts) < min_interval:
        return

    sustain_clean = {}
    for key, value in _dual_sustain_by_ctx.items():
        try:
            fv = float(value)
        except Exception:
            continue
        if isfinite(fv):
            sustain_clean[str(key)] = round(max(0.0, fv), 2)

    stats_clean = {}
    for ctx, actions in _dual_policy_stats.items():
        out_actions = {}
        for action_id, raw in actions.items():
            try:
                out_actions[str(action_id)] = {
                    "n": int(raw.get("n", 0)),
                    "mean": round(float(raw.get("mean", 0.0)), 6),
                    "m2": round(float(raw.get("m2", 0.0)), 6),
                    "last_score": round(float(raw.get("last_score", 0.0)), 6),
                    "last_score_source": str(raw.get("last_score_source", "energy_norm")),
                    "energy_norm_mean": round(float(raw.get("energy_norm_mean", 0.0)), 6),
                    "comfort_mean": round(float(raw.get("comfort_mean", 0.0)), 6),
                    "defrost_fraction_mean": round(float(raw.get("defrost_fraction_mean", 0.0)), 6),
                    "humidity_penalty_mean": round(float(raw.get("humidity_penalty_mean", 0.0)), 6),
                    "energy_score_n": int(raw.get("energy_score_n", 0)),
                    "energy_score_mean": round(float(raw.get("energy_score_mean", 0.0)), 6),
                    "energy_score_m2": round(float(raw.get("energy_score_m2", 0.0)), 6),
                    "cop_n": int(raw.get("cop_n", 0)),
                    "cop_score_n": int(raw.get("cop_score_n", raw.get("cop_n", 0))),
                    "cop_score_mean": round(float(raw.get("cop_score_mean", 0.0)), 6),
                    "cop_score_m2": round(float(raw.get("cop_score_m2", 0.0)), 6),
                    "combined_cop_mean": round(float(raw.get("combined_cop_mean", 0.0)), 6),
                    "combined_cop_m2": round(float(raw.get("combined_cop_m2", 0.0)), 6),
                    "cop_input_kwh": round(float(raw.get("cop_input_kwh", 0.0)), 6),
                    "estimated_thermal_kwh": round(float(raw.get("estimated_thermal_kwh", 0.0)), 6),
                }
            except Exception:
                continue
        if out_actions:
            stats_clean[str(ctx)] = out_actions

    cop_clean = {}
    for key, raw in _dual_cop_stats.items():
        public = _dual_cop_stat_public(raw)
        if public is None:
            continue
        per_unit = {}
        for name, unit in (raw.get("per_unit") or {}).items():
            try:
                per_unit[str(name)] = {
                    "samples": int(unit.get("samples", 0)),
                    "seconds": round(float(unit.get("seconds", 0.0)), 1),
                    "input_kwh": round(float(unit.get("input_kwh", 0.0)), 6),
                    "thermal_kwh": round(float(unit.get("thermal_kwh", 0.0)), 6),
                }
            except Exception:
                continue
        cop_clean[str(key)] = {
            "samples": int(raw.get("samples", 0)),
            "seconds": round(float(raw.get("seconds", 0.0)), 1),
            "input_kwh": round(float(raw.get("input_kwh", 0.0)), 6),
            "thermal_kwh": round(float(raw.get("thermal_kwh", 0.0)), 6),
            "mode": str(raw.get("mode", "heat")),
            "action": str(raw.get("action", "none")),
            "outdoor_bucket": str(raw.get("outdoor_bucket", "na")),
            "demands": dict(raw.get("demands") or {}),
            "per_unit": per_unit,
            "last_seen": round(float(raw.get("last_seen", 0.0)), 1),
        }

    payload_obj = {
        "sustain_by_ctx": sustain_clean,
        "policy_stats": stats_clean,
        "cop_stats": cop_clean,
        "active_action": _dual_active_action_id,
        "fan_rpm_profiles": _dual_fan_profile_snapshot(),
        "cdp_learned_release_demand": dict(_dual_cdp_learned_demand),
        "cdp_temperature_curve": _dual_cdp_curve_snapshot(),
        "separate_area_learning": dict(_separate_area_learning),
    }
    try:
        payload = json.dumps(payload_obj, sort_keys=True, separators=(",", ":"))
    except Exception:
        payload = None
    if not force and payload is not None and payload == _dual_last_store_payload:
        _dual_last_store_save_ts = now
        return

    entity = globals().get("DUAL_ZONE_STORE_ENTITY", "pyscript.daikin_dual_zone_store")
    try:
        state.set(
            entity,
            value=now,
            sustain_by_ctx=sustain_clean,
            policy_stats=stats_clean,
            cop_stats=cop_clean,
            active_action=_dual_active_action_id,
            fan_rpm_profiles=payload_obj["fan_rpm_profiles"],
            cdp_learned_release_demand=payload_obj["cdp_learned_release_demand"],
            cdp_temperature_curve=payload_obj["cdp_temperature_curve"],
            separate_area_learning=payload_obj["separate_area_learning"],
            schema_version=8,
        )
        _dual_last_store_save_ts = now
        _dual_last_store_payload = payload
    except Exception as e:
        log.error("Daikin dual: failed to save optimizer store: %s", e)


def _dual_total_power(cache=None, action=None):
    """Return total power, validity for expected units, and per-unit watts.

    When an allocation is supplied, only its expected running units are
    mandatory. A fresh standby reading is still included, but an intentionally
    off unit whose zero-watt sensor no longer updates cannot invalidate a
    one-unit efficiency episode.
    """
    cache = cache or _TickCache()
    required_names = {}
    if action is not None:
        for expected in _dual_expected_learning_units(action):
            required_names[str(expected.get("name", "daikin?"))] = True
        for u in DAIKINS:
            if _dual_heat_standby_fan_required(
                u, str(_dual_active_mode or "off"), action
            ):
                # The circulation fan is part of the Daikin2-only policy, so
                # its electricity must be present for a valid energy score.
                required_names[str(u.get("name", "daikin?"))] = True
    combined_ent = globals().get("DUAL_TOTAL_POWER_SENSOR")
    if combined_ent:
        combined_info = _sensor_freshness(
            "power_combined", combined_ent, globals().get("DUAL_POWER_MAX_AGE_S", 120.0), cache, True
        )
        combined = combined_info.get("value") if combined_info.get("fresh") else float("nan")
        try:
            combined *= float(globals().get("DUAL_TOTAL_POWER_SENSOR_SCALE", 1.0))
        except Exception:
            combined = float("nan")
        if isfinite(combined) and combined >= 0.0:
            per_unit = {}
            for u in DAIKINS:
                name = str(u.get("name", "daikin?"))
                ent = u.get("POWER_SENSOR")
                if ent:
                    info = _sensor_freshness(
                        "power_%s" % name, ent, globals().get("DUAL_POWER_MAX_AGE_S", 120.0), cache, True
                    )
                    value = info.get("value") if info.get("fresh") else float("nan")
                else:
                    value = float("nan")
                try:
                    value *= float(u.get("POWER_SENSOR_SCALE", 1.0))
                except Exception:
                    value = float("nan")
                per_unit[name] = round(float(value), 1) if isfinite(value) and value >= 0.0 else None
            return _clip(float(combined), 0.0, 100000.0), True, per_unit

    total = 0.0
    valid = True
    per_unit = {}
    for u in DAIKINS:
        name = str(u.get("name", "daikin?"))
        required = bool(required_names.get(name)) if action is not None else True
        ent = u.get("POWER_SENSOR")
        if not ent:
            if required:
                valid = False
            per_unit[name] = None
            continue
        info = _sensor_freshness(
            "power_%s" % name, ent, globals().get("DUAL_POWER_MAX_AGE_S", 120.0), cache, True
        )
        value = info.get("value") if info.get("fresh") else float("nan")
        try:
            scale = float(u.get("POWER_SENSOR_SCALE", 1.0))
        except Exception:
            scale = 1.0
        value = float(value) * scale if isfinite(value) and isfinite(scale) else float("nan")
        if not isfinite(value) or value < 0.0:
            if required:
                valid = False
            per_unit[name] = None
            continue
        value = _clip(value, 0.0, 50000.0)
        per_unit[name] = float(value)
        total += float(value)
    return float(total), bool(valid), per_unit


def _dual_cop_mode_supported(mode):
    configured = globals().get("DUAL_COP_MODES", ("heat",))
    if isinstance(configured, str):
        configured = [configured]
    modes = []
    for value in configured or []:
        modes.append(str(value).strip().lower())
    return str(mode or "").strip().lower() in modes


def _dual_cop_context(tout, mode, action, demands):
    """Return a matched mode/outdoor/action/actual-Demand context key."""
    outdoor_step = max(
        0.5, float(globals().get("DUAL_COP_CONTEXT_OUTDOOR_STEP_C", 2.0))
    )
    demand_step = max(
        1.0, float(globals().get("DUAL_COP_CONTEXT_DEMAND_STEP", 5.0))
    )
    outdoor_text = "na"
    if isfinite(tout):
        outdoor_text = "%+.1f" % (round(float(tout) / outdoor_step) * outdoor_step)
    action_id = str((action or {}).get("id", "none"))
    load_parts = []
    demands = demands or {}
    for u in DAIKINS:
        name = str(u.get("name", "daikin?"))
        value = demands.get(name)
        if value is None or not isfinite(value):
            load_parts.append("%s=off" % name)
        else:
            snapped = round(float(value) / demand_step) * demand_step
            load_parts.append("%s=%.0f" % (name, snapped))
    return "%s|%s|%s|%s" % (
        str(mode or "off"), outdoor_text, action_id, ",".join(load_parts)
    )


def _dual_cop_snapshot(cache=None, action=None, mode=None, tout=None):
    """Read per-unit COP and return a power-weighted combined COP snapshot."""
    cache = cache or _TickCache()
    mode = str(mode or _dual_active_mode or "off").strip().lower()
    action = action or _dual_action_by_id(_dual_active_action_id)
    expected = _dual_expected_learning_units(action) if mode in ("heat", "cool", "dry") else []
    expected_names = []
    expected_lookup = {}
    for u in expected:
        name = str(u.get("name", "daikin?"))
        expected_names.append(name)
        expected_lookup[name] = True

    min_cop = float(globals().get("DUAL_COP_MIN_VALUE", 0.50))
    max_cop = float(globals().get("DUAL_COP_MAX_VALUE", 12.0))
    min_power = max(0.0, float(globals().get("DUAL_COP_MIN_UNIT_POWER_W", 150.0)))
    max_age = float(globals().get("DUAL_COP_MAX_AGE_S", 120.0))
    per_unit_cop = {}
    per_unit_power = {}
    per_unit_thermal = {}
    demands = {}
    invalid_reasons = []
    input_w = 0.0
    thermal_w = 0.0

    if not _dual_cop_mode_supported(mode):
        invalid_reasons.append("unsupported_mode:%s" % mode)
    if not expected_names:
        invalid_reasons.append("no_expected_running_units")

    for u in DAIKINS:
        name = str(u.get("name", "daikin?"))
        standby_fan = _dual_heat_standby_fan_required(u, mode, action)
        if not expected_lookup.get(name) and not standby_fan:
            per_unit_cop[name] = None
            per_unit_power[name] = None
            per_unit_thermal[name] = None
            demands[name] = None
            continue

        demands[name] = (
            None if standby_fan
            else _dual_current_effective_demand(u, cache)
        )
        cop_ent = u.get("COP_SENSOR")
        power_ent = u.get("POWER_SENSOR")
        cop_value = float("nan")
        power_value = float("nan")
        if cop_ent:
            cop_info = _sensor_freshness(
                "cop_optional_%s" % name, cop_ent, max_age, cache, True
            )
            if cop_info.get("fresh") and cop_info.get("value") is not None:
                try:
                    cop_value = float(cop_info.get("value"))
                except Exception:
                    cop_value = float("nan")
        elif not standby_fan:
            invalid_reasons.append("%s_cop_sensor_not_configured" % name)

        if power_ent:
            power_info = _sensor_freshness(
                "power_%s" % name, power_ent,
                globals().get("DUAL_POWER_MAX_AGE_S", 120.0), cache, True,
            )
            if power_info.get("fresh") and power_info.get("value") is not None:
                try:
                    power_value = (
                        float(power_info.get("value")) *
                        float(u.get("POWER_SENSOR_SCALE", 1.0))
                    )
                except Exception:
                    power_value = float("nan")
        else:
            invalid_reasons.append("%s_power_sensor_not_configured" % name)

        cop_ok = bool(
            isfinite(cop_value) and min_cop <= float(cop_value) <= max_cop
        )
        power_ok = bool(
            isfinite(power_value) and float(power_value) >= (
                0.0 if standby_fan else min_power
            )
        )
        if standby_fan:
            # Circulation consumes electricity but supplies no compressor heat.
            # Include it in the system COP denominator without requiring COP
            # from a non-heating unit.
            if not power_ok:
                invalid_reasons.append(
                    "%s_standby_fan_power_invalid_or_stale" % name
                )
            per_unit_cop[name] = None
            per_unit_power[name] = (
                round(float(power_value), 1) if power_ok else None
            )
            per_unit_thermal[name] = 0.0 if power_ok else None
            if power_ok:
                input_w += float(power_value)
            continue
        if not cop_ok:
            invalid_reasons.append("%s_cop_invalid_or_stale" % name)
        if not power_ok:
            invalid_reasons.append("%s_power_below_min_or_invalid" % name)

        per_unit_cop[name] = round(float(cop_value), 4) if cop_ok else None
        per_unit_power[name] = round(float(power_value), 1) if power_ok else None
        if cop_ok and power_ok:
            unit_thermal = float(cop_value) * float(power_value)
            per_unit_thermal[name] = round(unit_thermal, 1)
            input_w += float(power_value)
            thermal_w += unit_thermal
        else:
            per_unit_thermal[name] = None

    valid = bool(
        _dual_cop_mode_supported(mode) and expected_names and
        not invalid_reasons and input_w > 0.0 and isfinite(thermal_w)
    )
    combined = thermal_w / input_w if valid and input_w > 0.0 else None
    if combined is not None and not (
        isfinite(combined) and min_cop <= float(combined) <= max_cop
    ):
        invalid_reasons.append("combined_cop_out_of_range")
        valid = False
        combined = None
    if tout is None or not isfinite(tout):
        tout = _dual_outdoor_temperature(cache)
    context = _dual_cop_context(tout, mode, action, demands)
    return {
        "valid": bool(valid),
        "reason": "valid" if valid else (
            str(invalid_reasons[0]) if invalid_reasons else "invalid"
        ),
        "invalid_reasons": list(invalid_reasons),
        "mode": mode,
        "action": str((action or {}).get("id", "none")),
        "context": context,
        "outdoor": float(tout) if isfinite(tout) else None,
        "expected_units": expected_names,
        "demands": demands,
        "per_unit_cop": per_unit_cop,
        "per_unit_power_w": per_unit_power,
        "per_unit_thermal_w": per_unit_thermal,
        "active_input_power_w": round(float(input_w), 2) if input_w > 0.0 else None,
        "estimated_thermal_output_w": round(float(thermal_w), 2) if thermal_w > 0.0 else None,
        "combined_cop": round(float(combined), 5) if combined is not None else None,
    }


def _dual_prune_cop_stats():
    max_contexts = max(20, int(globals().get("DUAL_COP_MAX_CONTEXTS", 300)))
    if len(_dual_cop_stats) <= max_contexts:
        return
    ordered = []
    for key, raw in _dual_cop_stats.items():
        try:
            last_seen = float(raw.get("last_seen", 0.0))
        except Exception:
            last_seen = 0.0
        ordered.append((last_seen, str(key)))
    ordered.sort(key=lambda item: item[0])
    remove_n = len(ordered) - max_contexts
    for _last_seen, key in ordered[:remove_n]:
        _dual_cop_stats.pop(key, None)


def _dual_update_cop_stats(snapshot, dt_s):
    """Accumulate energy-weighted COP for one exact operating context."""
    global _dual_cop_stats
    if not snapshot or not snapshot.get("valid"):
        return None
    dt_s = _clip(float(dt_s), 0.0, 120.0)
    if dt_s <= 0.0:
        return None
    input_w = snapshot.get("active_input_power_w")
    thermal_w = snapshot.get("estimated_thermal_output_w")
    if input_w is None or thermal_w is None:
        return None
    input_w = float(input_w)
    thermal_w = float(thermal_w)
    if not (isfinite(input_w) and isfinite(thermal_w) and input_w > 0.0):
        return None

    key = str(snapshot.get("context"))
    stat = _dual_cop_stats.setdefault(key, {
        "samples": 0,
        "seconds": 0.0,
        "input_kwh": 0.0,
        "thermal_kwh": 0.0,
        "combined_cop": 0.0,
        "average_input_w": 0.0,
        "mode": str(snapshot.get("mode")),
        "action": str(snapshot.get("action")),
        "outdoor_bucket": str(key).split("|")[1] if "|" in key else "na",
        "demands": dict(snapshot.get("demands") or {}),
        "per_unit": {},
        "last_seen": 0.0,
    })
    stat["samples"] = int(stat.get("samples", 0)) + 1
    stat["seconds"] = float(stat.get("seconds", 0.0)) + dt_s
    stat["input_kwh"] = float(stat.get("input_kwh", 0.0)) + input_w * dt_s / 3600000.0
    stat["thermal_kwh"] = float(stat.get("thermal_kwh", 0.0)) + thermal_w * dt_s / 3600000.0
    stat["combined_cop"] = (
        float(stat["thermal_kwh"]) / float(stat["input_kwh"])
        if float(stat["input_kwh"]) > 1e-9 else 0.0
    )
    stat["average_input_w"] = (
        float(stat["input_kwh"]) * 3600000.0 / float(stat["seconds"])
        if float(stat["seconds"]) > 0.0 else 0.0
    )
    stat["demands"] = dict(snapshot.get("demands") or {})
    stat["last_seen"] = time.time()

    for name in snapshot.get("expected_units") or []:
        cop_value = (snapshot.get("per_unit_cop") or {}).get(name)
        power_value = (snapshot.get("per_unit_power_w") or {}).get(name)
        if cop_value is None or power_value is None:
            continue
        unit_stat = stat["per_unit"].setdefault(str(name), {
            "samples": 0, "seconds": 0.0, "input_kwh": 0.0,
            "thermal_kwh": 0.0, "cop": 0.0, "average_input_w": 0.0,
        })
        unit_input = float(power_value) * dt_s / 3600000.0
        unit_thermal = float(cop_value) * float(power_value) * dt_s / 3600000.0
        unit_stat["samples"] = int(unit_stat.get("samples", 0)) + 1
        unit_stat["seconds"] = float(unit_stat.get("seconds", 0.0)) + dt_s
        unit_stat["input_kwh"] = float(unit_stat.get("input_kwh", 0.0)) + unit_input
        unit_stat["thermal_kwh"] = float(unit_stat.get("thermal_kwh", 0.0)) + unit_thermal
        unit_stat["cop"] = (
            float(unit_stat["thermal_kwh"]) / float(unit_stat["input_kwh"])
            if float(unit_stat["input_kwh"]) > 1e-9 else 0.0
        )
        unit_stat["average_input_w"] = (
            float(unit_stat["input_kwh"]) * 3600000.0 / float(unit_stat["seconds"])
            if float(unit_stat["seconds"]) > 0.0 else 0.0
        )

    _dual_prune_cop_stats()
    _dual_save_store(force=False)
    return stat


def _dual_cop_stat_public(raw):
    if not isinstance(raw, dict):
        return None
    per_unit = {}
    for name, unit in (raw.get("per_unit") or {}).items():
        try:
            per_unit[str(name)] = {
                "samples": int(unit.get("samples", 0)),
                "minutes": round(float(unit.get("seconds", 0.0)) / 60.0, 1),
                "cop": round(float(unit.get("cop", 0.0)), 3),
                "average_input_w": round(float(unit.get("average_input_w", 0.0)), 1),
                "input_kwh": round(float(unit.get("input_kwh", 0.0)), 4),
                "estimated_thermal_kwh": round(float(unit.get("thermal_kwh", 0.0)), 4),
            }
        except Exception:
            continue
    try:
        return {
            "samples": int(raw.get("samples", 0)),
            "minutes": round(float(raw.get("seconds", 0.0)) / 60.0, 1),
            "combined_cop": round(float(raw.get("combined_cop", 0.0)), 3),
            "average_input_w": round(float(raw.get("average_input_w", 0.0)), 1),
            "input_kwh": round(float(raw.get("input_kwh", 0.0)), 4),
            "estimated_thermal_kwh": round(float(raw.get("thermal_kwh", 0.0)), 4),
            "mode": raw.get("mode"),
            "action": raw.get("action"),
            "outdoor_bucket": raw.get("outdoor_bucket"),
            "demands": dict(raw.get("demands") or {}),
            "per_unit": per_unit,
            "last_seen": round(float(raw.get("last_seen", 0.0)), 1),
        }
    except Exception:
        return None


def _dual_cop_comparisons(snapshot):
    """Return mature contexts for the same mode and outdoor bucket, best first."""
    context = str((snapshot or {}).get("context") or "")
    parts = context.split("|")
    if len(parts) < 2:
        return []
    mode = parts[0]
    outdoor_bucket = parts[1]
    minimum_s = max(
        0.0, float(globals().get("DUAL_COP_MIN_CONTEXT_SECONDS", 600.0))
    )
    ranked = []
    for key, raw in _dual_cop_stats.items():
        key_parts = str(key).split("|")
        if len(key_parts) < 2 or key_parts[0] != mode or key_parts[1] != outdoor_bucket:
            continue
        try:
            if float(raw.get("seconds", 0.0)) < minimum_s:
                continue
            cop = float(raw.get("combined_cop", 0.0))
        except Exception:
            continue
        if isfinite(cop) and cop > 0.0:
            ranked.append((cop, str(key), raw))
    ranked.sort(key=lambda item: item[0], reverse=True)
    result = []
    for cop, key, raw in ranked[:12]:
        result.append({
            "context": key,
            "combined_cop": round(float(cop), 3),
            "action": raw.get("action"),
            "demands": dict(raw.get("demands") or {}),
            "minutes": round(float(raw.get("seconds", 0.0)) / 60.0, 1),
            "average_input_w": round(float(raw.get("average_input_w", 0.0)), 1),
        })
    return result


def _dual_publish_cop(snapshot, learning_allowed=False, learning_reason=None):
    """Publish valid live COP independently from optimizer learning eligibility."""
    global _dual_current_cop_snapshot
    snapshot = dict(snapshot or {})
    raw_valid = bool(snapshot.get("valid"))
    learning_eligible = bool(raw_valid and learning_allowed)
    if raw_valid and not learning_allowed:
        status_reason = "live_valid_learning_gate:%s" % str(
            learning_reason or "warming_up"
        )
    else:
        status_reason = str(snapshot.get("reason") or "invalid")
    current = _dual_cop_stats.get(str(snapshot.get("context")))
    public_current = _dual_cop_stat_public(current)
    comparisons = _dual_cop_comparisons(snapshot)
    snapshot["learning_allowed"] = bool(learning_allowed)
    # ``usable`` retains its historical optimizer meaning. The new
    # ``live_value_available`` flag explicitly separates sensor visibility
    # from whether the sample may update learned efficiency statistics.
    snapshot["usable"] = bool(learning_eligible)
    snapshot["learning_eligible"] = bool(learning_eligible)
    snapshot["live_value_available"] = bool(raw_valid)
    snapshot["status_reason"] = status_reason
    _dual_current_cop_snapshot = snapshot

    sensor = globals().get(
        "DUAL_COP_EFFICIENCY_SENSOR", "sensor.daikin_dual_cop_efficiency"
    )
    try:
        state.set(
            sensor,
            value=(
                round(float(snapshot.get("combined_cop")), 2)
                if raw_valid and snapshot.get("combined_cop") is not None
                else "unavailable"
            ),
            valid=bool(raw_valid),
            raw_cop_valid=bool(raw_valid),
            live_value_available=bool(raw_valid),
            learning_allowed=bool(learning_allowed),
            learning_eligible=bool(learning_eligible),
            learning_block_reason=(
                None if learning_allowed
                else str(learning_reason or "warming_up")
            ),
            reason=status_reason,
            measurement_reason=str(snapshot.get("reason") or "invalid"),
            operating_mode=snapshot.get("mode"),
            action=snapshot.get("action"),
            context=snapshot.get("context"),
            combined_cop=(
                round(float(snapshot.get("combined_cop")), 4)
                if snapshot.get("combined_cop") is not None else None
            ),
            expected_units=list(snapshot.get("expected_units") or []),
            actual_demands=dict(snapshot.get("demands") or {}),
            per_unit_cop=dict(snapshot.get("per_unit_cop") or {}),
            per_unit_power_w=dict(snapshot.get("per_unit_power_w") or {}),
            per_unit_estimated_thermal_w=dict(snapshot.get("per_unit_thermal_w") or {}),
            active_input_power_w=snapshot.get("active_input_power_w"),
            estimated_thermal_output_w=snapshot.get("estimated_thermal_output_w"),
            invalid_reasons=list(snapshot.get("invalid_reasons") or []),
            current_context_stats=public_current,
            comparable_contexts=comparisons,
            minimum_context_minutes=round(
                float(globals().get("DUAL_COP_MIN_CONTEXT_SECONDS", 600.0)) / 60.0, 1
            ),
            configured_units=len(DAIKINS),
        )
    except Exception as e:
        log.debug("Daikin dual: failed to publish COP efficiency state: %s", e)

    demand_sensor = globals().get(
        "DUAL_TOTAL_DEMAND_SENSOR", "sensor.daikin_dual_total_demand"
    )
    try:
        attrs = state.getattr(demand_sensor) or {}
        attrs["combined_cop"] = (
            round(float(snapshot.get("combined_cop")), 4)
            if raw_valid and snapshot.get("combined_cop") is not None else None
        )
        attrs["cop_valid"] = bool(raw_valid)
        attrs["cop_learning_eligible"] = bool(learning_eligible)
        attrs["cop_learning_block_reason"] = (
            None if learning_allowed else str(learning_reason or "warming_up")
        )
        attrs["cop_reason"] = status_reason
        attrs["per_unit_cop"] = dict(snapshot.get("per_unit_cop") or {})
        attrs["cop_context"] = snapshot.get("context")
        state.set(demand_sensor, value=state.get(demand_sensor), **attrs)
    except Exception:
        pass


def _dual_episode_minutes(cache=None):
    return _dual_float_helper(
        globals().get("DUAL_EPISODE_MINUTES_HELPER", ""),
        globals().get("DUAL_EPISODE_MINUTES_DEFAULT", 60.0),
        30.0, 240.0, cache,
    )


def _dual_abort_error(cache=None):
    return _dual_float_helper(
        globals().get("DUAL_ABORT_ERR_HELPER", ""),
        globals().get("DUAL_ABORT_ERR_C_DEFAULT", 0.60),
        0.2, 2.0, cache,
    )


def _dual_start_episode(ctx, action_id, sp, tout, total_target, mode="heat", humidity=None):
    global _dual_episode
    now = time.time()
    _dual_episode = {
        "start": now,
        "ctx": str(ctx),
        "action": str(action_id),
        "mode": str(mode),
        "sp_start": float(sp),
        "tout_start": float(tout),
        "total_target_start": float(total_target),
        "energy_kwh": 0.0,
        "cop_input_kwh": 0.0,
        "estimated_thermal_kwh": 0.0,
        "combined_cop": None,
        "degree_hours": 0.0,
        "comfort_penalty": 0.0,
        "humidity_start": float(humidity) if humidity is not None and isfinite(humidity) else None,
        "humidity_last": float(humidity) if humidity is not None and isfinite(humidity) else None,
        "humidity_penalty": 0.0,
        "defrost_s": 0.0,
        "cycles": 0,
        "samples": 0,
        "valid_power_samples": 0,
        "valid_cop_samples": 0,
        "invalid": False,
        "invalid_reason": None,
    }


def _dual_discard_episode(reason):
    global _dual_episode
    if _dual_episode is not None:
        log.info(
            "Daikin dual optimizer: discarding episode ctx=%s action=%s reason=%s",
            str(_dual_episode.get("ctx")), str(_dual_episode.get("action")), str(reason),
        )
    _dual_episode = None


def _dual_update_mean(old, sample, n_new):
    if n_new <= 1:
        return float(sample)
    return float(old) + (float(sample) - float(old)) / float(n_new)


def _dual_update_online_stat(raw, prefix, sample):
    """Update n/mean/m2 fields named <prefix>_n/_mean/_m2."""
    n_key = "%s_n" % prefix
    mean_key = "%s_mean" % prefix
    m2_key = "%s_m2" % prefix
    old_n = max(0, int(raw.get(n_key, 0)))
    new_n = old_n + 1
    old_mean = float(raw.get(mean_key, 0.0))
    delta = float(sample) - old_mean
    new_mean = old_mean + delta / float(new_n)
    new_m2 = float(raw.get(m2_key, 0.0)) + delta * (float(sample) - new_mean)
    raw[n_key] = new_n
    raw[mean_key] = new_mean
    raw[m2_key] = max(0.0, new_m2)
    return new_n, new_mean


def _dual_finalize_episode(force_discard=False):
    """Finalize the current efficiency episode. Returns its score or None."""
    global _dual_episode, _dual_policy_stats
    ep = _dual_episode
    if ep is None:
        return None
    now = time.time()
    duration_s = max(0.0, now - float(ep.get("start", now)))
    samples = max(0, int(ep.get("samples", 0)))
    valid_samples = max(0, int(ep.get("valid_power_samples", 0)))
    valid_fraction = float(valid_samples) / float(samples) if samples > 0 else 0.0
    valid_cop_samples = max(0, int(ep.get("valid_cop_samples", 0)))
    cop_valid_fraction = (
        float(valid_cop_samples) / float(samples) if samples > 0 else 0.0
    )
    min_duration_s = float(_dual_episode_minutes()) * 60.0
    required_fraction = float(globals().get("DUAL_MIN_VALID_POWER_FRACTION", 0.80))

    invalid = bool(force_discard or ep.get("invalid"))
    invalid = invalid or duration_s < min_duration_s
    invalid = invalid or valid_fraction < required_fraction
    invalid = invalid or float(ep.get("degree_hours", 0.0)) <= 0.05

    if invalid:
        log.info(
            "Daikin dual optimizer: episode not scored ctx=%s action=%s duration=%.0fs valid_power=%.0f%% reason=%s",
            str(ep.get("ctx")), str(ep.get("action")), duration_s,
            valid_fraction * 100.0, str(ep.get("invalid_reason")),
        )
        _dual_episode = None
        return None

    duration_h = max(duration_s / 3600.0, 1e-6)
    operating_mode = str(ep.get("mode", "heat"))
    energy_kwh = float(ep.get("energy_kwh", 0.0))
    degree_hours = float(ep.get("degree_hours", 0.0))
    energy_norm = energy_kwh / degree_hours
    cop_input_kwh = max(0.0, float(ep.get("cop_input_kwh", 0.0)))
    thermal_kwh = max(0.0, float(ep.get("estimated_thermal_kwh", 0.0)))
    episode_cop = thermal_kwh / cop_input_kwh if cop_input_kwh > 1e-9 else None
    cop_usable = bool(
        _dual_cop_mode_supported(operating_mode) and
        episode_cop is not None and isfinite(episode_cop) and
        float(globals().get("DUAL_COP_MIN_VALUE", 0.50)) <= float(episode_cop) <=
        float(globals().get("DUAL_COP_MAX_VALUE", 12.0)) and
        cop_valid_fraction >= float(globals().get("DUAL_COP_MIN_VALID_FRACTION", 0.80))
    )
    ep["combined_cop"] = float(episode_cop) if cop_usable else None
    comfort_rate = float(ep.get("comfort_penalty", 0.0)) / duration_h
    humidity_penalty_rate = float(ep.get("humidity_penalty", 0.0)) / duration_h
    defrost_fraction = float(ep.get("defrost_s", 0.0)) / max(duration_s, 1.0)
    cycles_per_h = float(ep.get("cycles", 0)) / duration_h
    penalty = (
        float(globals().get("DUAL_SCORE_COMFORT_WEIGHT", 2.0)) * comfort_rate +
        float(globals().get("DUAL_SCORE_HUMIDITY_WEIGHT", 0.05)) * humidity_penalty_rate +
        (float(globals().get("DUAL_SCORE_DEFROST_WEIGHT", 0.20)) * defrost_fraction if operating_mode == "heat" else 0.0) +
        float(globals().get("DUAL_SCORE_CYCLE_WEIGHT", 0.01)) * cycles_per_h
    )
    energy_score = energy_norm + penalty
    use_cop = bool(cop_usable and globals().get("DUAL_COP_USE_FOR_SCORING", True))
    cop_score = (1.0 / float(episode_cop) + penalty) if cop_usable else None
    score = float(cop_score) if use_cop else float(energy_score)
    score_source = "cop" if use_cop else "energy_norm"

    ctx = str(ep.get("ctx"))
    action_id = str(ep.get("action"))
    ctx_stats = _dual_policy_stats.setdefault(ctx, {})
    stat = ctx_stats.setdefault(action_id, {
        "n": 0, "mean": 0.0, "m2": 0.0, "last_score": 0.0,
        "energy_norm_mean": 0.0, "comfort_mean": 0.0, "defrost_fraction_mean": 0.0,
        "humidity_penalty_mean": 0.0, "last_score_source": "energy_norm",
        "energy_score_n": 0, "energy_score_mean": 0.0, "energy_score_m2": 0.0,
        "cop_n": 0, "cop_score_n": 0,
        "cop_score_mean": 0.0, "cop_score_m2": 0.0,
        "combined_cop_mean": 0.0, "combined_cop_m2": 0.0,
        "cop_input_kwh": 0.0, "estimated_thermal_kwh": 0.0,
    })
    old_n = int(stat.get("n", 0))
    new_n = old_n + 1
    old_mean = float(stat.get("mean", 0.0))
    delta = score - old_mean
    new_mean = old_mean + delta / float(new_n)
    new_m2 = float(stat.get("m2", 0.0)) + delta * (score - new_mean)
    stat["n"] = new_n
    stat["mean"] = new_mean
    stat["m2"] = max(0.0, new_m2)
    stat["last_score"] = score
    stat["last_score_source"] = score_source
    _dual_update_online_stat(stat, "energy_score", energy_score)
    if cop_usable:
        _dual_update_online_stat(stat, "cop_score", cop_score)
        # cop_score_n and combined_cop_n intentionally share the same episode
        # count; store combined COP under the public `cop_n` field.
        cop_n = max(0, int(stat.get("cop_n", 0))) + 1
        old_cop_mean = float(stat.get("combined_cop_mean", 0.0))
        cop_delta = float(episode_cop) - old_cop_mean
        new_cop_mean = old_cop_mean + cop_delta / float(cop_n)
        stat["combined_cop_m2"] = max(
            0.0,
            float(stat.get("combined_cop_m2", 0.0)) +
            cop_delta * (float(episode_cop) - new_cop_mean),
        )
        stat["combined_cop_mean"] = new_cop_mean
        stat["cop_n"] = cop_n
        stat["cop_input_kwh"] = (
            float(stat.get("cop_input_kwh", 0.0)) + cop_input_kwh
        )
        stat["estimated_thermal_kwh"] = (
            float(stat.get("estimated_thermal_kwh", 0.0)) + thermal_kwh
        )
    stat["energy_norm_mean"] = _dual_update_mean(stat.get("energy_norm_mean", 0.0), energy_norm, new_n)
    stat["comfort_mean"] = _dual_update_mean(stat.get("comfort_mean", 0.0), comfort_rate, new_n)
    stat["defrost_fraction_mean"] = _dual_update_mean(stat.get("defrost_fraction_mean", 0.0), defrost_fraction, new_n)
    stat["humidity_penalty_mean"] = _dual_update_mean(stat.get("humidity_penalty_mean", 0.0), humidity_penalty_rate, new_n)

    log.info(
        "Daikin dual optimizer: scored mode=%s ctx=%s action=%s score=%.5f source=%s COP=%s energy=%.3fkWh norm=%.5f comfort=%.5f humidity=%.5f defrost=%.1f%% n=%d",
        operating_mode, ctx, action_id, score, score_source,
        round(float(episode_cop), 3) if cop_usable else None,
        energy_kwh, energy_norm, comfort_rate, humidity_penalty_rate,
        defrost_fraction * 100.0, new_n,
    )
    _dual_episode = None
    _dual_save_store(force=True)
    return float(score)


def _dual_optimizer_sample(system_freeze_active=False):
    """One-minute mode-separated energy, comfort, humidity and cycle sample."""
    global _dual_optimizer_last_sample_ts, _dual_optimizer_prev_power_w
    global _dual_optimizer_prev_cop_power_w, _dual_optimizer_prev_thermal_w
    global _dual_optimizer_prev_heating
    now = time.time()
    cache = _TickCache()
    if not _normal_optimizer_inputs_available():
        _dual_discard_episode("safety_or_power_input_block")
    _requested, operating_mode, _summer, _block = _effective_hvac_mode(cache)
    tin = _dual_zone_temperature(cache)
    tout = _dual_outdoor_temperature(cache)
    humidity_raw = _dual_humidity(cache, filtered=False)
    if isfinite(humidity_raw):
        _dual_add_humidity_sample(now, humidity_raw)
    humidity = _dual_humidity(cache, filtered=True)
    humidity_target = _dual_supervisor_limits(cache)["humidity_target"]
    sp, _min_guard, _max_guard, deadband = _dual_effective_setpoint(
        cache, write_helpers=False, mode=operating_mode
    )

    active_action = _dual_action_by_id(_dual_active_action_id)
    learning = _dual_update_learning_gate(
        operating_mode,
        cache=cache,
        action=active_action,
        now=now,
        system_freeze_active=system_freeze_active,
        collect_sample=True,
    )

    total_power_w, power_valid, _per_unit = _dual_total_power(cache, active_action)
    last_ts = float(_dual_optimizer_last_sample_ts or 0.0)
    dt_s = now - last_ts if last_ts > 0.0 else 0.0
    dt_s = _clip(dt_s, 0.0, 120.0)
    _dual_optimizer_last_sample_ts = now
    cop_snapshot = _dual_cop_snapshot(
        cache, active_action, operating_mode, tout=tout
    )
    cop_usable_now = bool(
        learning.get("learning_allowed") and cop_snapshot.get("valid")
    )
    if cop_usable_now and dt_s > 0.0:
        _dual_update_cop_stats(cop_snapshot, dt_s)

    snap = _system_defrost_snapshot() if _dual_enabled() else {}
    running_now = {}
    for name, item in (learning.get("learning_unit_states") or {}).items():
        running_now[str(name)] = bool(item.get("confirmed_running"))

    if _dual_episode is not None and dt_s > 0.0 and bool(learning.get("learning_allowed")):
        ep = _dual_episode
        if str(ep.get("mode")) != str(operating_mode):
            ep["invalid"] = True
            ep["invalid_reason"] = "mode_changed"
        ep["samples"] = int(ep.get("samples", 0)) + 1
        if power_valid:
            ep["valid_power_samples"] = int(ep.get("valid_power_samples", 0)) + 1
            # Trapezoidal power integration when a prior valid sample exists.
            p_used = total_power_w
            if _dual_optimizer_prev_power_w is not None and isfinite(_dual_optimizer_prev_power_w):
                p_used = 0.5 * (float(total_power_w) + float(_dual_optimizer_prev_power_w))
            ep["energy_kwh"] = float(ep.get("energy_kwh", 0.0)) + p_used * dt_s / 3600000.0
        if cop_usable_now:
            cop_power_w = float(cop_snapshot.get("active_input_power_w"))
            thermal_w = float(cop_snapshot.get("estimated_thermal_output_w"))
            ep["valid_cop_samples"] = int(ep.get("valid_cop_samples", 0)) + 1
            cop_power_used = cop_power_w
            thermal_used = thermal_w
            if (
                _dual_optimizer_prev_cop_power_w is not None and
                isfinite(_dual_optimizer_prev_cop_power_w)
            ):
                cop_power_used = 0.5 * (
                    cop_power_w + float(_dual_optimizer_prev_cop_power_w)
                )
            if (
                _dual_optimizer_prev_thermal_w is not None and
                isfinite(_dual_optimizer_prev_thermal_w)
            ):
                thermal_used = 0.5 * (
                    thermal_w + float(_dual_optimizer_prev_thermal_w)
                )
            ep["cop_input_kwh"] = (
                float(ep.get("cop_input_kwh", 0.0)) +
                cop_power_used * dt_s / 3600000.0
            )
            ep["estimated_thermal_kwh"] = (
                float(ep.get("estimated_thermal_kwh", 0.0)) +
                thermal_used * dt_s / 3600000.0
            )
            if float(ep.get("cop_input_kwh", 0.0)) > 1e-9:
                ep["combined_cop"] = (
                    float(ep.get("estimated_thermal_kwh", 0.0)) /
                    float(ep.get("cop_input_kwh", 0.0))
                )

        if isfinite(tin) and isfinite(tout):
            if operating_mode == "heat":
                load = max(2.0, float(tin) - float(tout))
            elif operating_mode == "cool":
                load = max(1.0, float(tout) - float(tin))
            elif operating_mode == "dry" and isfinite(humidity):
                load = max(1.0, float(humidity) - float(humidity_target))
            else:
                load = 1.0
            ep["degree_hours"] = float(ep.get("degree_hours", 0.0)) + load * dt_s / 3600.0
            excess = max(0.0, abs(float(sp) - float(tin)) - float(deadband))
            ep["comfort_penalty"] = float(ep.get("comfort_penalty", 0.0)) + excess * excess * dt_s / 3600.0

        if operating_mode == "dry" and isfinite(humidity):
            rh_excess = max(0.0, float(humidity) - float(humidity_target))
            ep["humidity_penalty"] = float(ep.get("humidity_penalty", 0.0)) + rh_excess * rh_excess * dt_s / 3600.0
            ep["humidity_last"] = float(humidity)

        actual_defrost = operating_mode == "heat" and (bool(system_freeze_active) or any([
            item.get("defrosting") is True and bool(_dual_optimizer_prev_heating.get(name))
            for name, item in snap.items()
        ]))
        if actual_defrost:
            ep["defrost_s"] = float(ep.get("defrost_s", 0.0)) + dt_s

        defrosting_now = any([item.get("defrosting") is True for item in snap.values()])
        if not defrosting_now:
            for name, running in running_now.items():
                if running and _dual_optimizer_prev_heating.get(name) is False:
                    ep["cycles"] = int(ep.get("cycles", 0)) + 1

        if _dual_manual_override(cache):
            ep["invalid"] = True
            ep["invalid_reason"] = "manual_override"

        if isfinite(sp) and abs(float(sp) - float(ep.get("sp_start", sp))) > 0.20:
            ep["invalid"] = True
            ep["invalid_reason"] = "setpoint_changed"

    if bool(learning.get("learning_allowed")):
        _dual_optimizer_prev_power_w = float(total_power_w) if power_valid else None
        _dual_optimizer_prev_cop_power_w = (
            float(cop_snapshot.get("active_input_power_w"))
            if cop_usable_now else None
        )
        _dual_optimizer_prev_thermal_w = (
            float(cop_snapshot.get("estimated_thermal_output_w"))
            if cop_usable_now else None
        )
        _dual_optimizer_prev_heating = running_now
    else:
        _dual_optimizer_prev_power_w = None
        _dual_optimizer_prev_cop_power_w = None
        _dual_optimizer_prev_thermal_w = None
        _dual_optimizer_prev_heating = {}
    _dual_publish_cop(
        cop_snapshot,
        learning_allowed=bool(learning.get("learning_allowed")),
        learning_reason=learning.get("learning_block_reason"),
    )
    _dual_publish_learning_status(operating_mode)


def _dual_actions_cop_configured(eligible, mode):
    """True when every eligible allocation has all required COP/power sensors."""
    if not _dual_cop_mode_supported(mode):
        return False
    for action in eligible or []:
        units = _dual_expected_learning_units(action)
        if not units:
            return False
        for u in units:
            if not u.get("COP_SENSOR") or not u.get("POWER_SENSOR"):
                return False
    return bool(eligible)


def _dual_policy_cop_ready(stats, eligible, mode, min_samples=1):
    if not _dual_actions_cop_configured(eligible, mode):
        return False
    required = max(1, int(min_samples))
    for action in eligible or []:
        action_id = str(action.get("id"))
        if int((stats.get(action_id) or {}).get("cop_n", 0)) < required:
            return False
    return bool(eligible)


def _dual_action_score(raw, prefer_cop=False):
    if not isinstance(raw, dict):
        return None
    try:
        if prefer_cop and int(raw.get("cop_n", 0)) > 0:
            value = float(raw.get("cop_score_mean", 0.0))
            return value if isfinite(value) else None
        n = int(raw.get("energy_score_n", raw.get("n", 0)))
        value = float(raw.get("energy_score_mean", raw.get("mean", 0.0)))
        return value if n > 0 and isfinite(value) else None
    except Exception:
        return None


def _dual_best_action(ctx, eligible, mode=None):
    stats = _dual_policy_stats.get(str(ctx), {})
    prefer_cop = _dual_policy_cop_ready(stats, eligible, mode, min_samples=1)
    ranked = []
    for action in eligible:
        action_id = str(action.get("id"))
        stat = stats.get(action_id)
        score = _dual_action_score(stat, prefer_cop=prefer_cop)
        if score is None:
            continue
        ranked.append((float(score), action))
    if not ranked:
        return _dual_default_action(eligible)
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def _dual_choose_action(
    ctx, eligible, err, rate, rate_valid,
    optimizer_mode, operating_mode,
):
    if not eligible:
        return None, None, "no_eligible_actions"
    best = _dual_best_action(ctx, eligible, mode=operating_mode)
    recommendation = best
    default = _dual_default_action(eligible)

    if optimizer_mode == "disabled":
        return default, recommendation, "optimizer_disabled"
    if optimizer_mode == "shadow":
        return default, recommendation, "shadow_recommendation_only"

    stats = _dual_policy_stats.get(str(ctx), {})
    safe_to_explore = (
        isfinite(err) and abs(float(err)) <= float(globals().get("DUAL_SAFE_EXPLORE_ERR_C", 0.20)) and
        ((not rate_valid) or abs(float(rate)) <= 0.15)
    )
    min_samples = max(1, int(globals().get("DUAL_MIN_SAMPLES_PER_ACTION", 3)))
    collect_cop = _dual_actions_cop_configured(eligible, operating_mode)

    if safe_to_explore:
        under_sampled = []
        for action in eligible:
            action_id = str(action.get("id"))
            raw = stats.get(action_id) or {}
            n = int(
                raw.get("cop_n", 0) if collect_cop else
                raw.get("energy_score_n", raw.get("n", 0))
            )
            if n < min_samples:
                under_sampled.append((n, action_id, action))
        if under_sampled:
            under_sampled.sort(key=lambda item: (item[0], item[1]))
            return under_sampled[0][2], recommendation, "safe_exploration"

    prefer_cop = _dual_policy_cop_ready(
        stats, eligible, operating_mode, min_samples=1
    )
    known_total = 0
    for action in eligible:
        raw = stats.get(str(action.get("id"))) or {}
        if prefer_cop:
            known_total += max(0, int(raw.get("cop_n", 0)))
        else:
            known_total += max(0, int(raw.get("energy_score_n", raw.get("n", 0))))
    scored = []
    bonus_k = float(globals().get("DUAL_EXPLORATION_BONUS", 0.12))
    for action in eligible:
        action_id = str(action.get("id"))
        stat = stats.get(action_id)
        if prefer_cop:
            n = int((stat or {}).get("cop_n", 0))
        else:
            n = int((stat or {}).get("energy_score_n", (stat or {}).get("n", 0)))
        mean = _dual_action_score(stat, prefer_cop=prefer_cop)
        if n <= 0 or mean is None:
            continue
        bonus = bonus_k * (_math_log(float(known_total) + 1.0) / float(n)) ** 0.5
        scored.append((float(mean) - bonus, action))
    if scored:
        scored.sort(key=lambda item: item[0])
        return scored[0][1], recommendation, "confidence_weighted_best"
    return default, recommendation, "default_no_scores"


def _dual_publish_optimizer(ctx, active_action, recommendation, reason, mode, power_valid=None):
    sensor = globals().get("DUAL_OPTIMIZER_SENSOR", "sensor.daikin_dual_efficiency_optimizer")
    action_id = str((active_action or {}).get("id", "none"))
    recommendation_id = str((recommendation or {}).get("id", "none"))
    ctx_stats = _dual_policy_stats.get(str(ctx), {})
    summary = {}
    for key, raw in ctx_stats.items():
        try:
            n = int(raw.get("n", 0))
            mean = float(raw.get("mean", 0.0))
            summary[str(key)] = {
                "n": n,
                "mean_score": round(mean, 6),
                "stddev": round((float(raw.get("m2", 0.0)) / float(max(1, n - 1))) ** 0.5, 6) if n > 1 else None,
                "last_score_source": str(raw.get("last_score_source", "energy_norm")),
                "energy_score_n": int(raw.get("energy_score_n", 0)),
                "energy_score_mean": round(float(raw.get("energy_score_mean", 0.0)), 6),
                "cop_n": int(raw.get("cop_n", 0)),
                "cop_score_mean": round(float(raw.get("cop_score_mean", 0.0)), 6),
                "combined_cop_mean": round(float(raw.get("combined_cop_mean", 0.0)), 4)
                    if int(raw.get("cop_n", 0)) > 0 else None,
                "energy_norm": round(float(raw.get("energy_norm_mean", 0.0)), 6),
                "comfort": round(float(raw.get("comfort_mean", 0.0)), 6),
                "defrost_fraction": round(float(raw.get("defrost_fraction_mean", 0.0)), 6),
                "humidity_penalty": round(float(raw.get("humidity_penalty_mean", 0.0)), 6),
            }
        except Exception:
            continue

    ep = _dual_episode or {}
    learning = _dual_learning_status_snapshot()
    try:
        state.set(
            sensor,
            value=recommendation_id if mode == "shadow" else action_id,
            mode=str(mode),
            context=str(ctx),
            operating_mode=str((_dual_episode or {}).get("mode") or _dual_active_mode),
            active_action=action_id,
            recommended_action=recommendation_id,
            selection_reason=str(reason),
            action_scores=summary,
            episode_action=ep.get("action"),
            episode_started_at=round(float(ep.get("start", 0.0)), 1) if ep else None,
            episode_energy_kwh=round(float(ep.get("energy_kwh", 0.0)), 4) if ep else None,
            episode_combined_cop=round(float(ep.get("combined_cop")), 4)
                if ep and ep.get("combined_cop") is not None else None,
            episode_cop_input_kwh=round(float(ep.get("cop_input_kwh", 0.0)), 4) if ep else None,
            episode_estimated_thermal_kwh=round(float(ep.get("estimated_thermal_kwh", 0.0)), 4) if ep else None,
            episode_valid_cop_samples=int(ep.get("valid_cop_samples", 0)) if ep else 0,
            episode_valid_power_samples=int(ep.get("valid_power_samples", 0)) if ep else 0,
            episode_samples=int(ep.get("samples", 0)) if ep else 0,
            power_sensors_valid=power_valid,
            actual_running=bool(learning.get("actual_running")),
            learning_allowed=bool(learning.get("learning_allowed")),
            learning_block_reason=learning.get("learning_block_reason"),
            active_learning_span_seconds=round(float(learning.get("active_learning_span_seconds", 0.0)), 1),
            live_combined_cop=_dual_current_cop_snapshot.get("combined_cop"),
            live_cop_valid=bool(
                _dual_current_cop_snapshot.get("live_value_available")
            ),
            live_cop_learning_eligible=bool(
                _dual_current_cop_snapshot.get("learning_eligible")
            ),
            live_cop_reason=_dual_current_cop_snapshot.get("status_reason"),
        )
    except Exception as e:
        log.debug("Daikin dual: failed to publish optimizer state: %s", e)


def _dual_climate_target_temperature(hvac_mode):
    """Return the fixed device target; the shared setpoint remains supervisory."""
    mode = str(hvac_mode or "").strip().lower()
    if mode == "heat":
        raw = globals().get("DUAL_HEAT_HVAC_TARGET_C", 31.0)
    elif mode == "cool":
        raw = globals().get("DUAL_COOL_HVAC_TARGET_C", 16.0)
    elif mode == "dry":
        raw = globals().get("DUAL_DRY_HVAC_TARGET_C", 16.0)
    else:
        return None
    try:
        target = float(raw)
    except Exception:
        return None
    return target if isfinite(target) else None


def _dual_mode_initial_demand(mode):
    mode = str(mode or "heat").lower()
    if mode == "cool":
        value = globals().get("DUAL_COOL_INITIAL_TOTAL_DEMAND", 70.0)
    elif mode == "dry":
        value = globals().get("DUAL_DRY_INITIAL_TOTAL_DEMAND", 55.0)
    else:
        value = globals().get("DUAL_HEAT_INITIAL_TOTAL_DEMAND", 80.0)
    try:
        return float(value)
    except Exception:
        return 80.0


def _dual_apply_startup_demand(u, mode, cache=None):
    """Set a safe startup Demand before the reconciler starts the compressor."""
    global _dual_last_startup_demand, _dual_last_total_target
    cache = cache or _TickCache()
    select_ent = u.get("SELECT") if u else None
    if not select_ent:
        _dual_last_startup_demand = None
        return None
    configured_initial = _dual_mode_initial_demand(mode)
    desired = float(globals().get("DUAL_DEGRADED_DEMAND", 100.0)) if _safety_degraded_active else configured_initial
    current = _dual_current_effective_demand(u, cache)
    step_ent = u.get("STEP_LIMIT_HELPER")
    step = cache.get_float(step_ent, default=AUTO_STEP_BASE) if step_ent else float(AUTO_STEP_BASE)
    step = _clip(step, float(AUTO_STEP_MIN), float(AUTO_STEP_MAX))
    applied_target = _clip(float(desired), float(current) - step, float(current) + step)
    option = _snap_to_select(select_ent, min(100.0, applied_target), 0)
    quiet_ent = u.get("QUIET_OUTDOOR_SWITCH")
    quiet = not (applied_target > 100.0 + 1e-6) if quiet_ent else None
    _desired_update_unit(
        u,
        demand_option=option,
        quiet=quiet,
        reason="startup_demand",
    )
    _dual_last_startup_demand = {
        "mode": str(mode),
        "physical_hvac_mode": _dual_physical_hvac_mode(mode),
        "configured": round(float(configured_initial), 2),
        "requested": round(float(desired), 2),
        "previous": round(float(current), 2),
        "applied_target": round(float(applied_target), 2),
        "select_option": option,
        "step_limited": abs(float(applied_target) - float(desired)) > 0.01,
        "requested_at": time.time(),
    }
    _dual_last_total_target = float(applied_target)
    sensor = globals().get("DUAL_TOTAL_DEMAND_SENSOR", "sensor.daikin_dual_total_demand")
    try:
        state.set(
            sensor,
            value=round(float(applied_target), 1),
            unit_of_measurement="% points",
            operating_mode=str(mode),
            physical_hvac_mode=_dual_physical_hvac_mode(mode),
            phase="startup_demand",
            startup_demand=dict(_dual_last_startup_demand),
            configured_units=len(DAIKINS),
            single_unit_fallback=len(DAIKINS) == 1,
            health=str(_safety_health),
        )
    except Exception:
        pass
    return float(applied_target)


def _dual_set_hvac_active(
    u, active, hvac_mode="heat", setpoint=None,
    allow_during_anomaly=False,
):
    """Update the authoritative unit intent; the reconciler applies it."""
    climate_ent = u.get("CLIMATE")
    allow_control = bool(u.get("ALLOW_HVAC_MODE_CONTROL", u.get("ALLOW_HVAC_OFF")))
    if not (climate_ent and allow_control):
        return
    logical_mode = str(hvac_mode or "").strip().lower()
    if not active:
        _desired_update_unit(
            u,
            active=False,
            mode="off",
            target=None,
            fan_mode=_dual_desired_fan_mode(u, "off"),
            reason="hvac_off",
            force_stop=bool(allow_during_anomaly),
        )
        return

    if logical_mode == "off_fan":
        _desired_update_unit(
            u,
            active=True,
            mode=logical_mode,
            target=None,
            demand_option=None,
            quiet=None,
            fan_mode=_dual_desired_fan_mode(u, logical_mode),
            reason="hvac_%s" % logical_mode,
            force_stop=False,
        )
        return

    if logical_mode not in (
        "heat", "cool", "dry", "fan_only", "off_fan",
    ):
        return
    conditioning_unit = _dual_summer_conditioning_unit(logical_mode)
    if (
        logical_mode in ("cool", "dry") and
        (conditioning_unit is None or u is not conditioning_unit)
    ):
        # This is a hard role boundary: Daikin1 must never receive a cooling
        # or drying command, even if an allocator, fallback, or future caller
        # requests one. It remains available for physical heat/reheat.
        _desired_update_unit(
            u,
            active=False,
            mode="off",
            target=None,
            fan_mode=_dual_desired_fan_mode(u, "off"),
            reason="%s_role_rejected" % logical_mode,
            force_stop=False,
        )
        return
    climate_target = _dual_climate_target_temperature(logical_mode)
    if not bool(globals().get("DUAL_SET_CLIMATE_TEMPERATURE", True)):
        climate_target = None
    _desired_update_unit(
        u,
        active=True,
        mode=logical_mode,
        target=(
            round(float(climate_target), 1)
            if climate_target is not None else None
        ),
        fan_mode=_dual_desired_fan_mode(u, logical_mode),
        reason="hvac_%s" % logical_mode,
        force_stop=False,
    )


def _dual_set_heat_standby_fan(u):
    """Keep non-heating Daikin1 in physical fan_only at lowMedium."""
    if str((u or {}).get("name", "")) != "daikin1":
        return False
    climate_ent = u.get("CLIMATE")
    allow_control = bool(
        u.get("ALLOW_HVAC_MODE_CONTROL", u.get("ALLOW_HVAC_OFF"))
    )
    if not (climate_ent and allow_control):
        return False
    _desired_update_unit(
        u,
        active=True,
        mode="fan_only",
        target=None,
        demand_option=None,
        quiet=None,
        fan_mode=str(globals().get("DUAL_SHARED_FAN_MODE", "lowMedium")),
        reason="daikin2_only_heat_daikin1_fan_only",
        force_stop=False,
    )
    return True


def _dual_reassert_active_hvac_targets():
    """Reassert fixed device targets every supervisor minute while active."""
    if not _dual_zone_active or _safety_health == "fault" or _safety_degraded_active:
        return
    mode = str(_dual_active_mode)
    if mode not in ("heat", "cool", "dry"):
        return
    if mode == "dry" and _dual_coordinated_dry_available():
        lead, assist = _dual_coordinated_dry_units()
        if lead:
            _dual_set_hvac_active(
                lead, not _dual_dry_lead_paused, "dry", None
            )
        if assist:
            _dual_set_hvac_active(
                assist, bool(_dual_dry_reheat_active),
                "heat" if _dual_dry_reheat_active else "off", None,
            )
        return
    if mode in ("cool", "dry"):
        conditioning = _dual_summer_conditioning_unit(mode)
        for u in DAIKINS:
            if u is conditioning:
                _dual_set_hvac_active(u, True, mode, None)
            else:
                desired = _desired_units.get(_unit_name(u)) or {}
                if not (
                    bool(desired.get("active")) and
                    str(desired.get("mode") or "").lower() == "heat"
                ):
                    _dual_set_hvac_active(u, False, "off", None)
        return
    action = _dual_action_by_id(_dual_active_action_id)
    expected = _dual_expected_learning_units(action)
    for u in expected:
        _dual_set_hvac_active(u, True, mode, None)
    for u in DAIKINS:
        if _dual_heat_standby_fan_required(u, mode, action):
            _dual_set_heat_standby_fan(u)


def _dual_cdp_snapshot():
    result = {}
    for name, raw in _dual_cdp_release.items():
        item = dict(raw or {})
        for key in ("fan_value", "expected_rpm", "restricted_below_rpm", "released_above_rpm", "demand"):
            value = item.get(key)
            item[key] = round(float(value), 2) if value is not None else None
        result[str(name)] = item
    return result


def _dual_cdp_bucket(outdoor_c):
    """Return the lower edge of the configured outdoor-temperature bucket."""
    width = max(0.5, float(globals().get("DUAL_CDP_OUTDOOR_BUCKET_C", 2.0)))
    return float(int(float(outdoor_c) // width) * width)


def _dual_cdp_bucket_key(bucket_c):
    return "%.1f" % float(bucket_c)


def _dual_cdp_curve_snapshot():
    return {
        str(name): {
            str(fan): {str(bucket): dict(record) for bucket, record in buckets.items()}
            for fan, buckets in fans.items()
        }
        for name, fans in _dual_cdp_temperature_curve.items()
    }


def _dual_publish_cdp_curve(cache=None):
    """Publish graph-ready CDP temperature/Demand series and current floors."""
    cache = cache or _TickCache()
    sensor = globals().get(
        "DUAL_CDP_CURVE_SENSOR", "sensor.daikin_dual_cdp_temperature_curve"
    )
    outdoor_c = _dual_outdoor_temperature(cache, allow_conservative_fallback=False)
    heat_cap = _clip(
        float(globals().get("DUAL_SUMMER_HEAT_DEMAND_CAP", 30.0)), 0.0, 100.0
    )
    width = max(0.5, float(globals().get("DUAL_CDP_OUTDOOR_BUCKET_C", 2.0)))
    attrs = {
        "unit_of_measurement": "% Demand",
        "current_outdoor_temperature": round(float(outdoor_c), 2) if isfinite(outdoor_c) else None,
        "bucket_width_c": width,
        "x_axis": "outdoor_temperature_c",
        "y_axis": "minimum_demand_percent",
        "controller_version": globals().get("DUAL_CONTROLLER_VERSION"),
        "updated_at": time.time(),
        "curve": _dual_cdp_curve_snapshot(),
    }
    current_floors = {}
    state_value = float(heat_cap)
    for u in DAIKINS:
        name = _unit_name(u)
        semantic = _dual_fan_mode_semantic(_dual_desired_fan_mode(u, "heat"))
        points = []
        records = []
        for bucket, record in _dual_cdp_curve_records(name, semantic):
            point = [
                round(float(bucket) + width / 2.0, 2),
                round(float(record.get("sufficient_demand")), 2),
            ]
            points.append(point)
            item = dict(record)
            item["outdoor_midpoint_c"] = point[0]
            item["effective_confidence"] = _dual_cdp_effective_confidence(record)
            records.append(item)
        floor_value, floor_info = _dual_cdp_lookup_floor(
            name, semantic, outdoor_c, heat_cap,
        )
        current_floors[name] = {
            "demand": round(float(floor_value), 2),
            "fan_semantic": semantic,
            "source": floor_info.get("source"),
            "bucket": floor_info.get("bucket"),
            "confidence": floor_info.get("confidence"),
        }
        attrs["%s_curve_points" % name] = points
        attrs["%s_curve_records" % name] = records
        state_value = max(state_value, float(floor_value))
    attrs["current_floor_by_unit"] = current_floors
    try:
        state.set(sensor, value=round(float(state_value), 2), **attrs)
    except Exception as e:
        log.debug("Daikin CDP curve sensor publish failed: %s", e)


def _dual_cdp_effective_confidence(record, now=None):
    now = time.time() if now is None else float(now)
    confidence = max(0, int((record or {}).get("confidence", 0)))
    learned_at = float((record or {}).get("last_learned", 0.0) or 0.0)
    decay_days = max(1.0, float(globals().get("DUAL_CDP_CONFIDENCE_DECAY_DAYS", 45.0)))
    if learned_at > 0.0:
        confidence = max(0, confidence - int(max(0.0, now - learned_at) // (decay_days * 86400.0)))
    return confidence


def _dual_cdp_curve_records(name, fan_semantic):
    records = []
    buckets = ((_dual_cdp_temperature_curve.get(str(name)) or {}).get(str(fan_semantic)) or {})
    for key, record in buckets.items():
        if not isinstance(record, dict):
            continue
        try:
            bucket = float(key)
            sufficient = float(record.get("sufficient_demand"))
        except Exception:
            continue
        if isfinite(bucket) and isfinite(sufficient):
            records.append((bucket, record))
    records.sort(key=lambda item: item[0])
    return records


def _dual_cdp_lookup_floor(name, fan_semantic, outdoor_c, heat_cap, now=None):
    """Conservatively select/interpolate the learned CDP Demand floor."""
    if not isfinite(outdoor_c):
        return float(heat_cap), {"source": "outdoor_unavailable", "bucket": None}
    now = time.time() if now is None else float(now)
    step = max(1.0, float(globals().get("DUAL_CDP_DEMAND_STEP", 5.0)))
    max_demand = float(globals().get("DUAL_CDP_MAX_DEMAND", 60.0))
    bucket = _dual_cdp_bucket(outdoor_c)
    records = _dual_cdp_curve_records(name, fan_semantic)
    if not records:
        legacy = _dual_cdp_learned_demand.get(str(name))
        floor_value = float(legacy) if legacy is not None else float(heat_cap)
        return _clip(floor_value, heat_cap, max_demand), {
            "source": "legacy" if legacy is not None else "unlearned",
            "bucket": bucket,
        }

    exact = None
    for edge, record in records:
        if abs(edge - bucket) < 0.01:
            exact = record
            break
    if exact is not None:
        confidence = _dual_cdp_effective_confidence(exact, now)
        floor_value = float(exact.get("sufficient_demand"))
        if confidence < int(globals().get("DUAL_CDP_FULL_CONFIDENCE", 3)):
            floor_value += step
        return _clip(floor_value, heat_cap, max_demand), {
            "source": "exact", "bucket": bucket, "confidence": confidence,
        }

    warmer = [(edge, rec) for edge, rec in records if edge > bucket]
    colder = [(edge, rec) for edge, rec in records if edge < bucket]
    colder_item = colder[-1] if colder else None
    warmer_item = warmer[0] if warmer else None
    max_distance = max(2.0, float(globals().get("DUAL_CDP_MAX_LOOKUP_DISTANCE_C", 6.0)))
    source = "unlearned"
    floor_value = float(heat_cap)
    confidence = 0
    if colder_item and warmer_item:
        low_edge, low_rec = colder_item
        high_edge, high_rec = warmer_item
        ratio = (bucket - low_edge) / max(0.1, high_edge - low_edge)
        interpolated = float(low_rec.get("sufficient_demand")) + ratio * (
            float(high_rec.get("sufficient_demand")) - float(low_rec.get("sufficient_demand"))
        )
        # Round upward to a selectable Demand step and never weaken the colder
        # neighbor when the learned curve is non-monotonic or sparse.
        floor_value = max(float(low_rec.get("sufficient_demand")), interpolated)
        floor_value = float(int((floor_value + step - 0.001) // step) * step)
        confidence = min(
            _dual_cdp_effective_confidence(low_rec, now),
            _dual_cdp_effective_confidence(high_rec, now),
        )
        source = "interpolated"
    else:
        nearest = min(records, key=lambda item: abs(item[0] - bucket))
        if abs(nearest[0] - bucket) <= max_distance:
            floor_value = float(nearest[1].get("sufficient_demand"))
            confidence = _dual_cdp_effective_confidence(nearest[1], now)
            source = "nearest"
    if source != "unlearned" and confidence < int(globals().get("DUAL_CDP_FULL_CONFIDENCE", 3)):
        floor_value += step
    return _clip(floor_value, heat_cap, max_demand), {
        "source": source, "bucket": bucket, "confidence": confidence,
    }


def _dual_cdp_record_boundary(
    name, fan_semantic, outdoor_c, sufficient_demand,
    insufficient_demand=None, now=None,
):
    """Persist one temperature-specific sufficient/insufficient boundary."""
    global _dual_last_store_save_ts
    if not isfinite(outdoor_c):
        return None
    now = time.time() if now is None else float(now)
    bucket = _dual_cdp_bucket(outdoor_c)
    key = _dual_cdp_bucket_key(bucket)
    buckets = _dual_cdp_temperature_curve.setdefault(str(name), {}).setdefault(str(fan_semantic), {})
    old = dict(buckets.get(key) or {})
    old_sufficient = old.get("sufficient_demand")
    sufficient = float(sufficient_demand)
    # Raise quickly after a failure; lower only one confirmed step at a time.
    if old_sufficient is not None and float(old_sufficient) > sufficient:
        sufficient = max(sufficient, float(old_sufficient) - float(globals().get("DUAL_CDP_DEMAND_STEP", 5.0)))
    failed = float(insufficient_demand) if insufficient_demand is not None else None
    record = {
        "bucket_low_c": bucket,
        "bucket_high_c": bucket + max(0.5, float(globals().get("DUAL_CDP_OUTDOOR_BUCKET_C", 2.0))),
        "sufficient_demand": sufficient,
        "insufficient_demand": failed,
        "confidence": min(10, max(0, int(old.get("confidence", 0))) + 1),
        "samples": max(0, int(old.get("samples", 0))) + 1,
        "last_learned": now,
    }
    buckets[key] = record
    _dual_last_store_save_ts = 0.0
    _dual_publish_cdp_curve()
    return dict(record)


def _dual_fan_profile_snapshot():
    return {
        str(name): {str(mode): dict(record) for mode, record in profile.items()}
        for name, profile in _dual_fan_rpm_profiles.items()
    }


def _dual_record_fan_calibration(unit_name, fan_mode, samples, option=None):
    """Store a robust median RPM for one unit/fan setting."""
    clean = []
    for value in samples or []:
        try:
            number = float(value)
        except Exception:
            continue
        if isfinite(number) and number > 0.0:
            clean.append(number)
    if not clean:
        return None
    clean.sort()
    middle = len(clean) // 2
    median = clean[middle] if len(clean) % 2 else 0.5 * (clean[middle - 1] + clean[middle])
    semantic = _dual_fan_mode_semantic(fan_mode)
    if semantic is None:
        return None
    profile = _dual_fan_rpm_profiles.setdefault(str(unit_name), {})
    record = {
        "option": str(option if option is not None else fan_mode),
        "rpm": round(float(median), 2), "samples": len(clean),
        "minimum": round(float(min(clean)), 2),
        "maximum": round(float(max(clean)), 2),
        "calibrated_at": time.time(),
    }
    profile[str(semantic)] = record
    return dict(record)


def _dual_publish_fan_calibration():
    sensor = globals().get("DUAL_FAN_CALIBRATION_SENSOR", "sensor.daikin_dual_fan_rpm_calibration")
    try:
        state.set(
            sensor,
            value=str(_dual_fan_calibration_status.get("state") or "idle"),
            calibration=dict(_dual_fan_calibration_status),
            profiles=_dual_fan_profile_snapshot(),
        )
    except Exception as e:
        log.debug("Daikin fan RPM calibration publish failed: %s", e)


def _dual_summer_cdp_demand(u, heat_cap, cache=None, now=None):
    """Use calibrated requested-speed RPM to detect and release internal CDP."""
    global _dual_last_store_save_ts
    cache = cache or _TickCache()
    now = time.time() if now is None else float(now)
    name = _unit_name(u)
    previous = _dual_cdp_release.get(name) or {}

    def _inactive(reason, fan_value=None, hvac=None):
        hvac = hvac or {}
        _dual_cdp_release[name] = {
            "state": "inactive", "reason": str(reason), "fan_value": fan_value,
            "reported_hvac_mode": str(hvac.get("mode") or "unknown"),
            "reported_hvac_action": str(hvac.get("action") or "unknown"),
            "running_evidence": hvac.get("running"),
            "expected_rpm": None, "restricted_below_rpm": None,
            "released_above_rpm": None,
            "demand": float(heat_cap), "started_at": 0.0, "low_since": 0.0,
            "release_since": 0.0, "last_step_at": 0.0, "released_at": 0.0,
        }
        return float(heat_cap), "summer_heat_demand_cap"

    if not bool(globals().get("DUAL_CDP_RELEASE_ENABLED", True)):
        return _inactive("disabled")
    fan_ent = u.get("FAN_RPM_SENSOR")
    if not fan_ent:
        return _inactive("fan_sensor_not_configured")
    # The MQTT fan-frequency entity is retained-state telemetry: the value is
    # published when RPM changes and legitimately remains unchanged while CDP
    # holds the fan.  Age therefore does not indicate invalid data here.
    # Treat every currently available numeric state as usable regardless of
    # last_updated/last_reported age.
    fan_value = cache.get_float(fan_ent, default=float("nan"))
    if not isfinite(float(fan_value)) or float(fan_value) < 0.0:
        return _inactive("fan_sensor_invalid")
    fan_value = float(fan_value)
    hvac = _unit_hvac_state(u, cache)
    if str(hvac.get("mode") or "").lower() != "heat":
        return _inactive("physical_mode_not_heat", fan_value, hvac)
    # Home Assistant normally exposes climate state ``heat`` and hvac_action
    # ``heating``.  Some Faikin/MQTT mappings expose ``heat`` for both; accept
    # either action spelling while the physical climate mode is heat.  If the
    # integration reports another/idle action, the configured compressor or
    # running entity is also authoritative positive heating evidence.
    action = str(hvac.get("action") or "").lower()
    if action not in ("heat", "heating") and hvac.get("running") is not True:
        return _inactive("heating_not_confirmed", fan_value, hvac)
    requested_semantic = _dual_fan_mode_semantic(_dual_desired_fan_mode(u, "heat"))
    if requested_semantic != _dual_fan_mode_semantic("mediumHigh"):
        return _inactive("mediumhigh_not_requested", fan_value)
    actual_fan = str((cache.getattr(u.get("CLIMATE"), {}) or {}).get("fan_mode") or "")
    if _dual_fan_mode_semantic(actual_fan) != requested_semantic:
        return _inactive("mediumhigh_not_acknowledged", fan_value)
    profile = (_dual_fan_rpm_profiles.get(name) or {}).get(str(requested_semantic))
    if not isinstance(profile, dict):
        return _inactive("fan_rpm_calibration_required", fan_value)
    try:
        expected_rpm = float(profile.get("rpm"))
    except Exception:
        expected_rpm = float("nan")
    if not isfinite(expected_rpm) or expected_rpm <= 0.0:
        return _inactive("fan_rpm_calibration_invalid", fan_value)
    defrosting, _liquid = _read_defrosting(u, cache)
    if defrosting is not False:
        return _inactive("defrost_or_unknown", fan_value)
    hold = _dual_post_defrost_holds.get(name) or {}
    if now < float(hold.get("until") or 0.0):
        return _inactive("post_defrost_hold", fan_value)

    outdoor_c = _dual_outdoor_temperature(cache, allow_conservative_fallback=False)
    learned_floor, floor_info = _dual_cdp_lookup_floor(
        name, requested_semantic, outdoor_c, heat_cap, now=now,
    )

    if str(previous.get("state") or "inactive") == "inactive":
        learned_demand = float(learned_floor)
        previous = {
            "state": "startup", "reason": "heating_stabilization",
            "fan_value": fan_value, "expected_rpm": expected_rpm,
            "reported_hvac_mode": str(hvac.get("mode") or "unknown"),
            "reported_hvac_action": str(hvac.get("action") or "unknown"),
            "running_evidence": hvac.get("running"),
            "demand": float(learned_demand),
            "last_released_demand": float(learned_demand),
            "outdoor_c": outdoor_c if isfinite(outdoor_c) else None,
            "outdoor_bucket": floor_info.get("bucket"),
            "floor_source": floor_info.get("source"),
            "learned_floor": float(learned_floor),
            "probe_started_at": 0.0, "probe_from_demand": None,
            "probe_outdoor_min": None, "probe_outdoor_max": None,
            "learning_recorded_demand": None,
            "started_at": now, "low_since": 0.0, "release_since": 0.0,
            "last_step_at": now, "released_at": 0.0,
        }
    previous["fan_value"] = fan_value
    previous["expected_rpm"] = expected_rpm
    previous["reported_hvac_mode"] = str(hvac.get("mode") or "unknown")
    previous["reported_hvac_action"] = str(hvac.get("action") or "unknown")
    previous["running_evidence"] = hvac.get("running")
    previous["outdoor_c"] = outdoor_c if isfinite(outdoor_c) else None
    previous["outdoor_bucket"] = floor_info.get("bucket")
    previous["floor_source"] = floor_info.get("source")
    previous["learned_floor"] = float(learned_floor)
    if isfinite(outdoor_c) and float(previous.get("probe_started_at") or 0.0) > 0.0:
        old_min = previous.get("probe_outdoor_min")
        old_max = previous.get("probe_outdoor_max")
        previous["probe_outdoor_min"] = float(outdoor_c) if old_min is None else min(float(old_min), float(outdoor_c))
        previous["probe_outdoor_max"] = float(outdoor_c) if old_max is None else max(float(old_max), float(outdoor_c))
    startup_s = max(0.0, float(globals().get("DUAL_CDP_STARTUP_SECONDS", 20.0)))
    if now - float(previous.get("started_at") or now) < startup_s:
        previous["state"] = "startup"
        previous["reason"] = "heating_stabilization"
        _dual_cdp_release[name] = previous
        startup_demand = max(float(heat_cap), float(previous.get("demand") or heat_cap))
        return startup_demand, (
            "summer_heat_cdp_learned_start"
            if startup_demand > float(heat_cap) else "summer_heat_demand_cap"
        )

    restricted_below = expected_rpm * _clip(float(globals().get("DUAL_CDP_RESTRICTED_RPM_RATIO", 0.75)), 0.1, 0.95)
    released_above = expected_rpm * _clip(float(globals().get("DUAL_CDP_RELEASED_RPM_RATIO", 0.85)), 0.2, 1.0)
    if released_above <= restricted_below:
        released_above = restricted_below + 0.05 * expected_rpm
    if fan_value >= released_above:
        released_now = True
    elif fan_value < restricted_below:
        released_now = False
    else:
        released_now = str(previous.get("state")) in ("released", "confirming_release")
    confirm_s = max(0.0, float(globals().get("DUAL_CDP_RELEASE_CONFIRM_SECONDS", 20.0)))
    demand = max(
        float(heat_cap), float(learned_floor),
        float(previous.get("demand") or heat_cap),
    )

    if released_now:
        previous["low_since"] = 0.0
        if not float(previous.get("release_since") or 0.0):
            previous["release_since"] = now
        if now - float(previous.get("release_since") or now) >= confirm_s:
            previous["state"] = "released"
            previous["reason"] = "airflow_release_confirmed"
            previous["released_at"] = float(previous.get("released_at") or now)
            previous["last_released_demand"] = float(demand)
            stable_learning_s = max(
                60.0,
                float(globals().get("DUAL_CDP_LEARNING_STABLE_SECONDS", 180.0)),
            )
            probe_after = max(
                stable_learning_s,
                float(globals().get("DUAL_CDP_PROBE_DOWN_AFTER_SECONDS", 180.0)),
            )
            stable_for = now - float(previous.get("released_at") or now)
            if (
                demand <= float(heat_cap) + 0.01 and
                stable_for >= stable_learning_s and
                previous.get("learning_recorded_demand") != float(demand)
            ):
                _dual_cdp_record_boundary(
                    name, requested_semantic, outdoor_c, demand,
                    insufficient_demand=None, now=now,
                )
                previous["learning_recorded_demand"] = float(demand)
            if demand > float(heat_cap) and now - float(previous.get("released_at") or now) >= probe_after:
                previous["probe_from_demand"] = float(demand)
                demand = max(float(heat_cap), demand - float(globals().get("DUAL_CDP_DEMAND_STEP", 5.0)))
                previous["released_at"] = now
                previous["probe_started_at"] = now
                previous["probe_outdoor_min"] = outdoor_c if isfinite(outdoor_c) else None
                previous["probe_outdoor_max"] = outdoor_c if isfinite(outdoor_c) else None
                previous["reason"] = "probing_lower_release_demand"
        else:
            previous["state"] = "confirming_release"
            previous["reason"] = "airflow_rise_detected"
    else:
        previous["release_since"] = 0.0
        previous["released_at"] = 0.0
        last_released = float(previous.get("last_released_demand") or 0.0)
        if last_released > demand + 0.01:
            probe_started = float(previous.get("probe_started_at") or 0.0)
            probe_min = previous.get("probe_outdoor_min")
            probe_max = previous.get("probe_outdoor_max")
            outdoor_stable = bool(
                isfinite(outdoor_c) and probe_min is not None and probe_max is not None and
                float(probe_max) - float(probe_min) <= float(globals().get("DUAL_CDP_OUTDOOR_STABILITY_C", 3.0))
            )
            failure_confirm = max(
                20.0,
                float(globals().get("DUAL_CDP_FAILURE_CONFIRM_SECONDS", 20.0)),
            )
            if probe_started > 0.0 and now - probe_started >= failure_confirm and outdoor_stable:
                _dual_cdp_record_boundary(
                    name, requested_semantic, outdoor_c,
                    float(previous.get("probe_from_demand") or last_released),
                    insufficient_demand=demand, now=now,
                )
            demand = min(
                float(globals().get("DUAL_CDP_MAX_DEMAND", 60.0)),
                last_released,
            )
            previous["last_step_at"] = now
            previous["low_since"] = now
            previous["state"] = "escalating"
            previous["reason"] = "restored_last_release_demand"
            previous["probe_started_at"] = 0.0
            previous["probe_from_demand"] = None
            previous["demand"] = float(demand)
            previous["restricted_below_rpm"] = float(restricted_below)
            previous["released_above_rpm"] = float(released_above)
            _dual_cdp_release[name] = previous
            return float(demand), "summer_heat_cdp_release"
        if not float(previous.get("low_since") or 0.0):
            previous["low_since"] = now
        low_confirm = max(0.0, float(globals().get("DUAL_CDP_LOW_CONFIRM_SECONDS", 20.0)))
        step_interval = max(1.0, float(globals().get("DUAL_CDP_STEP_INTERVAL_SECONDS", 20.0)))
        if now - float(previous.get("low_since") or now) >= low_confirm and now - float(previous.get("last_step_at") or 0.0) >= step_interval:
            demand = min(float(globals().get("DUAL_CDP_MAX_DEMAND", 60.0)), demand + float(globals().get("DUAL_CDP_DEMAND_STEP", 5.0)))
            previous["last_step_at"] = now
            previous["state"] = "escalating" if demand < float(globals().get("DUAL_CDP_MAX_DEMAND", 60.0)) else "limited"
            previous["reason"] = "fan_suppressed_demand_step" if previous["state"] == "escalating" else "max_demand_without_release"
        else:
            previous["state"] = "suspected"
            previous["reason"] = "confirming_suppressed_airflow"

    previous["demand"] = float(demand)
    previous["restricted_below_rpm"] = float(restricted_below)
    previous["released_above_rpm"] = float(released_above)
    _dual_cdp_release[name] = previous
    return float(demand), ("summer_heat_cdp_release" if demand > float(heat_cap) else "summer_heat_demand_cap")


def _dual_apply_unit_demand(
    u, desired, lower, upper, cache=None, mode="heat", setpoint=None,
    downward_min_interval_s=None,
):
    """Apply one demand, enforcing Summer Mode Demand invariants when active."""
    cache = cache or _TickCache()
    name = str(u.get("name", "daikin?"))
    if desired is None:
        _dual_set_hvac_active(u, False, "off", None)
        return None

    _dual_set_hvac_active(u, True, mode, setpoint)
    select_ent = u.get("SELECT")
    if not select_ent:
        return None
    if (
        name == "daikin2" and str(mode).lower() in ("cool", "dry")
    ):
        # Cooling and logical drying are fixed-output Daikin2 services.  Their
        # Demand select must not be reduced by PI allocation, outdoor bounds,
        # post-defrost holds, step limiting or the normal change cadence.
        fixed_demand = _clip(
            float(globals().get("DUAL_DAIKIN2_SUMMER_DEMAND", 100.0)),
            0.0, 100.0,
        )
        desired_option = _snap_to_select(select_ent, fixed_demand, 0)
        quiet_ent = u.get("QUIET_OUTDOOR_SWITCH")
        desired_quiet = True if quiet_ent else None
        _desired_update_unit(
            u,
            demand_option=desired_option,
            quiet=desired_quiet,
            reason="daikin2_%s_fixed_demand" % str(mode).lower(),
        )
        return float(fixed_demand)
    separate_topology = _separate_areas_enabled(cache)
    summer_heating = (
        _separate_area_summer_enabled(u, cache)
        if separate_topology else _summer_mode_enabled(cache)
    )
    if str(mode).lower() == "heat" and summer_heating:
        # Summer Mode permits heating, but limits every actively heating unit
        # to a low output.  Enforce this before the normal bounds, defrost hold,
        # step limiter and change cadence so none can leave Demand above the
        # configured cap, even briefly after Summer Mode is switched on.
        heat_cap = _clip(
            float(globals().get("DUAL_SUMMER_HEAT_DEMAND_CAP", 30.0)),
            0.0, 100.0,
        )
        capped_demand, cdp_reason = _dual_summer_cdp_demand(
            u, heat_cap, cache=cache,
        )
        capped_demand = min(
            max(float(capped_demand), float(heat_cap)),
            float(globals().get("DUAL_CDP_MAX_DEMAND", 60.0)),
        )
        desired_option = _snap_to_select(select_ent, capped_demand, 0)
        quiet_ent = u.get("QUIET_OUTDOOR_SWITCH")
        desired_quiet = True if quiet_ent else None
        _desired_update_unit(
            u,
            demand_option=desired_option,
            quiet=desired_quiet,
            reason=cdp_reason,
        )
        return float(capped_demand)
    current_eff = _dual_current_effective_demand(u, cache)
    desired = _clip(float(desired), float(lower), float(upper))
    hold = _dual_post_defrost_holds.get(name) or {}
    if time.time() < float(hold.get("until") or 0.0):
        desired = _clip(
            float(hold.get("demand", desired)), float(lower), float(upper)
        )

    step_ent = u.get("STEP_LIMIT_HELPER")
    step = cache.get_float(step_ent, default=AUTO_STEP_BASE) if step_ent else float(AUTO_STEP_BASE)
    step = _clip(step, float(AUTO_STEP_MIN), float(AUTO_STEP_MAX))
    # Step limit is always enforced, including emergency and optimizer changes.
    desired = _clip(desired, float(current_eff) - step, float(current_eff) + step)
    # Do not re-clamp after the step limiter. If a weather/action limit moves
    # across the currently applied demand, the command approaches that new
    # bound over several steps instead of violating the always-step-limited
    # requirement with one abrupt correction.

    base = min(100.0, float(desired))
    desired_option = _snap_to_select(select_ent, base, 0)
    quiet_ent = u.get("QUIET_OUTDOOR_SWITCH")
    desired_quiet = None
    if quiet_ent:
        desired_quiet = not (float(desired) > 100.0 + 1e-6)

    current_select = cache.get_str(select_ent, default="")
    current_quiet = cache.get_str(quiet_ent, default="").lower() if quiet_ent else ""
    current_sig = (str(current_select), current_quiet)
    desired_sig = (str(desired_option), "on" if desired_quiet else "off" if desired_quiet is not None else "")
    min_interval = float(_demand_change_min_interval_s(u))
    if (
        downward_min_interval_s is not None and
        float(desired) < float(current_eff) - 0.01
    ):
        # Only the predictive reduction path may shorten this interval. Demand
        # increases and ordinary control retain the configured minimum.
        min_interval = min(
            min_interval,
            max(0.0, float(downward_min_interval_s)),
        )
    last_change = float(_last_demand_change_ts.get(name) or 0.0)
    if current_sig != desired_sig and time.time() - last_change < min_interval:
        return float(current_eff)

    changed = current_sig != desired_sig
    _desired_update_unit(
        u,
        demand_option=desired_option,
        quiet=desired_quiet,
        reason=(
            "post_defrost_hold"
            if time.time() < float(hold.get("until") or 0.0)
            else "demand_control"
        ),
    )
    if changed:
        _last_demand_change_ts[name] = time.time()
        _last_demand_sig[name] = desired_sig
    return float(desired)


def _dual_cdp_fast_tick(cache=None):
    """Reconcile CDP release on the 20-second adaptive scheduler."""
    cache = cache or _TickCache()
    heat_cap = _clip(float(globals().get("DUAL_SUMMER_HEAT_DEMAND_CAP", 30.0)), 0.0, 100.0)
    if not _summer_mode_enabled(cache):
        for u in DAIKINS:
            name = _unit_name(u)
            if name in _dual_cdp_release:
                _dual_cdp_release[name] = {
                    "state": "inactive", "reason": "summer_mode_off",
                    "fan_value": None, "expected_rpm": None,
                    "restricted_below_rpm": None, "released_above_rpm": None,
                    "demand": heat_cap,
                }
        return
    for u in DAIKINS:
        desired = _desired_units.get(_unit_name(u)) or {}
        if not bool(desired.get("active")) or str(desired.get("mode") or "").lower() != "heat":
            continue
        demand, reason = _dual_summer_cdp_demand(u, heat_cap, cache=cache)
        select_ent = u.get("SELECT")
        if not select_ent:
            continue
        _desired_update_unit(
            u,
            demand_option=_snap_to_select(select_ent, demand, 0),
            quiet=True if u.get("QUIET_OUTDOOR_SWITCH") else None,
            reason=reason,
        )
    _dual_publish_cdp_curve(cache)


def _dual_update_cold_active(tout, cache=None):
    global _dual_cold_active, _dual_action_changed_at
    if not isfinite(tout):
        return _dual_cold_active
    cache = cache or _TickCache()
    on_c = _dual_float_helper(
        globals().get("DUAL_ASSIST_ON_BELOW_HELPER", ""),
        globals().get("DUAL_ASSIST_ON_BELOW_C_DEFAULT", 3.0),
        -20.0, 15.0, cache,
    )
    off_c = _dual_float_helper(
        globals().get("DUAL_ASSIST_OFF_ABOVE_HELPER", ""),
        globals().get("DUAL_ASSIST_OFF_ABOVE_C_DEFAULT", 4.0),
        -19.0, 20.0, cache,
    )
    if off_c <= on_c:
        off_c = on_c + 0.5
    if (not _dual_cold_active) and float(tout) <= on_c:
        _dual_cold_active = True
        _dual_action_changed_at = time.time()
        _dual_discard_episode("heating_assist_enabled")
        log.warning("Daikin dual: cold assistance enabled at Tout=%.2f C", float(tout))
    elif _dual_cold_active and float(tout) >= off_c:
        _dual_cold_active = False
        _dual_action_changed_at = time.time()
        _dual_discard_episode("heating_assist_disabled")
        log.warning("Daikin dual: cold assistance disabled at Tout=%.2f C", float(tout))
    return _dual_cold_active


def _dual_update_assist(mode, tout, control_err, directed_rate, rate_valid, tin, sp, humidity, cache=None):
    """Mode-aware second-unit staging with hysteresis and response checks."""
    global _dual_action_changed_at
    cache = cache or _TickCache()
    if len(DAIKINS) < 2:
        _dual_assist_by_mode[mode] = False
        return False
    if mode == "heat":
        active = bool(_dual_update_cold_active(tout, cache))
        _dual_assist_by_mode["heat"] = active
        return active

    old = bool(_dual_assist_by_mode.get(mode, False))
    new = old
    reason = None

    if mode == "cool":
        # Daikin2 is the exclusive cooling unit. Daikin1 may only be used as
        # physical heat/reheat by a coordinated summer-mode path, never as a
        # second cooling compressor.
        new = False
        reason = "daikin2_exclusive_cooling"

    elif mode == "dry":
        # Humidity-driven dry never stages a second drying compressor. When
        # available, _dual_run_coordinated_dry owns the assist in heat mode;
        # otherwise dry remains lead-only.
        new = False
        reason = "coordinated_reheat_or_lead_only"

    else:
        new = False

    _dual_assist_by_mode[mode] = bool(new)
    if new != old:
        _dual_action_changed_at = time.time()
        _dual_discard_episode("assist_stage_changed")
        log.warning("Daikin dual: %s second-unit assist %s (%s)", mode, "enabled" if new else "disabled", reason)
    return bool(new)


def _dual_deactivate_assist_units():
    for u in DAIKINS:
        if u is _dual_primary_unit() or str(u.get("ROLE", "")).lower() == "lead":
            continue
        if bool(u.get("ALLOW_HVAC_MODE_CONTROL", u.get("ALLOW_HVAC_OFF"))) and bool(u.get("CLIMATE")):
            _dual_set_hvac_active(u, False, "off", None)


def _dual_reset_dry_coordination(reason="inactive"):
    global _dual_dry_reheat_active, _dual_dry_reheat_started_at
    global _dual_dry_reheat_stopped_at, _dual_dry_reheat_integral
    global _dual_dry_reheat_last_ctrl_ts, _dual_dry_reheat_last_demand
    global _dual_dry_lead_paused, _dual_dry_lead_paused_at
    global _dual_dry_lead_resumed_at, _dual_dry_coordination_status
    now = time.time()
    if _dual_dry_reheat_active:
        _dual_dry_reheat_stopped_at = now
    _dual_dry_reheat_active = False
    _dual_dry_reheat_started_at = 0.0
    _dual_dry_reheat_integral = 0.0
    _dual_dry_reheat_last_ctrl_ts = 0.0
    _dual_dry_reheat_last_demand = None
    _dual_dry_lead_paused = False
    _dual_dry_lead_paused_at = 0.0
    _dual_dry_lead_resumed_at = 0.0
    _dual_dry_coordination_status = {
        "available": bool(_dual_coordinated_dry_available()),
        "active": False,
        "lead_paused": False,
        "reason": str(reason),
    }


def _dual_run_coordinated_dry(cache=None):
    """Run lead-unit dehumidification with independent assist-unit reheat.

    The two outputs solve different errors and therefore never enter the
    ordinary same-mode allocator or its COP/action comparisons.
    """
    global _dual_err_int, _dual_last_ctrl_ts, _dual_last_sp
    global _dual_last_total_target, _dual_active_action_id
    global _dual_dry_reheat_active, _dual_dry_reheat_started_at
    global _dual_dry_reheat_stopped_at, _dual_dry_reheat_integral
    global _dual_dry_reheat_last_ctrl_ts, _dual_dry_reheat_last_demand
    global _dual_dry_lead_paused, _dual_dry_lead_paused_at
    global _dual_dry_lead_resumed_at, _dual_dry_coordination_status
    global _dual_learning_block_reason

    cache = cache or _TickCache()
    lead, assist = _dual_coordinated_dry_units()
    if not (
        lead and assist and _dual_zone_active and
        str(_dual_active_mode) == "dry" and
        not _safety_degraded_active and _safety_health != "fault"
    ):
        return False

    for u in DAIKINS:
        if u is not lead and u is not assist:
            _dual_set_hvac_active(u, False, "off", None)

    now = time.time()
    tin = _dual_zone_temperature(cache)
    humidity = _dual_humidity(cache, filtered=True)
    tout = _dual_outdoor_temperature(cache, allow_conservative_fallback=True)
    sp, min_guard, _max_guard, deadband = _dual_effective_setpoint(
        cache, write_helpers=True, mode="dry"
    )
    limits = _dual_supervisor_limits(cache)
    if not (isfinite(tin) and isfinite(humidity) and isfinite(tout) and isfinite(sp)):
        return False

    # Keep control-only rate histories alive even though mixed-service
    # efficiency learning is intentionally disabled.
    _dual_add_temperature_sample(now, tin)
    _dual_add_fast_temperature_sample(now, tin)
    humidity_raw = _dual_humidity(cache, filtered=False)
    if isfinite(humidity_raw):
        _dual_add_learning_humidity_sample(now, humidity_raw)

    temp_rate, temp_rate_valid, rate_span, rate_n = _dual_temperature_rate(now)
    fast_rate, fast_rate_valid, fast_span, fast_n = _dual_fast_temperature_rate(now)
    if fast_rate_valid:
        temp_rate = fast_rate
        temp_rate_valid = True
        rate_span = fast_span
        rate_n = fast_n
    temp_rate_cph = _clip(
        temp_rate,
        -float(globals().get("DUAL_RATE_CLAMP_CPH", 2.0)),
        float(globals().get("DUAL_RATE_CLAMP_CPH", 2.0)),
    ) if temp_rate_valid else 0.0
    rh_rate, rh_rate_valid = _dual_humidity_rate(now)
    assist_defrosting, _assist_liquid = _read_defrosting(assist, cache)

    # ----- Lead drying Demand (humidity loop) -----
    lead_lo, lead_hi, _lead_icing = _dual_unit_bounds(
        lead, tout, tin, min_guard, cache, mode="dry"
    )
    dry_factor = max(
        0.01,
        float(globals().get("DUAL_DRY_RH_ERROR_TO_CONTROL", 0.10)),
    )
    dry_err = max(0.0, float(humidity) - float(limits["humidity_target"])) * dry_factor
    dry_directed_rate = -float(rh_rate) * dry_factor if rh_rate_valid else 0.0
    current_lead = _dual_current_effective_demand(lead, cache)
    dry_ctx = "dry|%s" % _context_key_for_outdoor(tout)
    dry_sustain = float(_dual_sustain_by_ctx.get(
        dry_ctx,
        globals().get("DUAL_DRY_INITIAL_TOTAL_DEMAND", 55.0),
    ))
    dry_dt = now - float(_dual_last_ctrl_ts or 0.0)
    if not isfinite(dry_dt) or dry_dt <= 0.0:
        dry_dt = 60.0
    dry_dt = _clip(dry_dt, 10.0, 300.0)
    _dual_last_ctrl_ts = now
    if (not isfinite(_dual_last_sp)) or abs(float(sp) - float(_dual_last_sp)) > 0.5:
        _dual_err_int = 0.0
    _dual_last_sp = float(sp)
    dry_kp = float(globals().get("DUAL_DRY_TRACK_KP", 8.0))
    dry_ki = float(globals().get("DUAL_DRY_TRACK_KI", 1.0))
    dry_kd = float(globals().get("DUAL_DRY_TRACK_KD", 0.3))
    _dual_err_int = _clip(
        float(_dual_err_int) + dry_ki * dry_err * dry_dt / 3600.0,
        -float(globals().get("DUAL_TRACK_I_CLAMP", 35.0)),
        float(globals().get("DUAL_TRACK_I_CLAMP", 35.0)),
    )
    dry_raw = dry_sustain + dry_kp * dry_err + _dual_err_int - dry_kd * dry_directed_rate
    dry_target = _clip(dry_raw, lead_lo, lead_hi)

    pause_below = max(
        0.05,
        float(globals().get("DUAL_DRY_LEAD_PAUSE_BELOW_SETPOINT_C", 0.30)),
    )
    resume_below = _clip(
        float(globals().get("DUAL_DRY_LEAD_RESUME_BELOW_SETPOINT_C", 0.05)),
        0.0,
        pause_below - 0.01,
    )
    lead_min_off_s = max(
        0.0,
        float(globals().get("DUAL_DRY_LEAD_MIN_OFF_MINUTES", 5.0)) * 60.0,
    )
    if (
        not _dual_dry_lead_paused and
        (float(tin) <= float(sp) - pause_below or assist_defrosting)
    ):
        _dual_dry_lead_paused = True
        _dual_dry_lead_paused_at = now
        _dual_dry_lead_resumed_at = 0.0
        _dual_discard_episode(
            "coordinated_dry_assist_defrost"
            if assist_defrosting
            else "coordinated_dry_temperature_floor"
        )
    elif (
        _dual_dry_lead_paused and
        not assist_defrosting and
        float(tin) >= float(sp) - resume_below and
        now - float(_dual_dry_lead_paused_at or now) >= lead_min_off_s
    ):
        _dual_dry_lead_paused = False
        _dual_dry_lead_resumed_at = now
        _dual_dry_lead_paused_at = 0.0

    if _dual_dry_lead_paused:
        _dual_set_hvac_active(lead, False, "off", None)
        applied_lead = None
    else:
        applied_lead = _dual_apply_unit_demand(
            lead, dry_target, lead_lo, lead_hi, cache, mode="dry", setpoint=sp
        )

    # ----- Assist heating Demand (temperature loop) -----
    target_offset = float(globals().get("DUAL_DRY_REHEAT_TARGET_OFFSET_C", 0.10))
    on_margin = float(globals().get("DUAL_DRY_REHEAT_ON_MARGIN_C", 0.10))
    off_margin = max(
        on_margin + 0.05,
        float(globals().get("DUAL_DRY_REHEAT_OFF_MARGIN_C", 0.30)),
    )
    prediction_min = max(
        0.0,
        float(globals().get("DUAL_DRY_REHEAT_PREDICTION_MINUTES", 5.0)),
    )
    projected_tin = (
        float(tin) + float(temp_rate_cph) * prediction_min / 60.0
        if temp_rate_valid else float(tin)
    )
    reheat_needed = bool(
        float(tin) <= float(sp) + on_margin or
        (
            temp_rate_valid and float(temp_rate_cph) < -0.02 and
            float(projected_tin) <= float(sp) + on_margin
        ) or
        _dual_dry_lead_paused
    )
    min_on_s = max(
        0.0,
        float(globals().get("DUAL_DRY_REHEAT_MIN_ON_MINUTES", 10.0)) * 60.0,
    )
    min_off_s = max(
        0.0,
        float(globals().get("DUAL_DRY_REHEAT_MIN_OFF_MINUTES", 5.0)) * 60.0,
    )
    reheat_reason = "holding"
    if not _dual_dry_reheat_active:
        off_elapsed = (
            now - float(_dual_dry_reheat_stopped_at)
            if _dual_dry_reheat_stopped_at else 1e9
        )
        if reheat_needed and off_elapsed >= min_off_s:
            _dual_dry_reheat_active = True
            _dual_dry_reheat_started_at = now
            _dual_dry_reheat_integral = 0.0
            _dual_dry_reheat_last_ctrl_ts = now
            reheat_reason = "temperature_support_started"
        elif reheat_needed:
            reheat_reason = "reheat_minimum_off_wait"
        else:
            reheat_reason = "temperature_support_not_needed"
    else:
        on_elapsed = now - float(_dual_dry_reheat_started_at or now)
        recovered = bool(
            float(tin) >= float(sp) + off_margin and
            (not temp_rate_valid or float(temp_rate_cph) >= -0.02) and
            not _dual_dry_lead_paused
        )
        if recovered and on_elapsed >= min_on_s:
            _dual_dry_reheat_active = False
            _dual_dry_reheat_stopped_at = now
            _dual_dry_reheat_started_at = 0.0
            _dual_dry_reheat_integral = 0.0
            _dual_dry_reheat_last_ctrl_ts = 0.0
            _dual_dry_reheat_last_demand = None
            reheat_reason = "temperature_recovered"
        elif recovered:
            reheat_reason = "reheat_minimum_on_hold"
        elif _dual_dry_lead_paused:
            reheat_reason = "lead_paused_temperature_recovery"
        elif float(projected_tin) <= float(sp) + on_margin:
            reheat_reason = "projected_temperature_support"

    applied_reheat = None
    reheat_raw = None
    reheat_target = None
    if _dual_dry_reheat_active:
        heat_lo, heat_hi, _heat_icing = _dual_unit_bounds(
            assist, tout, tin, min_guard, cache, mode="heat"
        )
        heat_lo = max(
            heat_lo,
            float(globals().get("DUAL_DRY_REHEAT_MIN_DEMAND", 30.0)),
        )
        heat_hi = min(
            heat_hi,
            float(globals().get("DUAL_DRY_REHEAT_MAX_DEMAND", 70.0)),
        )
        if heat_hi < heat_lo:
            heat_hi = heat_lo
        if assist_defrosting:
            # Keep the pre-defrost Demand and stop adding a competing drying
            # load until the assist can produce heat again.
            applied_reheat = _dual_current_effective_demand(assist, cache)
            _dual_set_hvac_active(assist, True, "heat", sp)
            reheat_reason = "assist_defrost_lead_paused"
        else:
            reheat_dt = now - float(_dual_dry_reheat_last_ctrl_ts or now)
            reheat_dt = _clip(reheat_dt, 10.0, 300.0)
            _dual_dry_reheat_last_ctrl_ts = now
            temp_error = float(sp) + target_offset - float(tin)
            reheat_ki = float(globals().get("DUAL_DRY_REHEAT_KI", 4.0))
            _dual_dry_reheat_integral = _clip(
                float(_dual_dry_reheat_integral) +
                reheat_ki * temp_error * reheat_dt / 3600.0,
                -float(globals().get("DUAL_DRY_REHEAT_I_CLAMP", 20.0)),
                float(globals().get("DUAL_DRY_REHEAT_I_CLAMP", 20.0)),
            )
            reheat_raw = (
                float(globals().get("DUAL_DRY_REHEAT_INITIAL_DEMAND", 35.0)) +
                float(globals().get("DUAL_DRY_REHEAT_KP", 30.0)) * temp_error +
                _dual_dry_reheat_integral -
                float(globals().get("DUAL_DRY_REHEAT_KD", 10.0)) * float(temp_rate_cph)
            )
            reheat_target = _clip(reheat_raw, heat_lo, heat_hi)
            applied_reheat = _dual_apply_unit_demand(
                assist, reheat_target, heat_lo, heat_hi, cache,
                mode="heat", setpoint=sp,
            )
        chosen_reheat = applied_reheat if applied_reheat is not None else reheat_target
        _dual_dry_reheat_last_demand = (
            float(chosen_reheat) if chosen_reheat is not None else None
        )
    else:
        _dual_set_hvac_active(assist, False, "off", None)

    _dual_active_action_id = "coordinated_reheat"
    _dual_last_total_target = float(
        (applied_lead if applied_lead is not None else 0.0) +
        (applied_reheat if applied_reheat is not None else 0.0)
    )
    if str(_dual_learning_block_reason) != "coordinated_heat_dry_service":
        _dual_reset_learning_window(
            "coordinated_heat_dry_service", discard_episode=True
        )
    _dual_learning_block_reason = "coordinated_heat_dry_service"
    _dual_dry_coordination_status = {
        "available": True,
        "active": bool(_dual_dry_reheat_active),
        "lead_unit": str(lead.get("name")),
        "assist_unit": str(assist.get("name")),
        "lead_mode": "off" if _dual_dry_lead_paused else "dry",
        "lead_logical_mode": "dry",
        "lead_target": (
            None if _dual_dry_lead_paused
            else _dual_climate_target_temperature("dry")
        ),
        "lead_fan_mode": (
            globals().get("DUAL_DEFAULT_FAN_MODE", "auto")
            if _dual_dry_lead_paused
            else globals().get("DUAL_DRY_FAN_MODE", "lowMedium")
        ),
        "assist_mode": "heat" if _dual_dry_reheat_active else "off",
        "assist_defrosting": bool(assist_defrosting),
        "lead_paused": bool(_dual_dry_lead_paused),
        "lead_pause_floor": round(float(sp) - pause_below, 2),
        "lead_resume_temperature": round(float(sp) - resume_below, 2),
        "temperature_target": round(float(sp) + target_offset, 2),
        "projected_temperature": round(float(projected_tin), 3),
        "temperature_rate_cph": round(float(temp_rate_cph), 4),
        "temperature_rate_valid": bool(temp_rate_valid),
        "humidity": round(float(humidity), 2),
        "humidity_target": round(float(limits["humidity_target"]), 1),
        "dry_demand": round(float(applied_lead), 2) if applied_lead is not None else None,
        "reheat_demand": round(float(applied_reheat), 2) if applied_reheat is not None else None,
        "reheat_raw": round(float(reheat_raw), 2) if reheat_raw is not None else None,
        "reheat_minimum_on_remaining_seconds": round(
            max(0.0, min_on_s - (now - float(_dual_dry_reheat_started_at or now))),
            1,
        ) if _dual_dry_reheat_active else 0.0,
        "reheat_minimum_off_remaining_seconds": round(
            max(0.0, min_off_s - (
                now - float(_dual_dry_reheat_stopped_at or 0.0)
                if _dual_dry_reheat_stopped_at else 1e9
            )),
            1,
        ) if not _dual_dry_reheat_active else 0.0,
        "reason": reheat_reason,
    }
    sensor = globals().get("DUAL_TOTAL_DEMAND_SENSOR", "sensor.daikin_dual_total_demand")
    try:
        state.set(
            sensor,
            value=round(float(_dual_last_total_target), 1),
            unit_of_measurement="% points",
            operating_mode="dry",
            physical_hvac_mode="mixed_dry_heat",
            action="coordinated_reheat",
            indoor=round(float(tin), 2),
            setpoint=round(float(sp), 2),
            humidity=round(float(humidity), 2),
            humidity_target=round(float(limits["humidity_target"]), 1),
            temperature_rate_cph=round(float(temp_rate_cph), 4),
            rate_span_s=round(float(rate_span), 1),
            rate_samples=int(rate_n),
            optimizer_mode="suspended_mixed_service",
            learning_allowed=False,
            learning_block_reason="coordinated_heat_dry_service",
            coordinated_reheat=dict(_dual_dry_coordination_status),
            dry_demand=_dual_dry_coordination_status.get("dry_demand"),
            reheat_demand=_dual_dry_coordination_status.get("reheat_demand"),
            configured_units=len(DAIKINS),
            single_unit_fallback=False,
        )
    except Exception as e:
        log.debug("Daikin dual: failed to publish coordinated dry state: %s", e)
    _runtime_store_save(force=False)
    return True


def _run_dual_zone(mode=None):
    """One mode-normalized PI/sustain controller plus energy-aware allocation."""
    global _dual_err_int, _dual_last_ctrl_ts, _dual_last_sp
    global _dual_last_total_target, _dual_active_action_id, _dual_action_changed_at
    global _dual_zone_active, _dual_active_mode, _dual_zone_stopped_at
    global _dual_landing_status

    _dual_load_store()
    now = time.time()
    cache = _TickCache()
    _requested, effective_mode, _summer, _block = _effective_hvac_mode(cache)
    mode = effective_mode if mode not in ("heat", "cool", "dry") else mode
    if mode not in ("heat", "cool", "dry") or not _dual_zone_active:
        return
    if mode == "dry" and _dual_coordinated_dry_available():
        _dual_run_coordinated_dry(cache)
        return
    startup_hold_s = max(0.0, float(globals().get("DUAL_STARTUP_DEMAND_HOLD_S", 60.0)))
    if _dual_zone_started_at and now - float(_dual_zone_started_at) < startup_hold_s:
        # The supervisor has already replaced the retained cross-mode value.
        # Give that acknowledged startup command one minute before PI/optimizer
        # output is allowed to supersede it.
        return
    tin = _dual_zone_temperature(cache)
    tout = _dual_outdoor_temperature(cache, allow_conservative_fallback=True)
    humidity = _dual_humidity(cache, filtered=True)
    sp, min_guard, _max_guard, deadband = _dual_effective_setpoint(cache, write_helpers=True, mode=mode)
    if not (isfinite(tin) and isfinite(tout) and isfinite(sp)):
        log.warning("Daikin dual: sensors not ready Tin=%s Tout=%s SP=%s", tin, tout, sp)
        return
    if _dual_manual_override(cache):
        _dual_discard_episode("manual_override")
        _dual_publish_optimizer("manual", None, None, "manual_override", _dual_optimizer_mode(cache))
        return

    temp_rate, temp_rate_valid, rate_span, rate_n = _dual_temperature_rate(now)
    temp_rate_cph = _clip(temp_rate, -float(globals().get("DUAL_RATE_CLAMP_CPH", 2.0)), float(globals().get("DUAL_RATE_CLAMP_CPH", 2.0))) if temp_rate_valid else 0.0
    humidity_target = _dual_supervisor_limits(cache)["humidity_target"]
    if mode == "heat":
        err = float(sp) - float(tin)
        directed_rate = float(temp_rate_cph)
        rate_valid = temp_rate_valid
    elif mode == "cool":
        err = float(tin) - float(sp)
        directed_rate = -float(temp_rate_cph)
        rate_valid = temp_rate_valid
    else:
        rh_factor = max(0.01, float(globals().get("DUAL_DRY_RH_ERROR_TO_CONTROL", 0.10)))
        err = max(0.0, float(humidity) - float(humidity_target)) * rh_factor if isfinite(humidity) else 0.0
        rh_rate, rh_rate_valid = _dual_humidity_rate(now)
        directed_rate = -float(rh_rate) * rh_factor if rh_rate_valid else 0.0
        rate_valid = rh_rate_valid

    bounds = [_dual_unit_bounds(u, tout, tin, min_guard, cache, mode=mode) for u in DAIKINS]
    cap_weights = [
        _dual_float_helper(None, u.get("CAPACITY_WEIGHT", 1.0), 0.05, 20.0)
        for u in DAIKINS
    ]
    total_max_all = sum([
        cap_weights[idx] * bounds[idx][1] for idx in range(len(DAIKINS))
    ])
    prior_action = _dual_action_by_id(_dual_active_action_id)
    initial_key = "DUAL_%s_INITIAL_TOTAL_DEMAND" % mode.upper()
    configured_initial = float(globals().get(initial_key, globals().get("DUAL_INITIAL_TOTAL_DEMAND", 80.0)))
    # A retained demand-select value is not output while the compressor is off.
    # Use it as the controller reference only after actual operation is proven.
    if _dual_learning_actual_running:
        current_total = _dual_current_total_demand(cache, prior_action)
    else:
        current_total = configured_initial

    outdoor_ctx = "%s|%s" % (mode, _context_key_for_outdoor(tout))
    if outdoor_ctx not in _dual_sustain_by_ctx:
        # Never seed a learned context from a select which may merely retain an
        # old off-state value. Start from the explicit mode default instead.
        _dual_sustain_by_ctx[outdoor_ctx] = _clip(configured_initial, 0.0, total_max_all)
    sustain = float(_dual_sustain_by_ctx[outdoor_ctx])

    # Shared PI controller. Only this integral sees the common zone error.
    dt_s = now - float(_dual_last_ctrl_ts or 0.0)
    if not isfinite(dt_s) or dt_s <= 0.0:
        dt_s = 300.0
    dt_s = _clip(dt_s, 10.0, 300.0)
    _dual_last_ctrl_ts = now
    if (not isfinite(_dual_last_sp)) or abs(float(sp) - float(_dual_last_sp)) > 0.5:
        _dual_err_int = 0.0
    _dual_last_sp = float(sp)

    kp = float(globals().get("DUAL_%s_TRACK_KP" % mode.upper(), globals().get("DUAL_TRACK_KP", 10.0)))
    ki = float(globals().get("DUAL_%s_TRACK_KI" % mode.upper(), globals().get("DUAL_TRACK_KI", 2.0)))
    kd = float(globals().get("DUAL_%s_TRACK_KD" % mode.upper(), globals().get("DUAL_TRACK_KD", 0.5)))
    landing_pre = {
        "active": False, "reason": "not_heating", "projected_error": None,
        "projected_temperature": None, "eta_to_setpoint_minutes": None,
        "landing_cap": None,
    }
    if mode == "heat":
        landing_pre = _dual_heating_landing_snapshot(
            tin, sp, directed_rate, rate_valid, 0.0, total_max_all
        )
    previous_i = float(_dual_err_int)
    if mode == "heat" and bool(landing_pre.get("active")):
        # Do not let positive integral fight the landing cap. Existing positive
        # windup decays; it is cleared once the rate projection reaches target.
        positive_i = max(0.0, previous_i)
        negative_i = min(0.0, previous_i)
        if float(landing_pre.get("projected_error")) <= 0.0:
            positive_i = 0.0
        else:
            decay = _clip(float(globals().get("DUAL_HEAT_LANDING_I_DECAY", 0.50)), 0.0, 1.0)
            positive_i *= decay
        i_candidate = negative_i + positive_i
    else:
        i_candidate = previous_i + ki * err * dt_s / 3600.0
    i_candidate = _clip(i_candidate, -float(globals().get("DUAL_TRACK_I_CLAMP", 35.0)), float(globals().get("DUAL_TRACK_I_CLAMP", 35.0)))
    if abs(err) <= deadband:
        i_candidate *= (1.0 - float(TRACK_I_LEAK))

    p_term = kp * err
    rate_term = -kd * directed_rate
    raw_unclipped = (
        sustain +
        p_term +
        i_candidate -
        kd * directed_rate
    )
    raw_total = _clip(raw_unclipped, 0.0, total_max_all)
    if raw_total != raw_unclipped:
        if (err > 0.0 and raw_total >= total_max_all) or (err < 0.0 and raw_total <= 0.0):
            i_candidate = float(_dual_err_int)
    _dual_err_int = i_candidate

    total_step = _dual_float_helper(
        globals().get("DUAL_TOTAL_STEP_HELPER", ""),
        globals().get("DUAL_TOTAL_STEP_DEFAULT", 12.0),
        1.0, 40.0, cache,
    )
    reference_total = float(_dual_last_total_target) if isfinite(_dual_last_total_target) else current_total
    total_target = _clip(raw_total, reference_total - total_step, reference_total + total_step)
    total_target = _clip(total_target, 0.0, total_max_all)

    assist_allowed = _dual_update_assist(
        mode, tout, err, directed_rate, rate_valid, tin, sp, humidity, cache
    )
    eligible = _dual_eligible_actions(mode=mode, assist_allowed=assist_allowed)
    if not eligible:
        log.error(
            "Daikin dual: no eligible %s action (configure CLIMATE and ALLOW_HVAC_MODE_CONTROL for standby)",
            mode,
        )
        # Configuration/action anomalies must not cascade into a plant stop.
        # Leave current modes and Demand unchanged until an eligible action is
        # available again.
        _dual_reset_learning_window(
            "no_eligible_action_control_hold", discard_episode=True
        )
        return
    provisional_ctx = _dual_context(tout, total_target, total_max_all, mode=mode, humidity=humidity)
    configured_optimizer_mode = _dual_optimizer_mode(cache)
    optimizer_suspended = not _normal_optimizer_inputs_available()
    optimizer_mode = "disabled" if optimizer_suspended else configured_optimizer_mode

    # Close an episode only at its configured long horizon. Thermal inertia and
    # defrost costs make five-minute efficiency comparisons misleading.
    if _dual_episode is not None:
        ep_duration = now - float(_dual_episode.get("start", now))
        if ep_duration >= _dual_episode_minutes(cache) * 60.0:
            _dual_finalize_episode()

    active_action = _dual_action_by_id(_dual_active_action_id)
    if active_action not in eligible:
        active_action = None

    recommendation = _dual_best_action(provisional_ctx, eligible, mode=mode)
    selection_reason = "continue_episode"
    if active_action is None or _dual_episode is None:
        chosen, recommendation, selection_reason = _dual_choose_action(
            provisional_ctx, eligible, err, directed_rate, rate_valid,
            optimizer_mode=optimizer_mode,
            operating_mode=mode,
        )
        if optimizer_suspended:
            recommendation = None
            selection_reason = "feature_degraded_optimizer_suspended"
        active_action = chosen or _dual_default_action(eligible)
        new_id = str((active_action or {}).get("id", "none"))
        if new_id != str(_dual_active_action_id):
            _dual_action_changed_at = now
        _dual_active_action_id = new_id

    # Comfort abort: stop evaluating a weak experimental split and return to
    # the conservative balanced allocation. The total PI demand still observes
    # the configured step limit.
    abort_error = _dual_abort_error(cache)
    if err > abort_error:
        if _dual_episode is not None:
            _dual_episode["invalid"] = True
            _dual_episode["invalid_reason"] = "comfort_abort"
            _dual_finalize_episode(force_discard=True)
        safe_action = _dual_action_by_id(globals().get("DUAL_DEFAULT_ACTION", "balanced"))
        if safe_action in eligible:
            active_action = safe_action
            _dual_active_action_id = str(safe_action.get("id"))
            selection_reason = "comfort_abort_to_default"

    shares = _dual_action_shares(active_action)
    active_indices = [idx for idx, share in enumerate(shares) if share > 1e-6]
    total_min_action = sum([cap_weights[idx] * bounds[idx][0] for idx in active_indices])
    total_max_action = sum([cap_weights[idx] * bounds[idx][1] for idx in active_indices])
    landing = {
        "active": False, "reason": "not_heating", "projected_error": None,
        "projected_temperature": None, "eta_to_setpoint_minutes": None,
        "landing_cap": None, "landing_fraction": None,
    }
    if mode == "heat":
        landing = _dual_heating_landing_snapshot(
            tin, sp, directed_rate, rate_valid, total_min_action, total_max_action
        )
    learning_before_apply = _dual_learning_status_snapshot(now)
    observed_before_apply = None
    if bool(learning_before_apply.get("learning_allowed")):
        observed_before_apply = _dual_current_total_demand(cache, active_action)
    probe_target = None
    if mode == "heat":
        probe_target = _dual_sustain_probe_target(
            outdoor_ctx, sustain, total_min_action, total_max_action, err,
            directed_rate, rate_valid, observed_before_apply,
            bool(learning_before_apply.get("learning_allowed")), now,
        )

    governed_raw = float(raw_total)
    if mode == "heat" and bool(landing.get("active")):
        governed_raw = min(governed_raw, float(landing.get("landing_cap")))
    if probe_target is not None:
        # A settled probe must hold one exact lower level; the hard comfort
        # stop remains authoritative if temperature rises too far.
        governed_raw = float(probe_target)
    elif (
        mode == "heat" and str(_dual_sustain_probe.get("state")) == "failed" and
        str(_dual_sustain_probe.get("reason")) == "temperature_falling_away" and
        _dual_sustain_probe.get("baseline") is not None and float(err) > 0.0
    ):
        governed_raw = max(governed_raw, float(_dual_sustain_probe.get("baseline")))

    total_target = _clip(governed_raw, reference_total - total_step, reference_total + total_step)
    total_target = _clip(total_target, total_min_action, total_max_action)
    step_limited_total = float(total_target)
    _dual_landing_status = dict(landing)
    _dual_landing_status["raw_total"] = float(raw_total)
    _dual_landing_status["governed_raw"] = float(governed_raw)
    _dual_landing_status["step_limited_total"] = float(step_limited_total)
    _dual_landing_status["p_term"] = float(p_term)
    _dual_landing_status["i_term"] = float(i_candidate)
    _dual_landing_status["rate_term"] = float(rate_term)

    demands = _dual_allocate(total_target, active_action, bounds)
    applied = {}
    for idx, u in enumerate(DAIKINS):
        lo, hi, _icing = bounds[idx]
        if _dual_heat_standby_fan_required(u, mode, active_action):
            _dual_set_heat_standby_fan(u)
            result = None
        else:
            result = _dual_apply_unit_demand(
                u, demands[idx], lo, hi, cache, mode=mode, setpoint=sp
            )
        applied[str(u.get("name", idx))] = round(float(result), 2) if result is not None and isfinite(result) else None

    _dual_last_total_target = float(total_target)

    # Re-evaluate evidence against the action just applied. An allocation
    # change starts a new continuous window and cannot inherit the prior slope.
    learning = _dual_update_learning_gate(
        mode,
        cache=cache,
        action=active_action,
        now=now,
        system_freeze_active=False,
        collect_sample=False,
    )
    observed_total = None
    if bool(learning.get("learning_allowed")):
        observed_total = _dual_current_total_demand(cache, active_action)

    # Learn the total sustaining request only from steady, near-setpoint periods.
    probe_testing = str(_dual_sustain_probe.get("state")) == "testing"
    stable = (
        bool(learning.get("learning_allowed")) and observed_total is not None and
        not probe_testing and not bool(landing.get("active")) and
        rate_valid and abs(float(directed_rate)) <= float(globals().get("DUAL_SUSTAIN_RATE_MAX_CPH", 0.08)) and
        abs(float(err)) <= max(float(deadband), float(globals().get("DUAL_SUSTAIN_ERR_MAX_C", 0.20)))
    )
    if stable:
        alpha = _clip(float(globals().get("DUAL_SUSTAIN_ALPHA", 0.05)), 0.0, 1.0)
        learned = (1.0 - alpha) * sustain + alpha * float(observed_total)
        _dual_sustain_by_ctx[outdoor_ctx] = _clip(learned, total_min_action, total_max_action)
        _dual_save_store(force=False)

    ctx = _dual_context(tout, total_target, total_max_all, mode=mode, humidity=humidity)
    settle_s = max(0.0, float(globals().get("DUAL_ACTION_SETTLE_MINUTES", 15.0))) * 60.0
    action_settled = now - float(_dual_action_changed_at or 0.0) >= settle_s
    if (
        _dual_episode is None and optimizer_mode in ("shadow", "auto") and
        action_settled and bool(learning.get("learning_allowed"))
    ):
        _dual_start_episode(ctx, _dual_active_action_id, sp, tout, total_target, mode=mode, humidity=humidity)

    _total_power, power_valid, per_unit_power = _dual_total_power(cache, active_action)
    _dual_publish_optimizer(ctx, active_action, recommendation, selection_reason, optimizer_mode, power_valid)

    sensor = globals().get("DUAL_TOTAL_DEMAND_SENSOR", "sensor.daikin_dual_total_demand")
    try:
        state.set(
            sensor,
            value=round(float(total_target), 1),
            unit_of_measurement="% points",
            indoor=round(float(tin), 2),
            outdoor=round(float(tout), 2),
            setpoint=round(float(sp), 2),
            error=round(float(err), 3),
            rate_cph=round(float(temp_rate_cph), 4),
            directed_rate=round(float(directed_rate), 4),
            rate_valid=bool(rate_valid),
            rate_span_s=round(float(rate_span), 1),
            rate_samples=int(rate_n),
            sustain=round(float(_dual_sustain_by_ctx.get(outdoor_ctx, sustain)), 2),
            p_term=round(float(p_term), 3),
            i_term=round(float(i_candidate), 3),
            rate_term=round(float(rate_term), 3),
            raw_total=round(float(raw_total), 2),
            governed_raw=round(float(governed_raw), 2),
            projected_error=round(float(landing.get("projected_error")), 4)
                if landing.get("projected_error") is not None else None,
            projected_temperature=round(float(landing.get("projected_temperature")), 3)
                if landing.get("projected_temperature") is not None else None,
            eta_to_setpoint_minutes=round(float(landing.get("eta_to_setpoint_minutes")), 2)
                if landing.get("eta_to_setpoint_minutes") is not None else None,
            landing_active=bool(landing.get("active")),
            landing_cap=round(float(landing.get("landing_cap")), 2)
                if landing.get("landing_cap") is not None else None,
            step_limited_total=round(float(step_limited_total), 2),
            actual_select_demand=round(float(_dual_current_total_demand(cache, active_action)), 2),
            minimum_on_remaining_seconds=max(
                0.0,
                float(_dual_supervisor_limits(cache).get("min_on_min", 0.0)) * 60.0 -
                max(0.0, now - float(_dual_zone_started_at or now)),
            ),
            sustain_probe=_dual_probe_snapshot(),
            cdp_release=_dual_cdp_snapshot(),
            cdp_temperature_curve=_dual_cdp_curve_snapshot(),
            action=str(_dual_active_action_id),
            requested_demands=[round(float(v), 2) if v is not None else None for v in demands],
            applied_demands=applied,
            power_w=per_unit_power,
            optimizer_mode=optimizer_mode,
            operating_mode=mode,
            physical_hvac_mode=_dual_physical_hvac_mode(mode),
            hvac_target_temperature=round(float(_dual_climate_target_temperature(mode)), 1) if _dual_climate_target_temperature(mode) is not None else None,
            humidity=round(float(humidity), 2) if isfinite(humidity) else None,
            humidity_target=round(float(humidity_target), 1),
            second_unit_assist=bool(assist_allowed),
            heating_lead_unit=str(
                (_dual_heating_lead_unit(cache) or {}).get("name", "unavailable")
            ),
            daikin1_heat_circulation=bool(any([
                _dual_heat_standby_fan_required(u, mode, active_action)
                for u in DAIKINS
            ])),
            configured_units=len(DAIKINS),
            single_unit_fallback=(len(DAIKINS) == 1),
            cold_assistance=bool(mode == "heat" and assist_allowed),
            actual_running=bool(learning.get("actual_running")),
            learning_allowed=bool(learning.get("learning_allowed")),
            learning_block_reason=learning.get("learning_block_reason"),
            active_learning_span_seconds=round(float(learning.get("active_learning_span_seconds", 0.0)), 1),
            learning_required_seconds=round(float(learning.get("learning_required_seconds", 0.0)), 1),
            observed_total_demand=round(float(observed_total), 2) if observed_total is not None else None,
        )
    except Exception as e:
        log.debug("Daikin dual: failed to publish total demand: %s", e)
    _dual_publish_landing_status(_dual_landing_status, total_target)

    log.info(
        "Daikin dual: mode=%s Tin=%.2f SP=%.2f Tout=%.2f RH=%s err=%.2f rate=%.3f total=%.1f action=%s demands=%s",
        mode, tin, sp, tout, round(float(humidity), 1) if isfinite(humidity) else None,
        err, directed_rate, total_target, str(_dual_active_action_id), str(applied),
    )
    _dual_publish_learning_status(mode)

# ============================================================
# 5) SERIALIZED STARTUP + HOUSEKEEPING
# ============================================================

_housekeeping_last = {}


def _housekeeping_due(name, interval_s, now):
    last = float(_housekeeping_last.get(str(name)) or 0.0)
    if last > 0.0 and float(now) - last < max(0.0, float(interval_s) - 0.5):
        return False
    _housekeeping_last[str(name)] = float(now)
    return True


@time_trigger("startup")
def _ml_startup_ok():
    """Load persistent state before any path is allowed to save or command."""
    startup_started = time.time()
    _runtime_store_load()
    try:
        state.persist(
            globals().get(
                "DUAL_ZONE_STORE_ENTITY", "pyscript.daikin_dual_zone_store"
            )
        )
    except Exception:
        pass
    _dual_load_store()
    _initialize_diagnostic_entities()
    cache = _TickCache()
    _sensor_report_poll(prime_only=True)
    _safety_evaluate(cache)
    _refresh_all_unit_runtimes(cache, startup_started)
    _validate_configuration_impl(publish=True)
    _publish_health()
    _runtime_store_save(force=True)
    _heartbeat_success(startup_started, note="startup")
    log.info(
        "Daikin shared-zone controller: startup loaded for %d unit(s)",
        len(DAIKINS),
    )


def _separate_areas_enabled(cache=None):
    cache = cache or _TickCache()
    helper = globals().get(
        "DUAL_SEPARATE_AREAS_HELPER", "input_boolean.daikin_separate_areas"
    )
    return cache.get_str(helper, default="off").strip().lower() == "on"


def _separate_area_summer_enabled(u, cache=None):
    if not _separate_areas_enabled(cache):
        return False
    cache = cache or _TickCache()
    helper = (u or {}).get("SEPARATE_SUMMER_HELPER")
    return bool(helper and cache.get_str(helper, default="off").strip().lower() == "on")


def _separate_area_configuration_issues():
    issues = []
    sensors = []
    for u in DAIKINS:
        name = _unit_name(u)
        indoor = u.get("SEPARATE_INDOOR")
        if not indoor:
            issues.append("%s_separate_indoor_missing" % name)
        else:
            sensors.append(str(indoor))
        for key in ("SEPARATE_SETPOINT_HELPER", "SEPARATE_MODE_HELPER"):
            if not u.get(key):
                issues.append("%s_%s_missing" % (name, key.lower()))
    if len(sensors) > 1 and len(set(sensors)) != len(sensors):
        issues.append("separate_areas_require_unique_indoor_sensors")
    return issues


def _separate_area_setpoint(u, cache=None):
    cache = cache or _TickCache()
    helper = u.get("SEPARATE_SETPOINT_HELPER")
    value = cache.get_float(helper, default=float("nan")) if helper else float("nan")
    if isfinite(value):
        return _clip(value, 10.0, 35.0)
    shared = cache.get_float("input_number.daikin_setpoint", default=22.0)
    return _clip(shared, 10.0, 35.0)


def _separate_area_requested_mode(u, cache=None):
    cache = cache or _TickCache()
    helper = u.get("SEPARATE_MODE_HELPER")
    raw = cache.get_str(helper, default="off").strip().lower()
    aliases = {"off + fan": "fan", "fan_only": "fan"}
    raw = aliases.get(raw, raw)
    allowed = [str(value).lower() for value in (u.get("SEPARATE_ALLOWED_MODES") or ["off", "heat"])]
    return raw if raw in allowed else "off"


def _separate_area_temperature(u, cache=None):
    cache = cache or _TickCache()
    entity = u.get("SEPARATE_INDOOR")
    info = _sensor_freshness(
        "separate_indoor_%s" % _unit_name(u), entity,
        globals().get("DUAL_INDOOR_MAX_AGE_S", 300.0), cache, True, True,
    )
    value = info.get("value")
    return float(value) if info.get("usable_for_control") and value is not None and isfinite(value) else float("nan")


def _separate_area_resolve_mode(u, requested, tin, sp, runtime, cache=None):
    limits = _dual_supervisor_limits(cache or _TickCache())
    active_mode = str(runtime.get("mode") or "off") if runtime.get("active") else "off"
    if requested in ("off", "fan", "dry"):
        return requested
    if requested == "auto":
        if active_mode == "heat" and tin < sp + float(limits["heat_off"]):
            return "heat"
        if active_mode == "cool" and tin > sp - float(limits["cool_off"]):
            return "cool"
        if tin <= sp - float(limits["auto_heat_on"]):
            return "heat"
        if "cool" in [str(v).lower() for v in (u.get("SEPARATE_ALLOWED_MODES") or [])] and tin >= sp + float(limits["auto_cool_on"]):
            return "cool"
        return "off"
    if requested == "heat":
        if active_mode == "heat":
            return "heat" if tin < sp + float(limits["heat_off"]) else "off"
        return "heat" if tin <= sp - float(limits["heat_on"]) else "off"
    if requested == "cool":
        if active_mode == "cool":
            return "cool" if tin > sp - float(limits["cool_off"]) else "off"
        return "cool" if tin >= sp + float(limits["cool_on"]) else "off"
    return "off"


def _separate_area_minimum_minutes(u, key, default_helper, default_value, cache=None):
    cache = cache or _TickCache()
    helper = u.get(key)
    if helper:
        value = cache.get_float(helper, default=float("nan"))
        if isfinite(value):
            return _clip(value, 0.0, 240.0)
    return _dual_float_helper(default_helper, default_value, 0.0, 240.0, cache)


def _separate_area_update_learning(u, mode, demand, cache=None):
    cache = cache or _TickCache()
    info = _unit_hvac_state(u, cache)
    if str(mode) != "heat" or not (
        str(info.get("action") or "").lower() in ("heat", "heating") or
        info.get("running") is True
    ):
        return
    power_ent = u.get("POWER_SENSOR")
    cop_ent = u.get("COP_SENSOR")
    power = cache.get_float(power_ent, default=float("nan")) if power_ent else float("nan")
    cop = cache.get_float(cop_ent, default=float("nan")) if cop_ent else float("nan")
    tout = _dual_outdoor_temperature(cache, allow_conservative_fallback=False)
    if not (isfinite(power) and power > 0.0 and isfinite(cop) and cop > 0.0 and isfinite(tout)):
        return
    bucket = int(float(tout) // 2.0) * 2
    key = "%s|heat|%s" % (_unit_name(u), bucket)
    old = dict(_separate_area_learning.get(key) or {})
    n = int(old.get("samples", 0)) + 1
    old["samples"] = n
    old["mean_demand"] = float(old.get("mean_demand", 0.0)) + (float(demand) - float(old.get("mean_demand", 0.0))) / n
    old["mean_power_w"] = float(old.get("mean_power_w", 0.0)) + (float(power) - float(old.get("mean_power_w", 0.0))) / n
    old["mean_cop"] = float(old.get("mean_cop", 0.0)) + (float(cop) - float(old.get("mean_cop", 0.0))) / n
    old["last_seen"] = time.time()
    _separate_area_learning[key] = old


def _publish_separate_areas_status(cache=None, issues=None):
    cache = cache or _TickCache()
    sensor = globals().get(
        "DUAL_SEPARATE_AREAS_SENSOR", "sensor.daikin_separate_areas_status"
    )
    enabled = _separate_areas_enabled(cache)
    zones = {}
    for u in DAIKINS:
        name = _unit_name(u)
        runtime = dict(_separate_area_state.get(name) or {})
        runtime["area_name"] = str(u.get("SEPARATE_AREA_NAME") or name)
        runtime["indoor_sensor"] = u.get("SEPARATE_INDOOR")
        runtime["setpoint_helper"] = u.get("SEPARATE_SETPOINT_HELPER")
        runtime["mode_helper"] = u.get("SEPARATE_MODE_HELPER")
        zones[name] = runtime
    try:
        state.set(
            sensor,
            value="separate_areas" if enabled and not issues else "configuration_error" if enabled else "shared_zone",
            separate_areas=bool(enabled),
            control_topology="separate_areas" if enabled else "shared_zone",
            configuration_issues=list(issues or []),
            zones=zones,
            independent_learning=dict(_separate_area_learning),
            shared_allocator_active=not bool(enabled),
            heating_lead_active=not bool(enabled),
            dual_assistance_active=not bool(enabled),
            controller_version=globals().get("DUAL_CONTROLLER_VERSION"),
        )
    except Exception as e:
        log.debug("Daikin separate-area status publish failed: %s", e)


def _run_separate_areas(cache=None, now=None):
    """Run independent per-unit thermostats without shared allocation."""
    global _separate_topology_previous, _dual_err_int, _dual_episode
    cache = cache or _TickCache()
    now = time.time() if now is None else float(now)
    issues = _separate_area_configuration_issues()
    if issues:
        _publish_separate_areas_status(cache, issues)
        return False
    if not _separate_topology_previous:
        _dual_err_int = 0.0
        _dual_episode = None
        _dual_discard_episode("separate_areas_enabled")
        for u in DAIKINS:
            _desired_update_unit(
                u, reason="separate_areas_transition", force_generation=True
            )
        _separate_topology_previous = True

    tout = _dual_outdoor_temperature(cache, allow_conservative_fallback=True)
    for u in DAIKINS:
        name = _unit_name(u)
        runtime = dict(_separate_area_state.get(name) or {
            "active": False, "mode": "off", "integral": 0.0,
            "last_control_at": 0.0, "started_at": 0.0,
            "stopped_at": now - 86400.0,
        })
        requested = _separate_area_requested_mode(u, cache)
        tin = _separate_area_temperature(u, cache)
        sp = _separate_area_setpoint(u, cache)
        runtime.update({
            "requested_mode": requested,
            "indoor_temperature": round(float(tin), 3) if isfinite(tin) else None,
            "setpoint": round(float(sp), 2),
            "sensor_valid": bool(isfinite(tin)),
        })
        if not isfinite(tin):
            runtime["reason"] = "indoor_sensor_invalid"
            _separate_area_state[name] = runtime
            continue
        target_mode = _separate_area_resolve_mode(u, requested, tin, sp, runtime, cache)
        min_on = _separate_area_minimum_minutes(
            u, "SEPARATE_MIN_ON_HELPER", globals().get("DUAL_MIN_ON_MINUTES_HELPER", ""),
            globals().get("DUAL_MIN_ON_MINUTES_DEFAULT", 15.0), cache,
        ) * 60.0
        min_off = _separate_area_minimum_minutes(
            u, "SEPARATE_MIN_OFF_HELPER", globals().get("DUAL_MIN_OFF_MINUTES_HELPER", ""),
            globals().get("DUAL_MIN_OFF_MINUTES_DEFAULT", 5.0), cache,
        ) * 60.0
        if runtime.get("active") and target_mode == "off" and now - float(runtime.get("started_at") or now) < min_on:
            target_mode = str(runtime.get("mode") or "heat")
            runtime["reason"] = "minimum_on_time"
        if not runtime.get("active") and target_mode in ("heat", "cool", "dry") and now - float(runtime.get("stopped_at") or 0.0) < min_off:
            target_mode = "off"
            runtime["reason"] = "minimum_off_time"

        if target_mode == "fan":
            _desired_update_unit(
                u, active=True, mode="fan_only", target=None,
                demand_option=None, quiet=None,
                fan_mode=str(globals().get("DUAL_SHARED_FAN_MODE", "lowMedium")),
                reason="separate_area_fan", force_stop=False,
            )
            runtime.update({"active": False, "mode": "fan", "applied_demand": None, "reason": "fan"})
            _separate_area_state[name] = runtime
            continue
        if target_mode == "off":
            if runtime.get("active"):
                runtime["stopped_at"] = now
            _dual_set_hvac_active(u, False, "off", None)
            runtime.update({"active": False, "mode": "off", "applied_demand": None, "integral": 0.0})
            runtime.setdefault("reason", "comfort_satisfied" if requested != "off" else "requested_off")
            _separate_area_state[name] = runtime
            continue

        defrosting, _liquid = _read_defrosting(u, cache)
        if defrosting is True:
            runtime["reason"] = "unit_defrosting_demand_held"
            runtime["defrosting"] = True
            _separate_area_state[name] = runtime
            continue
        runtime["defrosting"] = False
        if not runtime.get("active") or str(runtime.get("mode")) != target_mode:
            runtime["started_at"] = now
            runtime["integral"] = 0.0
        dt = max(0.0, min(300.0, now - float(runtime.get("last_control_at") or now)))
        error = float(sp) - float(tin) if target_mode == "heat" else float(tin) - float(sp)
        kp = float(globals().get("DUAL_SEPARATE_KP", 25.0))
        ki = float(globals().get("DUAL_SEPARATE_KI", 3.0))
        kd = float(globals().get("DUAL_SEPARATE_KD", 4.0))
        previous_tin = runtime.get("last_indoor_temperature")
        raw_rate_cph = 0.0
        rate_valid = bool(
            previous_tin is not None and dt >= 10.0 and isfinite(float(previous_tin))
        )
        if rate_valid:
            raw_rate_cph = _clip(
                (float(tin) - float(previous_tin)) * 3600.0 / dt,
                -float(globals().get("DUAL_RATE_CLAMP_CPH", 2.0)),
                float(globals().get("DUAL_RATE_CLAMP_CPH", 2.0)),
            )
        directed_rate = raw_rate_cph if target_mode == "heat" else -raw_rate_cph
        integral = _clip(
            float(runtime.get("integral") or 0.0) + ki * error * dt / 60.0,
            -float(globals().get("DUAL_SEPARATE_I_CLAMP", 25.0)),
            float(globals().get("DUAL_SEPARATE_I_CLAMP", 25.0)),
        )
        runtime["integral"] = integral
        lower, upper, _icing = _dual_unit_bounds(u, tout, tin, sp - 3.0, cache, mode=target_mode)
        requested_demand = _clip(
            float(lower) + kp * max(0.0, error) + integral - kd * directed_rate,
            lower, upper,
        )
        prediction_minutes = max(
            1.0, float(globals().get("DUAL_SEPARATE_PREDICTION_MINUTES", 10.0))
        )
        projected_tin = (
            float(tin) + raw_rate_cph * prediction_minutes / 60.0
            if rate_valid else float(tin)
        )
        projected_overshoot = bool(
            rate_valid and (
                (target_mode == "heat" and projected_tin >= float(sp)) or
                (target_mode in ("cool", "dry") and projected_tin <= float(sp))
            )
        )
        if projected_overshoot:
            current_demand = _dual_current_effective_demand(u, cache)
            requested_demand = min(
                requested_demand,
                max(float(lower), float(current_demand) - float(globals().get("DUAL_CDP_DEMAND_STEP", 5.0))),
            )
        applied = _dual_apply_unit_demand(
            u, requested_demand, lower, upper, cache,
            mode=target_mode, setpoint=sp,
        )
        runtime.update({
            "active": True, "mode": target_mode, "last_control_at": now,
            "last_indoor_temperature": float(tin),
            "error": round(float(error), 3),
            "temperature_rate_cph": round(float(raw_rate_cph), 4) if rate_valid else None,
            "projected_temperature": round(float(projected_tin), 3),
            "predictive_landing_active": projected_overshoot,
            "requested_demand": round(float(requested_demand), 2),
            "applied_demand": round(float(applied), 2) if applied is not None else None,
            "summer_mode": _separate_area_summer_enabled(u, cache),
            "reason": "independent_%s_control" % target_mode,
        })
        _separate_area_state[name] = runtime
        if applied is not None:
            _separate_area_update_learning(u, target_mode, applied, cache)
    _publish_separate_areas_status(cache, [])
    _dual_publish_cdp_curve(cache)
    return True


def _daikin_dual_housekeeping_impl(now=None):
    """One serialized scheduler for safety, control, learning, and commands."""
    global _separate_topology_previous
    now = time.time() if now is None else float(now)
    cache = _TickCache()

    # The explicit fan-RPM calibration service temporarily owns both indoor
    # units. Do not let normal desired-state reconciliation overwrite its
    # fan_only/fan-speed sequence.
    if bool(_dual_fan_calibration_active):
        _dual_publish_fan_calibration()
        return

    # Snapshot/report bookkeeping and safety are evaluated exactly once.
    _sensor_report_poll(prime_only=False)
    _safety_evaluate(cache)
    _refresh_all_unit_runtimes(cache, now)

    if _separate_areas_enabled(cache):
        if _housekeeping_due(
            "separate_areas",
            float(globals().get("DUAL_SEPARATE_CONTROL_INTERVAL_S", 20.0)),
            now,
        ):
            _run_separate_areas(cache, now)
        _reconcile_all_desired(cache, now)
        _command_monitor_tick()
        _publish_health()
        _runtime_store_save(force=False)
        return
    if _separate_topology_previous:
        _separate_topology_previous = False
        _housekeeping_last["supervisor"] = 0.0
        _housekeeping_last["adaptive"] = 0.0
        _housekeeping_last["pi"] = 0.0
        _publish_separate_areas_status(cache, [])
    system_freeze = _update_system_defrost_freeze()

    supervisor_info = None
    requested_now = _requested_hvac_mode(cache)
    summer_now = _summer_mode_enabled(cache)
    immediate_stop_intent = bool(
        requested_now == "off" or
        (
            not summer_now and
            str(_dual_active_mode) in ("cool", "dry")
        )
    )
    if (
        immediate_stop_intent or
        _housekeeping_due("supervisor", 60.0, now)
    ):
        _housekeeping_last["supervisor"] = now
        if not system_freeze:
            supervisor_info = _dual_supervisor_tick(cache)
        _dual_optimizer_sample(system_freeze_active=system_freeze)
        _stats_tick(cache, system_freeze_active=system_freeze)
        if supervisor_info is not None:
            _dual_publish_supervisor(supervisor_info)

    if (
        not system_freeze and
        _housekeeping_due("adaptive", 20.0, now)
    ):
        if (
            _dual_zone_active and str(_dual_active_mode) == "dry" and
            _dual_coordinated_dry_available()
        ):
            _dual_run_coordinated_dry(cache)
        else:
            _dual_fast_landing_tick(cache)
        _dual_cdp_fast_tick(cache)

    if not system_freeze and _housekeeping_due("pi", 300.0, now):
        _requested, effective_mode, _summer, _block = _effective_hvac_mode(cache)
        if (
            _dual_zone_active and
            effective_mode in ("heat", "cool", "dry") and
            not _safety_degraded_active and
            _safety_health != "fault"
        ):
            _run_dual_zone(effective_mode)
        else:
            _dual_discard_episode("plant_inactive")

    nordpool_interval = (
        60.0 * max(
            1.0,
            float(globals().get("NORDPOOL_WINDOW_UPDATE_EVERY_MIN", 15.0)),
        )
    )
    if _housekeeping_due("nordpool_window", nordpool_interval, now):
        _update_nordpool_avg_window_hours_multi()

    # Expired post-defrost holds simply release the unit back to normal PI
    # allocation. The next adaptive/PI decision updates its desired Demand.
    hold_expired = False
    for name in list(_dual_post_defrost_holds.keys()):
        hold = _dual_post_defrost_holds.get(name) or {}
        if now >= float(hold.get("until") or 0.0):
            _dual_post_defrost_holds.pop(name, None)
            hold_expired = True
    if hold_expired:
        _housekeeping_last["pi"] = 0.0
        _housekeeping_last["adaptive"] = 0.0

    # New decisions supersede old generations before any retry is attempted.
    _reconcile_all_desired(cache, now)
    _command_monitor_tick()
    _publish_health()
    _runtime_store_save(force=False)


@time_trigger("period(now, 10sec)")
def daikin_dual_housekeeping(**kwargs):
    """The controller's only recurring trigger."""
    started_at = time.time()
    try:
        task.unique("daikin_dual_housekeeping")
    except Exception:
        pass
    try:
        _daikin_dual_housekeeping_impl(started_at)
        _heartbeat_success(started_at, note="housekeeping")
    except Exception as e:
        _heartbeat_error(started_at, e)
        log.error("Daikin dual housekeeping failure: %s", e)


# ============================================================
# 6) SERVICES
# ============================================================
def _validate_configuration_impl(publish=True):
    issues = []
    warnings = []
    checked = {}
    names = []
    confirmed_outdoor_sensors = []
    for configured_entity in list(
        globals().get("DUAL_CONFIRMED_OUTDOOR_SENSORS", []) or []
    ):
        normalized_entity = str(configured_entity or "").strip().lower()
        if normalized_entity and normalized_entity not in confirmed_outdoor_sensors:
            confirmed_outdoor_sensors.append(normalized_entity)
    heat_target = _dual_climate_target_temperature("heat")
    cool_target = _dual_climate_target_temperature("cool")
    dry_target = _dual_climate_target_temperature("dry")
    if heat_target is None or abs(float(heat_target) - 31.0) >= 0.05:
        issues.append("fixed_heat_hvac_target_must_be_31")
    if cool_target is None or abs(float(cool_target) - 16.0) >= 0.05:
        issues.append("fixed_cool_hvac_target_must_be_16")
    if dry_target is None or abs(float(dry_target) - 16.0) >= 0.05:
        issues.append("fixed_dry_hvac_target_must_be_16")
    if _dual_physical_hvac_mode("cool") != "cool":
        issues.append("cooling_physical_hvac_mode_must_be_cool")
    if _dual_physical_hvac_mode("dry") != "dry":
        issues.append("drying_physical_hvac_mode_must_be_dry")
    cooling_unit = _dual_cooling_unit()
    drying_unit = _dual_drying_unit()
    reheat_unit = _dual_dry_reheat_unit()
    if len(DAIKINS) >= 2 and cooling_unit is None:
        issues.append("configured_cooling_unit_missing")
    if len(DAIKINS) >= 2 and drying_unit is None:
        issues.append("configured_drying_unit_missing")
    if len(DAIKINS) >= 2 and reheat_unit is None:
        issues.append("configured_dry_reheat_unit_missing")
    if drying_unit is not None and reheat_unit is drying_unit:
        issues.append("drying_and_reheat_units_must_be_distinct")
    if not DAIKINS:
        issues.append("no_daikin_units_configured")
    for u in DAIKINS:
        name = str(u.get("name") or "unnamed")
        if name in names:
            issues.append("duplicate_unit_name:%s" % name)
        names.append(name)
        unit_result = {
            "entities": {}, "hvac_modes": [], "fan_modes": [],
            "demand_options": [],
        }
        for key in ("CLIMATE", "SELECT", "INDOOR", "OUTDOOR", "LIQUID"):
            ent = u.get(key)
            exists = _entity_exists(ent)
            unit_result["entities"][key] = {"entity_id": ent, "exists": bool(exists)}
            if key in ("CLIMATE", "SELECT", "INDOOR") and not exists:
                issues.append("%s_%s_missing:%s" % (name, key.lower(), str(ent)))
            elif key in ("OUTDOOR", "LIQUID") and ent and not exists:
                warnings.append("%s_%s_missing:%s" % (name, key.lower(), str(ent)))

        climate_ent = u.get("CLIMATE")
        if climate_ent and _entity_exists(climate_ent):
            attrs = state.getattr(climate_ent) or {}
            modes = list(attrs.get("hvac_modes") or [])
            fan_modes = list(attrs.get("fan_modes") or [])
            unit_result["hvac_modes"] = modes
            unit_result["fan_modes"] = fan_modes
            try:
                climate_min_temp = float(attrs.get("min_temp"))
                if not isfinite(climate_min_temp):
                    climate_min_temp = None
            except Exception:
                climate_min_temp = None
            try:
                climate_max_temp = float(attrs.get("max_temp"))
                if not isfinite(climate_max_temp):
                    climate_max_temp = None
            except Exception:
                climate_max_temp = None
            unit_result["min_temp"] = climate_min_temp
            unit_result["max_temp"] = climate_max_temp
            for required_mode in ("heat",):
                if required_mode not in modes:
                    issues.append("%s_climate_missing_mode:%s" % (name, required_mode))
            if u is cooling_unit and "cool" not in [str(value).lower() for value in modes]:
                issues.append("%s_climate_missing_cooling_mode:cool" % name)
            if u is drying_unit and "cool" not in [str(value).lower() for value in modes]:
                issues.append("%s_climate_missing_drying_mode:cool" % name)
            default_fan = str(globals().get("DUAL_DEFAULT_FAN_MODE", "auto"))
            resolved_default_fan = _dual_resolve_fan_mode(u, default_fan, _TickCache())
            if not fan_modes:
                issues.append("%s_climate_missing_fan_modes" % name)
            elif str(resolved_default_fan).lower() not in [str(value).lower() for value in fan_modes]:
                issues.append("%s_climate_missing_default_fan_mode:%s" % (name, default_fan))
            heat_fan = str(globals().get("DUAL_HEAT_FAN_MODE", "mediumHigh"))
            resolved_heat_fan = _dual_resolve_fan_mode(u, heat_fan, _TickCache())
            unit_result["resolved_heat_fan_mode"] = resolved_heat_fan
            if resolved_heat_fan is None:
                issues.append("%s_climate_missing_heat_fan_mode:%s" % (name, heat_fan))
            if u is drying_unit:
                dry_fan = str(globals().get("DUAL_DRY_FAN_MODE", "lowMedium"))
                resolved_dry_fan = _dual_resolve_fan_mode(u, dry_fan, _TickCache())
                unit_result["resolved_dry_fan_mode"] = resolved_dry_fan
                if resolved_dry_fan is None:
                    issues.append("%s_climate_missing_dry_fan_mode:%s" % (name, dry_fan))
            if u is cooling_unit:
                cool_fan = str(globals().get("DUAL_COOL_FAN_MODE", "lowMedium"))
                resolved_cool_fan = _dual_resolve_fan_mode(u, cool_fan, _TickCache())
                unit_result["resolved_cool_fan_mode"] = resolved_cool_fan
                if resolved_cool_fan is None:
                    issues.append("%s_climate_missing_cool_fan_mode:%s" % (name, cool_fan))
            required_summer_targets = []
            if u is cooling_unit and cool_target is not None:
                required_summer_targets.append(float(cool_target))
            if u is drying_unit and dry_target is not None:
                required_summer_targets.append(float(dry_target))
            if required_summer_targets and climate_min_temp is not None:
                required_minimum = min(required_summer_targets)
                if climate_min_temp > required_minimum + 0.05:
                    issues.append(
                        "%s_climate_min_temp_%.1f_above_required_summer_target_%.1f" % (
                            name, climate_min_temp, required_minimum,
                        )
                    )
            if bool(globals().get("DUAL_COIL_DRY_ENABLED", True)) and bool(u.get("ALLOW_FAN_ONLY_COIL_DRY", True)):
                if str(globals().get("DUAL_COIL_DRY_HVAC_MODE", "fan_only")) not in modes:
                    warnings.append("%s_climate_missing_fan_only_coil_dry_disabled" % name)
            normalized_modes = [str(value).strip().lower() for value in modes]
            if "fan_only" not in normalized_modes:
                issues.append("%s_climate_missing_shared_off_fan_mode:fan_only" % name)
            shared_fan = str(globals().get("DUAL_SHARED_FAN_MODE", "lowMedium"))
            resolved_shared_fan = _dual_resolve_fan_mode(u, shared_fan, _TickCache())
            unit_result["resolved_shared_fan_mode"] = resolved_shared_fan
            if resolved_shared_fan is None:
                issues.append("%s_climate_missing_shared_fan_mode:%s" % (name, shared_fan))

        select_ent = u.get("SELECT")
        if select_ent and _entity_exists(select_ent):
            nums, _has_pct = _select_options_nums(select_ent)
            unit_result["demand_options"] = list(nums)
            if not nums:
                issues.append("%s_demand_select_has_no_numeric_options" % name)
            else:
                if min(nums) > 30.0:
                    warnings.append("%s_demand_min_above_30" % name)
                if max(nums) < 100.0:
                    issues.append("%s_demand_max_below_100" % name)

        power_ent = u.get("POWER_SENSOR")
        if power_ent:
            if not _entity_exists(power_ent):
                warnings.append("%s_power_sensor_missing_optimizer_disabled" % name)
            else:
                attrs = state.getattr(power_ent) or {}
                unit = str(attrs.get("unit_of_measurement") or "")
                unit_result["power_unit"] = unit
                if unit.lower() not in ("w", "kw"):
                    warnings.append("%s_power_unit_not_w_or_kw:%s" % (name, unit))
                if unit.lower() == "kw" and abs(float(u.get("POWER_SENSOR_SCALE", 1.0)) - 1000.0) > 0.01:
                    issues.append("%s_power_kw_requires_scale_1000" % name)
        cop_ent = u.get("COP_SENSOR")
        unit_result["entities"]["COP_SENSOR"] = {
            "entity_id": cop_ent,
            "exists": bool(_entity_exists(cop_ent)) if cop_ent else False,
        }
        if not cop_ent:
            warnings.append("%s_cop_sensor_not_configured_energy_score_fallback" % name)
        elif not _entity_exists(cop_ent):
            warnings.append("%s_cop_sensor_missing_energy_score_fallback:%s" % (name, cop_ent))
        else:
            try:
                cop_value = float(state.get(cop_ent))
            except Exception:
                cop_value = float("nan")
            unit_result["cop_value"] = cop_value if isfinite(cop_value) else None
            if not isfinite(cop_value):
                warnings.append("%s_cop_sensor_not_numeric:%s" % (name, cop_ent))
        separate_entities = {}
        for separate_key in (
            "SEPARATE_INDOOR", "SEPARATE_SETPOINT_HELPER",
            "SEPARATE_MODE_HELPER", "SEPARATE_SUMMER_HELPER",
            "SEPARATE_MIN_ON_HELPER", "SEPARATE_MIN_OFF_HELPER",
        ):
            separate_ent = u.get(separate_key)
            separate_exists = bool(_entity_exists(separate_ent)) if separate_ent else False
            separate_entities[separate_key] = {
                "entity_id": separate_ent, "exists": separate_exists,
            }
            if _separate_areas_enabled(_TickCache()) and not separate_exists:
                issues.append(
                    "%s_%s_missing:%s" % (
                        name, separate_key.lower(), str(separate_ent),
                    )
                )
        unit_result["separate_area"] = {
            "name": u.get("SEPARATE_AREA_NAME"),
            "entities": separate_entities,
            "allowed_modes": list(u.get("SEPARATE_ALLOWED_MODES") or []),
        }
        outdoor_ent = str(u.get("OUTDOOR") or "").strip()
        outdoor_ent_normalized = outdoor_ent.lower()
        outdoor_name_looks_like_supply_air = (
            "tulo" in outdoor_ent_normalized and
            "ulko" not in outdoor_ent_normalized
        )
        outdoor_sensor_confirmed_external = (
            outdoor_ent_normalized in confirmed_outdoor_sensors
        )
        unit_result["outdoor_name_looks_like_supply_air"] = bool(
            outdoor_name_looks_like_supply_air
        )
        unit_result["outdoor_sensor_confirmed_external"] = bool(
            outdoor_sensor_confirmed_external
        )
        if (
            outdoor_name_looks_like_supply_air and
            not outdoor_sensor_confirmed_external
        ):
            warnings.append("%s_outdoor_entity_name_looks_like_supply_air:%s" % (name, outdoor_ent))
        checked[name] = unit_result

    if _separate_areas_enabled(_TickCache()):
        for separate_issue in _separate_area_configuration_issues():
            if separate_issue not in issues:
                issues.append(separate_issue)

    combined_power = globals().get("DUAL_TOTAL_POWER_SENSOR")
    if combined_power:
        if not _entity_exists(combined_power):
            warnings.append("combined_power_sensor_missing_optimizer_disabled")
        else:
            attrs = state.getattr(combined_power) or {}
            unit = str(attrs.get("unit_of_measurement") or "")
            if unit.lower() not in ("w", "kw"):
                warnings.append("combined_power_unit_not_w_or_kw:%s" % unit)
            if unit.lower() == "kw" and abs(float(globals().get("DUAL_TOTAL_POWER_SENSOR_SCALE", 1.0)) - 1000.0) > 0.01:
                issues.append("combined_power_kw_requires_scale_1000")

    helper_entities = [
        globals().get("DUAL_HVAC_MODE_HELPER"),
        globals().get("DUAL_SUMMER_MODE_HELPER"),
        globals().get("DUAL_DAIKIN1_HEATING_LEAD_HELPER"),
        globals().get("DUAL_OPTIMIZER_MODE_HELPER"),
        globals().get("DUAL_CONTROLLER_SHADOW_HELPER"),
        globals().get("DUAL_MIN_ON_MINUTES_HELPER"),
        globals().get("DUAL_MIN_OFF_MINUTES_HELPER"),
        globals().get("DUAL_HUMIDITY_TARGET_HELPER"),
    ]
    for ent in helper_entities:
        if ent and not _entity_exists(ent):
            issues.append("helper_missing:%s" % str(ent))
    if not _entity_exists("input_boolean.daikin_watchdog_enabled"):
        warnings.append("watchdog_package_not_loaded")
    if not _entity_exists("group.daikin_watchdog_climates"):
        warnings.append("watchdog_climate_group_not_loaded_or_not_configured")
    else:
        try:
            group_attrs = state.getattr("group.daikin_watchdog_climates") or {}
            members = list(group_attrs.get("entity_id") or [])
        except Exception:
            members = []
        if not members:
            issues.append("watchdog_climate_group_empty")
        for member in members:
            if "replace_with" in str(member) or not _entity_exists(member):
                issues.append("watchdog_climate_group_invalid_member:%s" % str(member))

    value = "fail" if issues else "warning" if warnings else "pass"
    result = {
        "result": value,
        "checked_at": time.time(),
        "configured_units": len(DAIKINS),
        "single_unit_fallback": len(DAIKINS) == 1,
        "issues": issues,
        "warnings": warnings,
        "units": checked,
        "fixed_heat_hvac_target": heat_target,
        "fixed_cool_hvac_target": cool_target,
        "fixed_dry_hvac_target": dry_target,
        "cooling_physical_hvac_mode": _dual_physical_hvac_mode("cool"),
        "drying_physical_hvac_mode": _dual_physical_hvac_mode("dry"),
        "cooling_unit": str(cooling_unit.get("name")) if cooling_unit else None,
        "drying_unit": str(drying_unit.get("name")) if drying_unit else None,
        "summer_reheat_unit": str(reheat_unit.get("name")) if reheat_unit else None,
        "heating_fan_mode": str(globals().get("DUAL_HEAT_FAN_MODE", "mediumHigh")),
        "cooling_fan_mode": str(globals().get("DUAL_COOL_FAN_MODE", "lowMedium")),
        "drying_fan_mode": str(globals().get("DUAL_DRY_FAN_MODE", "lowMedium")),
        "default_fan_mode": str(globals().get("DUAL_DEFAULT_FAN_MODE", "auto")),
        "coordinated_reheat_drying_available": bool(_dual_coordinated_dry_available()),
        "coordinated_reheat_drying_enabled": bool(globals().get("DUAL_DRY_REHEAT_ENABLED", True)),
        "dry_lead_unit": (
            str(_dual_coordinated_dry_units()[0].get("name"))
            if _dual_coordinated_dry_units()[0] else None
        ),
        "dry_reheat_assist_unit": (
            str(_dual_coordinated_dry_units()[1].get("name"))
            if _dual_coordinated_dry_units()[1] else None
        ),
        "heat_landing_start_c": float(globals().get("DUAL_HEAT_LANDING_START_C", 0.40)),
        "heat_prediction_minutes": float(globals().get("DUAL_HEAT_PREDICTION_MINUTES", 10.0)),
        "heat_hard_stop_above_c": float(globals().get("DUAL_HEAT_HARD_STOP_ABOVE_C", 0.25)),
        "cop_modes": list(globals().get("DUAL_COP_MODES", ("heat",))),
        "cop_scoring_enabled": bool(globals().get("DUAL_COP_USE_FOR_SCORING", True)),
        "confirmed_outdoor_sensors": list(confirmed_outdoor_sensors),
        "watchdog_enabled": str(state.get("input_boolean.daikin_watchdog_enabled") or "off").lower() == "on" if _entity_exists("input_boolean.daikin_watchdog_enabled") else False,
        "shadow_mode": _controller_shadow_enabled(),
    }
    if publish:
        sensor = globals().get("DUAL_COMMISSIONING_SENSOR", "sensor.daikin_dual_commissioning")
        try:
            state.set(sensor, value=value, **result)
        except Exception as e:
            log.error("Daikin commissioning diagnostics publish failed: %s", e)
    return result


def _replay_auto_mode(sample, heat_delta, cool_delta, humidity_target, humidity_delta, dry_margin):
    try:
        tin = float(sample.get("indoor"))
        sp = float(sample.get("setpoint"))
    except Exception:
        return "invalid"
    if not isfinite(tin) or not isfinite(sp):
        return "invalid"
    humidity = sample.get("humidity")
    try:
        humidity = float(humidity)
    except Exception:
        humidity = float("nan")
    summer = bool(sample.get("summer"))
    if tin <= sp - float(heat_delta):
        return "heat"
    if summer and tin >= sp + float(cool_delta):
        return "cool"
    dry_temperature_ok = tin >= sp + float(dry_margin)
    if _dual_coordinated_dry_available():
        dry_temperature_ok = tin >= sp - max(
            0.05,
            float(globals().get("DUAL_DRY_LEAD_PAUSE_BELOW_SETPOINT_C", 0.30)),
        )
    if (
        summer and isfinite(humidity) and
        humidity >= float(humidity_target) + float(humidity_delta) and
        dry_temperature_ok
    ):
        return "dry"
    return "off"


def _run_history_replay(samples, heat_delta=None, cool_delta=None, humidity_target=None,
                        humidity_delta=None, dry_margin=None, confirmation_minutes=None,
                        source="internal_24h"):
    cache = _TickCache()
    heat_delta = _dual_float_helper(
        globals().get("DUAL_AUTO_HEAT_ON_DELTA_HELPER", ""),
        globals().get("DUAL_AUTO_HEAT_ON_DELTA_C", 0.40), 0.05, 5.0, cache,
    ) if heat_delta is None else float(heat_delta)
    cool_delta = _dual_float_helper(
        globals().get("DUAL_AUTO_COOL_ON_DELTA_HELPER", ""),
        globals().get("DUAL_AUTO_COOL_ON_DELTA_C", 0.40), 0.05, 5.0, cache,
    ) if cool_delta is None else float(cool_delta)
    limits = _dual_supervisor_limits(cache)
    humidity_target = float(limits.get("humidity_target")) if humidity_target is None else float(humidity_target)
    humidity_delta = float(limits.get("humidity_on_delta")) if humidity_delta is None else float(humidity_delta)
    dry_margin = float(limits.get("dry_temp_margin")) if dry_margin is None else float(dry_margin)
    confirmation_minutes = float(limits.get("auto_confirm_min")) if confirmation_minutes is None else float(confirmation_minutes)
    confirmation_s = max(0.0, confirmation_minutes * 60.0)
    counts = {"heat": 0, "cool": 0, "dry": 0, "off": 0, "invalid": 0}
    selected_counts = {"heat": 0, "cool": 0, "dry": 0, "off": 0}
    transitions = []
    selected = "off"
    candidate = "off"
    candidate_since = 0.0
    for idx, sample in enumerate(samples or []):
        if not isinstance(sample, dict):
            counts["invalid"] += 1
            continue
        desired = _replay_auto_mode(sample, heat_delta, cool_delta, humidity_target, humidity_delta, dry_margin)
        counts[desired] = int(counts.get(desired, 0)) + 1
        if desired == "invalid":
            continue
        try:
            ts = float(sample.get("ts"))
        except Exception:
            ts = float(idx) * 300.0
        if desired != candidate:
            candidate = desired
            candidate_since = ts
        emergency_heat = False
        try:
            emergency_heat = float(sample.get("indoor")) <= float(sample.get("setpoint")) - float(globals().get("DUAL_AUTO_EMERGENCY_HEAT_DELTA_C", 1.0))
        except Exception:
            emergency_heat = False
        confirmed = desired == "off" or emergency_heat or ts - float(candidate_since) >= confirmation_s
        if confirmed and selected != desired:
            transitions.append({"ts": ts, "from": selected, "to": desired})
            selected = desired
        selected_counts[selected] = int(selected_counts.get(selected, 0)) + 1
    result = {
        "source": str(source),
        "sample_count": len(samples or []),
        "valid_samples": sum([int(v) for k, v in counts.items() if k != "invalid"]),
        "candidate_counts": counts,
        "selected_counts": selected_counts,
        "transition_count": len(transitions),
        "transitions": transitions[-100:],
        "thresholds": {
            "heat_on_delta": heat_delta,
            "cool_on_delta": cool_delta,
            "humidity_target": humidity_target,
            "humidity_on_delta": humidity_delta,
            "dry_temperature_margin": dry_margin,
            "confirmation_minutes": confirmation_minutes,
        },
        "commands_sent": False,
        "completed_at": time.time(),
    }
    sensor = globals().get("DUAL_REPLAY_SENSOR", "sensor.daikin_dual_history_replay")
    value = "complete" if result.get("sample_count") else "no_samples"
    try:
        state.set(sensor, value=value, **result)
    except Exception as e:
        log.error("Daikin history replay publish failed: %s", e)
    return result


@service
def daikin_dual_validate_configuration():
    """Validate configured entities, modes, Demand options, helpers and units."""
    result = _validate_configuration_impl(publish=True)
    log.warning("Daikin commissioning validation: %s", str(result))


@service
def daikin_dual_calibrate_fan_rpm():
    """Profile every fixed advertised fan setting using physical fan_only.

    For safety and predictability this service runs only while the shared mode
    selector is `off + FAN` (internal value off_fan). Normal controller work is
    paused for the calibration and both units are restored to fan_only with
    lowMedium when it finishes or fails.
    """
    global _dual_fan_calibration_active, _dual_fan_calibration_status
    if bool(_dual_fan_calibration_active):
        return
    cache = _TickCache()
    if _requested_hvac_mode(cache) != "off_fan":
        _dual_fan_calibration_status = {
            "state": "blocked", "reason": "select_off_plus_fan_first",
            "unit": None, "fan_mode": None, "completed_at": time.time(),
        }
        _dual_publish_fan_calibration()
        log.warning("Daikin fan RPM calibration blocked: select off + FAN first")
        return

    settle_s = max(5.0, float(globals().get("DUAL_FAN_CALIBRATION_SETTLE_SECONDS", 45.0)))
    sample_s = max(10.0, float(globals().get("DUAL_FAN_CALIBRATION_SAMPLE_SECONDS", 30.0)))
    interval_s = max(1.0, float(globals().get("DUAL_FAN_CALIBRATION_SAMPLE_INTERVAL_SECONDS", 5.0)))
    _dual_fan_calibration_active = True
    _dual_fan_calibration_status = {
        "state": "starting", "reason": "calibration_requested",
        "unit": None, "fan_mode": None, "started_at": time.time(),
    }
    _dual_publish_fan_calibration()
    failure = None
    try:
        for u in DAIKINS:
            name = _unit_name(u)
            climate_ent = u.get("CLIMATE")
            fan_ent = u.get("FAN_RPM_SENSOR")
            if not climate_ent or not fan_ent:
                failure = "%s_missing_climate_or_fan_rpm_sensor" % name
                break
            modes = _dual_fan_modes(u, _TickCache())
            fixed_modes = []
            for option in modes:
                semantic = _dual_fan_mode_semantic(option)
                if semantic is not None and semantic != "auto":
                    fixed_modes.append(str(option))
            if not fixed_modes:
                failure = "%s_no_fixed_fan_modes_advertised" % name
                break
            climate.set_hvac_mode(entity_id=climate_ent, hvac_mode="fan_only")
            task.sleep(10.0)
            for option in fixed_modes:
                if _requested_hvac_mode(_TickCache()) != "off_fan":
                    failure = "shared_mode_changed_during_calibration"
                    break
                _dual_fan_calibration_status = {
                    "state": "calibrating", "reason": "waiting_for_stable_rpm",
                    "unit": name, "fan_mode": option,
                    "started_at": _dual_fan_calibration_status.get("started_at"),
                }
                _dual_publish_fan_calibration()
                climate.set_fan_mode(entity_id=climate_ent, fan_mode=option)
                task.sleep(settle_s)
                samples = []
                sample_started = time.time()
                while time.time() - sample_started < sample_s:
                    # Fan RPM is retained MQTT state and only publishes when
                    # it changes.  Its age must not invalidate calibration.
                    value = _TickCache().get_float(fan_ent, default=float("nan"))
                    if isfinite(value):
                        try:
                            rpm = float(value)
                            if isfinite(rpm) and rpm > 0.0:
                                samples.append(rpm)
                        except Exception:
                            pass
                    task.sleep(interval_s)
                record = _dual_record_fan_calibration(name, option, samples, option=option)
                if record is None:
                    failure = "%s_%s_no_valid_rpm_samples" % (name, option)
                    break
                _dual_publish_fan_calibration()
            if failure:
                break
    except Exception as e:
        failure = "calibration_exception:%s" % str(e)
    finally:
        # Return to the shared off + FAN contract regardless of partial failure.
        for u in DAIKINS:
            climate_ent = u.get("CLIMATE")
            if not climate_ent:
                continue
            try:
                climate.set_fan_mode(
                    entity_id=climate_ent,
                    fan_mode=_dual_resolve_fan_mode(
                        u, globals().get("DUAL_SHARED_FAN_MODE", "lowMedium"),
                        _TickCache(),
                    ) or globals().get("DUAL_SHARED_FAN_MODE", "lowMedium"),
                )
                climate.set_hvac_mode(entity_id=climate_ent, hvac_mode="fan_only")
            except Exception:
                pass
        _dual_fan_calibration_active = False
        _dual_fan_calibration_status = {
            "state": "failed" if failure else "complete",
            "reason": failure or "all_fixed_fan_modes_profiled",
            "unit": None, "fan_mode": None, "completed_at": time.time(),
        }
        _dual_save_store(force=True)
        _dual_publish_fan_calibration()


@service
def daikin_dual_apply_recommended_defaults():
    """Explicit one-time application of documented recommended helper values."""
    number_defaults = {
        "input_number.daikin_heat_on_delta": 0.15,
        "input_number.daikin_heat_off_delta": 0.10,
        "input_number.daikin_cool_on_delta": 0.20,
        "input_number.daikin_cool_off_delta": 0.10,
        "input_number.daikin_auto_heat_on_delta": 0.40,
        "input_number.daikin_auto_cool_on_delta": 0.40,
        "input_number.daikin_auto_mode_confirmation_minutes": 3.0,
        "input_number.daikin_coil_dry_minutes": 15.0,
        "input_number.daikin_minimum_on_minutes": 20.0,
        "input_number.daikin_minimum_off_minutes": 10.0,
        "input_number.daikin_humidity_target": 50.0,
        "input_number.daikin_humidity_on_delta": 3.0,
        "input_number.daikin_dry_temperature_margin": 0.20,
        "input_number.daikin_dual_total_step_limit": 12.0,
        "input_number.daikin_dual_episode_minutes": 60.0,
        "input_number.daikin_dual_abort_error": 0.60,
        "input_number.daikin_watchdog_timeout_minutes": 4.0,
    }
    applied = {}
    skipped = []
    for ent, value in number_defaults.items():
        if _entity_exists(ent):
            applied[ent] = bool(_set_input_number_if_needed(ent, value))
        else:
            skipped.append(ent)
    select_defaults = {
        globals().get("DUAL_HVAC_MODE_HELPER", "input_select.daikin_dual_hvac_mode"): "auto",
        globals().get("DUAL_OPTIMIZER_MODE_HELPER", "input_select.daikin_dual_optimizer_mode"): "shadow",
    }
    for ent, option in select_defaults.items():
        if _entity_exists(ent):
            try:
                input_select.select_option(entity_id=ent, option=option)
                applied[ent] = True
            except Exception:
                applied[ent] = False
        else:
            skipped.append(ent)
    for ent in (
        globals().get("DUAL_MANUAL_OVERRIDE_HELPER", "input_boolean.daikin_dual_optimizer_manual_override"),
        globals().get("DUAL_CONTROLLER_SHADOW_HELPER", "input_boolean.daikin_controller_shadow_mode"),
    ):
        if _entity_exists(ent):
            try:
                input_boolean.turn_off(entity_id=ent)
                applied[ent] = True
            except Exception:
                applied[ent] = False
        else:
            skipped.append(ent)
    result = _validate_configuration_impl(publish=True)
    sensor = globals().get("DUAL_COMMISSIONING_SENSOR", "sensor.daikin_dual_commissioning")
    try:
        attrs = state.getattr(sensor) or {}
        attrs["defaults_applied_at"] = time.time()
        attrs["defaults_applied"] = applied
        attrs["defaults_skipped"] = skipped
        state.set(sensor, value=result.get("result"), **attrs)
    except Exception:
        pass


@service
def daikin_dual_replay_history(samples_json="", heat_on_delta=None, cool_on_delta=None,
                               humidity_target=None, humidity_on_delta=None,
                               dry_temperature_margin=None, confirmation_minutes=None):
    """Replay internal 24h samples or a supplied JSON list; never sends commands."""
    samples = list(_replay_samples)
    source = "internal_24h"
    if samples_json:
        try:
            parsed = json.loads(str(samples_json))
            if isinstance(parsed, dict):
                parsed = parsed.get("samples") or []
            if isinstance(parsed, list):
                samples = parsed
                source = "samples_json"
        except Exception as e:
            log.error("Daikin history replay JSON invalid: %s", e)
            samples = []
            source = "invalid_samples_json"
    _run_history_replay(
        samples,
        heat_delta=heat_on_delta,
        cool_delta=cool_on_delta,
        humidity_target=humidity_target,
        humidity_delta=humidity_on_delta,
        dry_margin=dry_temperature_margin,
        confirmation_minutes=confirmation_minutes,
        source=source,
    )


@service
def daikin_ml_step():
    """Run every serialized controller cadence once for commissioning."""
    _housekeeping_last.clear()
    _daikin_dual_housekeeping_impl(time.time())


@service
def daikin_ml_reset():
    """Reset shared-zone learning without altering physical plant state."""
    global _dual_err_int, _dual_last_ctrl_ts, _dual_last_sp
    global _dual_last_control_mode, _dual_last_total_target
    global _dual_sustain_by_ctx, _dual_policy_stats, _dual_cop_stats
    global _dual_current_cop_snapshot, _dual_active_action_id, _dual_episode
    global _dual_last_store_save_ts, _dual_last_store_payload
    global _dual_optimizer_last_sample_ts, _dual_optimizer_prev_power_w
    global _dual_optimizer_prev_cop_power_w, _dual_optimizer_prev_thermal_w
    global _dual_optimizer_prev_heating, _dual_action_changed_at
    global _dual_last_fast_taper_ts, _dual_landing_status
    global _dual_sustain_probe, _dual_cdp_release, _dual_cdp_learned_demand
    global _dual_cdp_temperature_curve
    global _separate_area_state, _separate_area_learning

    _runtime_store_save(force=True)
    _dual_tin_hist[:] = []
    _dual_fast_tin_hist[:] = []
    _dual_humidity_hist[:] = []
    _dual_learning_humidity_hist[:] = []
    _dual_err_int = 0.0
    _dual_last_ctrl_ts = 0.0
    _dual_last_sp = float("nan")
    _dual_last_control_mode = None
    _dual_last_total_target = float("nan")
    _dual_sustain_by_ctx = {}
    _dual_policy_stats = {}
    _dual_cop_stats = {}
    _dual_cdp_release = {}
    _dual_cdp_learned_demand = {}
    _dual_cdp_temperature_curve = {}
    _separate_area_state = {}
    _separate_area_learning = {}
    _dual_current_cop_snapshot = {
        "valid": False,
        "reason": "reset",
        "combined_cop": None,
        "context": None,
    }
    _dual_active_action_id = None
    _dual_episode = None
    _dual_last_store_save_ts = 0.0
    _dual_last_store_payload = None
    _dual_optimizer_last_sample_ts = 0.0
    _dual_optimizer_prev_power_w = None
    _dual_optimizer_prev_cop_power_w = None
    _dual_optimizer_prev_thermal_w = None
    _dual_optimizer_prev_heating = {}
    _dual_action_changed_at = 0.0
    _dual_last_fast_taper_ts = 0.0
    _dual_landing_status = {
        "active": False,
        "reason": "reset",
        "projected_error": None,
        "projected_temperature": None,
        "eta_to_setpoint_minutes": None,
        "landing_cap": None,
        "step_limited_total": None,
        "rapid_approach": False,
        "control_interval_seconds": None,
        "fast_rate_cph": None,
        "long_rate_cph": None,
        "rate_source": "none",
    }
    _dual_sustain_probe = {
        "state": "idle",
        "context": None,
        "candidate": None,
        "baseline": None,
        "started_at": 0.0,
        "settled_at": 0.0,
        "cooldown_until": 0.0,
        "reason": "reset",
    }
    _dual_reset_learning_window("reset", discard_episode=False)
    _dual_save_store(force=True)
    _dual_publish_learning_status(_dual_active_mode)
    log.warning("Daikin shared-zone learning reset; plant state preserved")

@service
def daikin_dual_optimizer_reset():
    """Reset only the learned dual-unit efficiency policy and active episode."""
    global _dual_policy_stats, _dual_cop_stats, _dual_current_cop_snapshot
    global _dual_episode, _dual_active_action_id
    global _dual_last_store_save_ts, _dual_last_store_payload
    _dual_policy_stats = {}
    _dual_cop_stats = {}
    _dual_current_cop_snapshot = {
        "valid": False, "reason": "optimizer_reset",
        "combined_cop": None, "context": None,
    }
    _dual_episode = None
    _dual_active_action_id = None
    _dual_last_store_save_ts = 0.0
    _dual_last_store_payload = None
    _dual_reset_learning_window("optimizer_reset", discard_episode=False)
    _dual_save_store(force=True)
    _dual_publish_learning_status(_dual_active_mode)
    log.warning("Daikin dual optimizer: learned policy reset")


@service
def daikin_dual_cop_reset():
    """Reset only the matched COP-by-load map; retain sustain and policy scores."""
    global _dual_cop_stats, _dual_current_cop_snapshot
    global _dual_last_store_save_ts, _dual_last_store_payload
    _dual_cop_stats = {}
    _dual_current_cop_snapshot = {
        "valid": False, "reason": "cop_map_reset",
        "combined_cop": None, "context": None,
    }
    _dual_last_store_save_ts = 0.0
    _dual_last_store_payload = None
    _dual_save_store(force=True)
    _dual_publish_cop(
        _dual_current_cop_snapshot,
        learning_allowed=False,
        learning_reason="cop_map_reset",
    )
    log.warning("Daikin dual optimizer: matched COP map reset")


# ============================================================
# 8) STORE PERSIST (optional helper)
# ============================================================
@service
def daikin_ml_persist():
    """Persist only the shared optimizer and restart-safe runtime stores."""
    try:
        state.persist(
            globals().get(
                "DUAL_ZONE_STORE_ENTITY", "pyscript.daikin_dual_zone_store"
            )
        )
        state.persist(
            globals().get(
                "DUAL_RUNTIME_STORE_ENTITY",
                "pyscript.daikin_dual_runtime_store",
            )
        )
    except Exception:
        pass
    _dual_save_store(force=True)
    _runtime_store_save(force=True)
