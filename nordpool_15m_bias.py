# pyscript/nordpool_15m_bias.py
#
# Nord Pool 15-min day-ahead price bias calculator (standalone layer).
#
# Supports multiple sensor attribute structures:
#
# A) Your original:
#   state = current price
#   attributes.records:
#     - Time: '2025-12-22T00:00:00'
#       End:  '2025-12-22T00:15:00'
#       Price: 5.45
#
# B) New structure (your request):
#   attributes.raw_today:
#     - start: '2025-12-23T00:00:00+02:00'
#       end:   '2025-12-23T00:15:00+02:00'
#       value: 7.64
#
# Average window (hours) via input_number helper:
#   input_number.nordpool_avg_window_hours
#
# Writes debug sensors:
#   sensor.nordpool_15m_avg_price
#   sensor.nordpool_15m_rel_price
#   sensor.nordpool_15m_bias_points
#   sensor.nordpool_15m_now_price
#   sensor.nordpool_15m_record_count
#   sensor.nordpool_15m_avg_window_hours
#   sensor.nordpool_15m_source_format
#
# Writes per-unit bias helpers:
#   input_number.daikin1_price_bias_points
#
# Notes:
# - Negative bias => reduce demand when price is high.
# - Positive bias => increase demand when price is low.
# - This script DOES NOT change your Daikin demand directly; it only writes helpers/sensors.

from math import isfinite
import time
from datetime import datetime, timezone

# ----------------------------
# CONFIG
# ----------------------------
PRICE_SENSOR = "sensor.day_ahead_price"

# Optional global enable switch (create in HA if you want)
BIAS_ENABLED_SWITCH = "input_boolean.nordpool_bias_enabled"  # set to None to disable switch usage

# Average window helper (hours)
AVG_WINDOW_HOURS_HELPER = "input_number.nordpool_avg_window_hours"
DEFAULT_AVG_WINDOW_HOURS = 24.0
MIN_AVG_WINDOW_HOURS = 0.25   # 15 minutes
MAX_AVG_WINDOW_HOURS = 48.0   # up to 2 days

# Per-unit bias targets (create these input_numbers in HA)
UNIT_BIAS_HELPERS = {
    "daikin1": "input_number.daikin1_price_bias_points",
    # "daikin2": "input_number.daikin2_price_bias_points",
}

# Mapping/tuning
MAX_BIAS_POINTS = 20.0      # maximum +/- demand points
REL_AT_MAX = 0.50           # rel=+0.50 (50% above avg) => full negative bias
REL_DEADBAND = 0.05         # +-5% around average => no bias
EMA_ALPHA = 0.30            # smoothing for rel: new = alpha*now + (1-alpha)*prev. Set 1.0 to disable.
MIN_AVG_PRICE = 0.0001      # safety

# Update cadence: run each minute + on sensor change
CRON_EXPR = "cron(* * * * *)"

# Debug sensors
S_AVG = "sensor.nordpool_15m_avg_price"
S_REL = "sensor.nordpool_15m_rel_price"
S_BIAS = "sensor.nordpool_15m_bias_points"
S_NOW = "sensor.nordpool_15m_now_price"
S_REC_COUNT = "sensor.nordpool_15m_record_count"
S_WIN_H = "sensor.nordpool_15m_avg_window_hours"
S_FMT = "sensor.nordpool_15m_source_format"

# Store for EMA
STATE_STORE = "pyscript.nordpool_15m_bias_store"


# ----------------------------
# Helpers
# ----------------------------
def _clip(v, lo, hi):
    try:
        vf = float(v)
        if not isfinite(vf):
            return lo
    except Exception:
        return lo
    if vf < lo:
        return lo
    if vf > hi:
        return hi
    return vf

def _read_bool(entity_id, default=True):
    if not entity_id:
        return default
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

def _safe_float(x, default=None):
    try:
        v = float(x)
        if isfinite(v):
            return v
        return default
    except Exception:
        return default

def _read_avg_window_hours():
    v = _safe_float(state.get(AVG_WINDOW_HOURS_HELPER), default=None)
    if v is None:
        v = DEFAULT_AVG_WINDOW_HOURS
    v = _clip(v, MIN_AVG_WINDOW_HOURS, MAX_AVG_WINDOW_HOURS)
    return float(v)

def _parse_iso_dt(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s))
        return dt
    except Exception:
        return None

def _normalize_dt_pair(record_dt, now_dt):
    # Normalize tz mismatch (naive vs aware) by making both naive
    if record_dt is None:
        return None, now_dt
    if (record_dt.tzinfo is None) != (now_dt.tzinfo is None):
        record_dt = record_dt.replace(tzinfo=None)
        now_dt = now_dt.replace(tzinfo=None)
    return record_dt, now_dt

def _in_window(record_time_dt, now_dt, window_hours):
    if record_time_dt is None:
        return False
    record_time_dt, now_dt = _normalize_dt_pair(record_time_dt, now_dt)
    delta = record_time_dt - now_dt
    return (delta.total_seconds() >= 0) and (delta.total_seconds() <= window_hours * 3600.0)

def _detect_source_format(attrs):
    # Prefer records if it exists and is list; otherwise raw_today if it exists and is list.
    rec = attrs.get("records")
    if isinstance(rec, list) and len(rec) > 0:
        return "records"
    rt = attrs.get("raw_today")
    if isinstance(rt, list) and len(rt) > 0:
        return "raw_today"
    # Try raw_tomorrow too (some sensors split)
    rtom = attrs.get("raw_tomorrow")
    if isinstance(rtom, list) and len(rtom) > 0:
        return "raw_tomorrow"
    # Fallback: if any of them exist but empty
    if isinstance(rec, list):
        return "records_empty"
    if isinstance(rt, list):
        return "raw_today_empty"
    if isinstance(rtom, list):
        return "raw_tomorrow_empty"
    return "unknown"

def _extract_prices_from_records_window(records, now_dt, window_hours):
    prices = []
    count_total = 0
    count_in_window = 0

    if not isinstance(records, list):
        return prices, count_total, count_in_window

    for r in records:
        if not isinstance(r, dict):
            continue
        count_total += 1
        tdt = _parse_iso_dt(r.get("Time"))
        if not _in_window(tdt, now_dt, window_hours):
            continue
        p = _safe_float(r.get("Price"), default=None)
        if p is not None:
            prices.append(p)
            count_in_window += 1

    return prices, count_total, count_in_window

def _extract_prices_from_raw_window(raw_list, now_dt, window_hours, time_key, value_key):
    prices = []
    count_total = 0
    count_in_window = 0

    if not isinstance(raw_list, list):
        return prices, count_total, count_in_window

    for r in raw_list:
        if not isinstance(r, dict):
            continue
        count_total += 1
        tdt = _parse_iso_dt(r.get(time_key))
        if not _in_window(tdt, now_dt, window_hours):
            continue
        p = _safe_float(r.get(value_key), default=None)
        if p is not None:
            prices.append(p)
            count_in_window += 1

    return prices, count_total, count_in_window

def _calc_avg(prices):
    if not prices:
        return None
    s = 0.0
    for p in prices:
        s += p
    return s / float(len(prices))

def _get_prev_rel():
    attrs = state.getattr(STATE_STORE) or {}
    prev = _safe_float(attrs.get("rel_ema"), default=None)
    return prev

def _set_prev_rel(rel_ema):
    try:
        state.set(STATE_STORE, value=time.time(), rel_ema=float(rel_ema))
    except Exception as e:
        log.error("Nordpool bias: failed to persist EMA store: %s", e)

def _compute_bias(price_now, avg_price):
    avg = max(float(avg_price), MIN_AVG_PRICE)
    rel_now = (float(price_now) - avg) / avg

    if EMA_ALPHA >= 1.0:
        rel_ema = rel_now
    else:
        prev = _get_prev_rel()
        if prev is None:
            rel_ema = rel_now
        else:
            rel_ema = (EMA_ALPHA * rel_now) + ((1.0 - EMA_ALPHA) * prev)

    _set_prev_rel(rel_ema)

    if abs(rel_ema) < REL_DEADBAND:
        bias = 0.0
    else:
        bias = -(rel_ema / REL_AT_MAX) * MAX_BIAS_POINTS
        bias = _clip(bias, -MAX_BIAS_POINTS, +MAX_BIAS_POINTS)

    return rel_now, rel_ema, bias

def _set_input_number(entity_id, value):
    if not entity_id:
        return
    try:
        input_number.set_value(entity_id=entity_id, value=float(value))
    except Exception as e:
        log.debug("Nordpool bias: could not set %s (create it in HA?). err=%s", entity_id, e)

def _update_debug_sensors(price_now, avg_price, rel_now, rel_ema, bias, total_count, in_window_count, window_hours, fmt):
    try:
        state.set(S_NOW, value=round(float(price_now), 6))
        state.set(S_AVG, value=round(float(avg_price), 6),
                  window_hours=round(float(window_hours), 2),
                  in_window=int(in_window_count),
                  total=int(total_count))
        state.set(S_REL, value=round(float(rel_ema), 6), rel_now=round(float(rel_now), 6))
        state.set(S_BIAS, value=round(float(bias), 3),
                  max_bias=MAX_BIAS_POINTS,
                  rel_at_max=REL_AT_MAX,
                  deadband=REL_DEADBAND)
        state.set(S_REC_COUNT, value=int(in_window_count), total=int(total_count))
        state.set(S_WIN_H, value=round(float(window_hours), 2), helper=AVG_WINDOW_HOURS_HELPER)
        state.set(S_FMT, value=str(fmt))
    except Exception as e:
        log.error("Nordpool bias: failed to update debug sensors: %s", e)


# ----------------------------
# Main update function
# ----------------------------
def _update_nordpool_bias():
    # enable switch (optional)
    if BIAS_ENABLED_SWITCH:
        enabled = _read_bool(BIAS_ENABLED_SWITCH, default=True)
        if not enabled:
            for unit, helper in UNIT_BIAS_HELPERS.items():
                _set_input_number(helper, 0.0)
            try:
                state.set(S_BIAS, value=0.0, note="disabled")
            except Exception:
                pass
            return

    attrs = state.getattr(PRICE_SENSOR) or {}

    # current price: prefer sensor state, fallback to first record in the closest-to-now slot (if state missing)
    price_now = _safe_float(state.get(PRICE_SENSOR), default=None)

    window_hours = _read_avg_window_hours()
    now_dt = datetime.now(timezone.utc).astimezone()  # local tz aware

    fmt = _detect_source_format(attrs)

    total_count = 0
    in_window_count = 0
    prices = []

    if fmt.startswith("records"):
        rec = attrs.get("records") or []
        prices, total_count, in_window_count = _extract_prices_from_records_window(rec, now_dt, window_hours)

        # fallback current price from first in-window record if state missing
        if price_now is None and prices:
            price_now = prices[0]

    elif fmt.startswith("raw_today"):
        rt = attrs.get("raw_today") or []
        prices, total_count, in_window_count = _extract_prices_from_raw_window(rt, now_dt, window_hours, "start", "value")

        if price_now is None and prices:
            price_now = prices[0]

    elif fmt.startswith("raw_tomorrow"):
        rtom = attrs.get("raw_tomorrow") or []
        prices, total_count, in_window_count = _extract_prices_from_raw_window(rtom, now_dt, window_hours, "start", "value")

        if price_now is None and prices:
            price_now = prices[0]

    else:
        # try both if unknown
        rec = attrs.get("records")
        if isinstance(rec, list):
            prices, total_count, in_window_count = _extract_prices_from_records_window(rec, now_dt, window_hours)
            fmt = "records_fallback"
        else:
            rt = attrs.get("raw_today")
            if isinstance(rt, list):
                prices, total_count, in_window_count = _extract_prices_from_raw_window(rt, now_dt, window_hours, "start", "value")
                fmt = "raw_today_fallback"

        if price_now is None and prices:
            price_now = prices[0]

    if price_now is None:
        log.warning("Nordpool bias: %s state not a finite number and no usable record fallback. state=%s fmt=%s",
                    PRICE_SENSOR, str(state.get(PRICE_SENSOR)), fmt)
        return

    avg_price = _calc_avg(prices)

    # Fallback if no prices in window: avg = now -> neutral
    if avg_price is None or (not isfinite(avg_price)):
        avg_price = float(price_now)
        in_window_count = 0

    rel_now, rel_ema, bias = _compute_bias(price_now, avg_price)

    for unit, helper in UNIT_BIAS_HELPERS.items():
        _set_input_number(helper, bias)

    _update_debug_sensors(price_now, avg_price, rel_now, rel_ema, bias, total_count, in_window_count, window_hours, fmt)

    log.info(
        "Nordpool bias: fmt=%s win=%.2fh now=%.4f avg=%.4f rel_now=%.3f rel_ema=%.3f -> bias=%.2f pts (in_window=%d total=%d)",
        str(fmt), float(window_hours), float(price_now), float(avg_price),
        float(rel_now), float(rel_ema), float(bias),
        int(in_window_count), int(total_count)
    )


# ----------------------------
# Triggers
# ----------------------------
@time_trigger("startup")
def nordpool_bias_startup():
    try:
        state.persist(STATE_STORE)
    except Exception:
        pass
    _update_nordpool_bias()

@time_trigger(CRON_EXPR)
@state_trigger(PRICE_SENSOR, AVG_WINDOW_HOURS_HELPER)
def nordpool_bias_update(**kwargs):
    _update_nordpool_bias()


# ----------------------------
# Services
# ----------------------------
@service
def nordpool_bias_step():
    """Manually recompute Nordpool bias now."""
    _update_nordpool_bias()

@service
def nordpool_bias_reset_ema():
    """Reset EMA smoothing memory."""
    _set_prev_rel(0.0)
    log.info("Nordpool bias: EMA reset to 0.0")
