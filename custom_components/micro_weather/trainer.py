"""ML Model Trainer for Micro Weather Station."""

import datetime
import logging
import os
from typing import Any, Dict

from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    device_registry as dr,
    entity_registry as er,
)
from homeassistant.util import dt as dt_util

from .const import (
    CONF_HUMIDITY_SENSOR,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_PRESSURE_SENSOR,
    CONF_RAIN_RATE_SENSOR,
    CONF_SOLAR_RADIATION_SENSOR,
    CONF_WIND_SPEED_SENSOR,
)

_LOGGER = logging.getLogger(__name__)


def train_model(hass: HomeAssistant, sensor_map: Dict[str, str], model_path: str) -> Dict[str, Any]:
    """Train the weather model using historical data.

    This function runs in the executor thread.
    """
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from homeassistant.components.recorder import history

    _LOGGER.info("Starting ML model training for Micro Weather Station")

    # 1. Gather historical data
    end_time = dt_util.utcnow()
    start_time = end_time - datetime.timedelta(days=30)

    # We need specific sensors for the 9 features requested:
    # 1. Temp, 2. Humidity, 3. Pressure, 4. Solar, 5. Wind
    # And we need Rain for the target.
    
    required_keys = [
        CONF_OUTDOOR_TEMP_SENSOR,
        CONF_HUMIDITY_SENSOR,
        CONF_PRESSURE_SENSOR,
        CONF_SOLAR_RADIATION_SENSOR,
        CONF_WIND_SPEED_SENSOR,
        CONF_RAIN_RATE_SENSOR,
    ]

    entity_ids = []
    key_to_entity = {}
    for key in required_keys:
        entity_id = sensor_map.get(key)
        if not entity_id:
            _LOGGER.error("Missing required sensor for training: %s", key)
            return {"success": False, "error": f"Missing {key}"}
        entity_ids.append(entity_id)
        key_to_entity[key] = entity_id

    _LOGGER.debug("Fetching history for entities: %s", entity_ids)
    
    # get_significant_states returns a dict {entity_id: [states]}
    hist_data = history.get_significant_states(
        hass, start_time, end_time, entity_ids=entity_ids, significant_changes_only=False
    )

    dfs = []
    for key, entity_id in key_to_entity.items():
        states = hist_data.get(entity_id)
        if not states:
            _LOGGER.warning("No history found for %s", entity_id)
            continue
        
        data = []
        for state in states:
            try:
                val = float(state.state)
                data.append({"timestamp": state.last_updated, key: val})
            except (ValueError, TypeError):
                continue
        
        if data:
            df_s = pd.DataFrame(data)
            df_s["timestamp"] = pd.to_datetime(df_s["timestamp"], utc=True)
            df_s.set_index("timestamp", inplace=True)
            # Resample to 15min intervals to align data
            df_s = df_s.resample("15min").mean()
            dfs.append(df_s)

    if len(dfs) < len(required_keys):
        _LOGGER.error("Insufficient data sources for training. Found %d/%d", len(dfs), len(required_keys))
        return {"success": False, "error": "Insufficient data sources"}

    # 2. Merge & Feature Engineering
    df = pd.concat(dfs, axis=1)
    # Interpolate missing values (up to 30 mins)
    df = df.interpolate(method="time", limit=2).dropna()

    if df.empty:
        _LOGGER.error("No overlapping data found for sensors")
        return {"success": False, "error": "No overlapping data"}

    # Feature Engineering (Matching the 9 features requested)
    # Order: Temp, Hum, Pres, Solar, Wind, PresTrend1h, PresDelta3h, HumTrend1h, SolarDrop1h
    
    # Trends (resampled to 15min, so 1h = 4 steps, 3h = 12 steps)
    df["pressure_trend_1h"] = df[CONF_PRESSURE_SENSOR].diff(4)
    df["pressure_delta_3h"] = df[CONF_PRESSURE_SENSOR].diff(12)
    df["humidity_trend_1h"] = df[CONF_HUMIDITY_SENSOR].diff(4)
    df["solar_drop_1h"] = df[CONF_SOLAR_RADIATION_SENSOR].diff(4)

    # Target: Will it rain in the next 45 minutes? (3 * 15min steps)
    df["target"] = (df[CONF_RAIN_RATE_SENSOR].shift(-3) > 0).astype(int)
    
    df = df.dropna()
    
    if df.empty:
        _LOGGER.error("Not enough data points after feature engineering")
        return {"success": False, "error": "Insufficient data points"}

    # 3. Data Balancing
    # Logic from WeatherModel.py: Undersample stable weather
    mask_raining_now = df[CONF_RAIN_RATE_SENSOR] > 0
    mask_will_rain = df["target"] == 1

    df_continuation = df[mask_raining_now]
    df_onset = df[~mask_raining_now & mask_will_rain]
    df_stable = df[~mask_raining_now & ~mask_will_rain]

    _LOGGER.info(
        "Distribution -> Stable: %d, Continuation: %d, Onset: %d",
        len(df_stable), len(df_continuation), len(df_onset)
    )

    n_rain_events = len(df_onset) + len(df_continuation)
    n_keep_stable = max(n_rain_events * 3, 50)

    if len(df_stable) > n_keep_stable:
        df_stable = df_stable.sample(n=n_keep_stable, random_state=42)

    df_balanced = pd.concat([df_onset, df_continuation, df_stable]).sample(frac=1, random_state=42)

    # 4. Training
    feature_cols = [
        CONF_OUTDOOR_TEMP_SENSOR,
        CONF_HUMIDITY_SENSOR,
        CONF_PRESSURE_SENSOR,
        CONF_SOLAR_RADIATION_SENSOR,
        CONF_WIND_SPEED_SENSOR,
        "pressure_trend_1h",
        "pressure_delta_3h",
        "humidity_trend_1h",
        "solar_drop_1h",
    ]

    X = df_balanced[feature_cols]
    y = df_balanced["target"]

    if len(X) < 20:
        _LOGGER.error("Too few samples for training: %d", len(X))
        return {"success": False, "error": "Insufficient samples"}

    try:
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=4, n_jobs=1
        )
        clf.fit(X, y)

        # Save model
        joblib.dump(clf, model_path)
        
        accuracy = clf.score(X, y)
        importances = dict(zip(feature_cols, clf.feature_importances_.round(2)))
        
        _LOGGER.info("Model training successful. Accuracy: %.2f", accuracy)
        _LOGGER.debug("Feature importances: %s", importances)

        return {
            "success": True,
            "accuracy": accuracy,
            "last_trained": datetime.datetime.now().isoformat(),
            "feature_importances": importances
        }

    except Exception as err:
        _LOGGER.exception("Training failed: %s", err)
        return {"success": False, "error": str(err)}
