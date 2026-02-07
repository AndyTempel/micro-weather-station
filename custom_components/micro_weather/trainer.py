# bandit: skipfile
"""ML Model Trainer for Micro Weather Station using pure Python implementation."""

import datetime
import logging
import random
from typing import Any, Dict, List, Optional, Tuple, cast

from homeassistant.core import HomeAssistant, State
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


class DecisionTree:
    """A pure Python Decision Tree Classifier using Gini Impurity."""

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X: List[List[float]], y: List[int]):
        """Build the decision tree."""
        self.tree = self._build_tree(X, y, depth=0)

    def _gini(self, y: List[int]) -> float:
        """Calculate Gini Impurity."""
        m = len(y)
        if m == 0:
            return 0
        p_1 = sum(y) / m
        return 1.0 - p_1**2 - (1.0 - p_1) ** 2

    def _best_split(
        self, X: List[List[float]], y: List[int]
    ) -> Tuple[Optional[int], Optional[float]]:
        """Find the best feature and threshold to split on."""
        best_gini = 1.0
        best_idx, best_thr = None, None
        n_features = len(X[0])

        for idx in range(n_features):
            # Sort feature values to find best split points
            feature_values = sorted(set(row[idx] for row in X))
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2

                y_left = [y[j] for j, row in enumerate(X) if row[idx] <= threshold]
                y_right = [y[j] for j, row in enumerate(X) if row[idx] > threshold]

                if not y_left or not y_right:
                    continue

                gini = (
                    len(y_left) * self._gini(y_left)
                    + len(y_right) * self._gini(y_right)
                ) / len(y)

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = threshold

        return best_idx, best_thr

    def _build_tree(self, X: List[List[float]], y: List[int], depth: int) -> Any:
        """Recursive function to build the tree structure."""
        num_samples = len(y)
        num_class_1 = sum(y)

        # Base cases: pure node, max depth, or too few samples
        if num_class_1 == 0:
            return 0
        if num_class_1 == num_samples:
            return 1
        if depth >= self.max_depth or num_samples < self.min_samples_split:
            return 1 if num_class_1 / num_samples >= 0.5 else 0

        idx, thr = self._best_split(X, y)
        if idx is None or thr is None:
            return 1 if num_class_1 / num_samples >= 0.5 else 0

        indices_left = [i for i, row in enumerate(X) if row[idx] <= thr]
        indices_right = [i for i, row in enumerate(X) if row[idx] > thr]

        left = self._build_tree(
            [X[i] for i in indices_left], [y[i] for i in indices_left], depth + 1
        )
        right = self._build_tree(
            [X[i] for i in indices_right], [y[i] for i in indices_right], depth + 1
        )

        return (idx, thr, left, right)

    def predict_row(self, row: List[float]) -> int:
        """Predict class for a single row."""
        node = self.tree
        if node is None:
            return 0
        while isinstance(node, tuple):
            idx, thr, left, right = node
            node = left if row[idx] <= thr else right
        return cast(int, node)


class RandomForest:
    """A pure Python Random Forest Classifier."""

    def __init__(
        self, n_estimators: int = 10, max_depth: int = 10, min_samples_split: int = 5
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees: List[DecisionTree] = []

    def fit(self, X: List[List[float]], y: List[int]):
        """Train the forest using bootstrap samples."""
        self.trees = []
        n_samples = len(X)

        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = [
                random.randint(0, n_samples - 1) for _ in range(n_samples)  # nosec B311
            ]
            X_sample = [X[i] for i in indices]
            y_sample = [y[i] for i in indices]

            tree = DecisionTree(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict by averaging tree outputs."""
        predictions = []
        for row in X:
            votes = [tree.predict_row(row) for tree in self.trees]
            # Majority vote
            predictions.append(1 if sum(votes) / len(votes) >= 0.5 else 0)
        return predictions

    def score(self, X: List[List[float]], y: List[int]) -> float:
        """Calculate accuracy."""
        preds = self.predict(X)
        correct = sum(1 for p, gt in zip(preds, y) if p == gt)
        return correct / len(y)


async def train_model(
    hass: HomeAssistant, sensor_map: Dict[str, str], model_path: str
) -> Dict[str, Any]:
    """Train the weather model using historical data.

    This function runs in the event loop but offloads DB and CPU work.
    """
    from homeassistant.components.recorder import history
    from homeassistant.helpers.recorder import get_instance

    _LOGGER.info("Starting Pure-Python ML model training for Micro Weather Station")

    # 1. Gather historical data
    end_time = dt_util.utcnow()
    start_time = end_time - datetime.timedelta(days=30)

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

    # Use the recorder's executor for database access as recommended by HA
    # Correct positional arguments to avoid NotImplementedError (filters)
    # 1: hass, 2: start, 3: end, 4: entity_ids, 5: filters (None),
    # 6: include_start_time_state (True), 7: significant_changes_only (False)
    hist_data = await get_instance(hass).async_add_executor_job(
        history.get_significant_states,
        hass,
        start_time,
        end_time,
        entity_ids,
        None,
        True,
        False,
    )

    def process_data():
        """Process data and train model in executor."""
        # Move heavy imports here to avoid blocking the event loop
        import joblib
        import pandas as pd

        dfs = []
        for key, entity_id in key_to_entity.items():
            states = hist_data.get(entity_id)
            if not states:
                _LOGGER.warning("No history found for %s", entity_id)
                continue

            data = []
            for state in states:
                if not isinstance(state, State):
                    continue
                try:
                    val = float(state.state)
                    data.append({"timestamp": state.last_updated, key: val})
                except (ValueError, TypeError):
                    continue

            if data:
                df_s = pd.DataFrame(data)
                df_s["timestamp"] = pd.to_datetime(df_s["timestamp"], utc=True)
                df_s.set_index("timestamp", inplace=True)
                df_s = df_s.resample("15min").mean()
                dfs.append(df_s)

        if len(dfs) < len(required_keys):
            return {"success": False, "error": "Insufficient data sources"}

        # 2. Merge & Feature Engineering
        df = pd.concat(dfs, axis=1)
        df = df.interpolate(method="time", limit=2).dropna()

        if df.empty:
            return {"success": False, "error": "No overlapping data"}

        # Trends
        df["pressure_trend_1h"] = df[CONF_PRESSURE_SENSOR].diff(4)
        df["pressure_delta_3h"] = df[CONF_PRESSURE_SENSOR].diff(12)
        df["humidity_trend_1h"] = df[CONF_HUMIDITY_SENSOR].diff(4)
        df["solar_drop_1h"] = df[CONF_SOLAR_RADIATION_SENSOR].diff(4)

        # Target: Will it rain in the next 45 minutes?
        df["target"] = (df[CONF_RAIN_RATE_SENSOR].shift(-3) > 0).astype(int)
        df = df.dropna()

        if df.empty:
            return {"success": False, "error": "Insufficient data points"}

        # 3. Data Balancing
        mask_raining_now = df[CONF_RAIN_RATE_SENSOR] > 0
        mask_will_rain = df["target"] == 1

        df_continuation = df[mask_raining_now]
        df_onset = df[~mask_raining_now & mask_will_rain]
        df_stable = df[~mask_raining_now & ~mask_will_rain]

        n_rain_events = len(df_onset) + len(df_continuation)
        n_keep_stable = max(n_rain_events * 3, 50)

        df_stable_balanced = df_stable
        if len(df_stable) > n_keep_stable:
            df_stable_balanced = df_stable.sample(n=n_keep_stable, random_state=42)

        df_balanced = pd.concat([df_onset, df_continuation, df_stable_balanced]).sample(
            frac=1, random_state=42
        )

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

        X = df_balanced[feature_cols].values.tolist()
        y = df_balanced["target"].values.tolist()

        if len(X) < 20:
            return {"success": False, "error": "Insufficient samples"}

        try:
            # Use smaller forest for performance in pure Python
            clf = RandomForest(n_estimators=10, max_depth=8)
            clf.fit(X, y)

            # Save model using joblib (it handles custom objects well)
            joblib.dump(clf, model_path)

            accuracy = clf.score(X, y)

            _LOGGER.info(
                "Pure-Python model training successful. Accuracy: %.2f", accuracy
            )

            return {
                "success": True,
                "accuracy": accuracy,
                "last_trained": datetime.datetime.now().isoformat(),
            }

        except Exception as err:
            _LOGGER.exception("Training failed: %s", err)
            return {"success": False, "error": str(err)}

    return await hass.async_add_executor_job(process_data)
