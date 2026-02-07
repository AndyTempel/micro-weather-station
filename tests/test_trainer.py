"""Tests for Micro Weather Station ML trainer."""

import datetime
from unittest.mock import MagicMock, patch, AsyncMock

from homeassistant.core import HomeAssistant, State
import pytest

from custom_components.micro_weather.const import (
    CONF_HUMIDITY_SENSOR,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_PRESSURE_SENSOR,
    CONF_RAIN_RATE_SENSOR,
    CONF_SOLAR_RADIATION_SENSOR,
    CONF_WIND_SPEED_SENSOR,
)
from custom_components.micro_weather.trainer import train_model


@pytest.fixture
def mock_history_states():
    """Mock historical states for sensors."""
    now = datetime.datetime.now(datetime.timezone.utc)

    def create_states(entity_id, values):
        states = []
        for i, val in enumerate(values):
            last_updated = now - datetime.timedelta(
                minutes=15 * (len(values) - i)
            )
            state = State(entity_id, str(val), last_updated=last_updated)
            states.append(state)
        return states

    return {
        "sensor.temp": create_states("sensor.temp", [20.0 + i for i in range(50)]),
        "sensor.hum": create_states("sensor.hum", [50.0 + i for i in range(50)]),
        "sensor.pres": create_states(
            "sensor.pres", [1013.0 + i * 0.1 for i in range(50)]
        ),
        "sensor.solar": create_states("sensor.solar", [200.0 + i for i in range(50)]),
        "sensor.wind": create_states("sensor.wind", [5.0 + i * 0.1 for i in range(50)]),
        "sensor.rain": create_states("sensor.rain", [0.0] * 45 + [1.0] * 5),
    }


@pytest.fixture
def sensor_map():
    """Mock sensor map."""
    return {
        CONF_OUTDOOR_TEMP_SENSOR: "sensor.temp",
        CONF_HUMIDITY_SENSOR: "sensor.hum",
        CONF_PRESSURE_SENSOR: "sensor.pres",
        CONF_SOLAR_RADIATION_SENSOR: "sensor.solar",
        CONF_WIND_SPEED_SENSOR: "sensor.wind",
        CONF_RAIN_RATE_SENSOR: "sensor.rain",
    }


async def test_train_model_success(
    hass: HomeAssistant, sensor_map, mock_history_states
):
    """Test successful model training."""
    mock_recorder = MagicMock()
    mock_recorder.async_add_executor_job = AsyncMock(side_effect=lambda f, *args: f(*args))
    
    with (
        patch(
            "homeassistant.components.recorder.get_instance",
            return_value=mock_recorder,
        ),
        patch(
            "homeassistant.components.recorder.history.get_significant_states",
            return_value=mock_history_states,
        ),
        patch("joblib.dump") as mock_dump,
    ):
        result = await train_model(hass, sensor_map, "mock_path.joblib")

        assert result["success"] is True
        assert isinstance(result["accuracy"], float)
        assert "last_trained" in result
        mock_dump.assert_called_once()


async def test_train_model_missing_sensor(hass: HomeAssistant):
    """Test training failure when sensor is missing from map."""
    result = await train_model(hass, {}, "mock_path.joblib")
    assert result["success"] is False
    assert "Missing" in result["error"]


async def test_train_model_insufficient_data(hass: HomeAssistant, sensor_map):
    """Test training failure with insufficient data."""
    mock_recorder = MagicMock()
    mock_recorder.async_add_executor_job = AsyncMock(side_effect=lambda f, *args: f(*args))

    with (
        patch(
            "homeassistant.components.recorder.get_instance",
            return_value=mock_recorder,
        ),
        patch(
            "homeassistant.components.recorder.history.get_significant_states",
            return_value={},
        ),
    ):
        result = await train_model(hass, sensor_map, "mock_path.joblib")
        assert result["success"] is False
        assert "Insufficient data sources" in result["error"]


async def test_train_model_exception(
    hass: HomeAssistant, sensor_map, mock_history_states
):
    """Test training failure on exception."""
    mock_recorder = MagicMock()
    mock_recorder.async_add_executor_job = AsyncMock(side_effect=lambda f, *args: f(*args))

    with (
        patch(
            "homeassistant.components.recorder.get_instance",
            return_value=mock_recorder,
        ),
        patch(
            "homeassistant.components.recorder.history.get_significant_states",
            return_value=mock_history_states,
        ),
        patch(
            "custom_components.micro_weather.trainer.RandomForest.fit",
            side_effect=Exception("Training error"),
        ),
    ):
        result = await train_model(hass, sensor_map, "mock_path.joblib")
        assert result["success"] is False
        assert "Training error" in result["error"]
