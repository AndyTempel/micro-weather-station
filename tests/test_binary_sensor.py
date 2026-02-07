"""Tests for Micro Weather Station binary sensors."""

from unittest.mock import MagicMock

from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.core import HomeAssistant

from custom_components.micro_weather.binary_sensor import (
    MicroWeatherAISensor,
    async_setup_entry,
)
from custom_components.micro_weather.const import DOMAIN


async def test_binary_sensor_setup(hass: HomeAssistant, mock_config_entry):
    """Test binary sensor setup."""
    mock_coordinator = MagicMock()
    hass.data[DOMAIN] = {mock_config_entry.entry_id: mock_coordinator}

    async_add_entities = MagicMock()
    await async_setup_entry(hass, mock_config_entry, async_add_entities)

    assert async_add_entities.called
    assert len(async_add_entities.call_args[0][0]) == 1
    assert isinstance(async_add_entities.call_args[0][0][0], MicroWeatherAISensor)


async def test_ai_active_sensor_properties(hass: HomeAssistant, mock_config_entry):
    """Test AI active binary sensor properties."""
    mock_coordinator = MagicMock()
    mock_coordinator.ml_active = True
    mock_coordinator.ml_model_loaded = True
    mock_coordinator.ml_accuracy = 0.85
    mock_coordinator.ml_last_trained = "2025-01-01T12:00:00"

    sensor = MicroWeatherAISensor(mock_coordinator, mock_config_entry)

    assert sensor.name == "AI Active"
    assert sensor.device_class == BinarySensorDeviceClass.RUNNING
    assert sensor.is_on is True
    assert sensor.extra_state_attributes == {
        "model_loaded": True,
        "training_accuracy": 0.85,
        "last_trained": "2025-01-01T12:00:00",
    }
    assert sensor.unique_id == f"{mock_config_entry.entry_id}_ai_active"
    assert sensor.device_info["name"] == "Micro Weather Station"


async def test_ai_active_sensor_off(hass: HomeAssistant, mock_config_entry):
    """Test AI active binary sensor when off."""
    mock_coordinator = MagicMock()
    mock_coordinator.ml_active = False
    mock_coordinator.ml_model_loaded = False
    mock_coordinator.ml_accuracy = None
    mock_coordinator.ml_last_trained = None

    sensor = MicroWeatherAISensor(mock_coordinator, mock_config_entry)

    assert sensor.is_on is False
    assert sensor.extra_state_attributes["model_loaded"] is False
