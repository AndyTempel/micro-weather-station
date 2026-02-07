"""Tests for Micro Weather Station sensors."""

from unittest.mock import MagicMock

from homeassistant.core import HomeAssistant

from custom_components.micro_weather.const import DOMAIN, SENSOR_TYPES
from custom_components.micro_weather.sensor import MicroWeatherSensor, async_setup_entry


async def test_sensor_setup(hass: HomeAssistant, mock_config_entry):
    """Test sensor setup."""
    mock_coordinator = MagicMock()
    hass.data[DOMAIN] = {mock_config_entry.entry_id: mock_coordinator}

    async_add_entities = MagicMock()
    await async_setup_entry(hass, mock_config_entry, async_add_entities)

    assert async_add_entities.called
    assert len(async_add_entities.call_args[0][0]) == len(SENSOR_TYPES)
    assert isinstance(async_add_entities.call_args[0][0][0], MicroWeatherSensor)


async def test_sensor_properties(hass: HomeAssistant, mock_config_entry):
    """Test sensor properties."""
    mock_coordinator = MagicMock()
    mock_coordinator.data = {"temperature": 22.5}
    mock_coordinator.last_update_success = True

    sensor = MicroWeatherSensor(mock_coordinator, mock_config_entry, "temperature")

    assert sensor.name == "Temperature"
    assert sensor.native_value == 22.5
    assert sensor.native_unit_of_measurement == "°C"
    assert sensor.icon == "mdi:thermometer"
    assert sensor.available is True
    assert sensor.unique_id == f"{mock_config_entry.entry_id}_temperature"


async def test_sensor_no_data(hass: HomeAssistant, mock_config_entry):
    """Test sensor when no data is available."""
    mock_coordinator = MagicMock()
    mock_coordinator.data = None
    mock_coordinator.last_update_success = False

    sensor = MicroWeatherSensor(mock_coordinator, mock_config_entry, "temperature")

    assert sensor.native_value is None
    assert sensor.available is False
