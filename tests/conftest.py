"""Test configuration for Micro Weather Station."""

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.micro_weather.const import (
    CONF_ENABLE_ML,
    CONF_HUMIDITY_SENSOR,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_UPDATE_INTERVAL,
    DOMAIN,
)


@pytest.fixture
def mock_config_entry():
    """Create a mock config entry."""
    return MockConfigEntry(
        domain=DOMAIN,
        data={},
        options={
            CONF_OUTDOOR_TEMP_SENSOR: "sensor.outdoor_temperature",
            CONF_HUMIDITY_SENSOR: "sensor.humidity",
            CONF_UPDATE_INTERVAL: 30,
            CONF_ENABLE_ML: False,
        },
        entry_id="test_entry_id",
        title="Micro Weather Station Test",
    )


@pytest.fixture
def mock_sensor_data():
    """Mock sensor data for testing."""
    return {
        "outdoor_temp": 72.0,  # Fahrenheit
        "indoor_temp": 70.0,
        "humidity": 65.0,
        "pressure": 29.92,  # inHg
        "wind_speed": 5.5,  # mph
        "wind_direction": 180.0,
        "wind_gust": 8.0,
        "rain_rate": 0.0,
        "rain_state": "Dry",
        "solar_radiation": 250.0,  # W/m²
        "solar_lux": 25000.0,
        "uv_index": 3.0,
    }
