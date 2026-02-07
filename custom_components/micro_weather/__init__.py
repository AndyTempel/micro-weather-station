"""Micro Weather Station integration for Home Assistant.

This integration creates a smart weather station by analyzing existing sensor data
to determine accurate weather conditions for specific locations and microclimates.

The integration provides:
- Weather entity with current conditions and intelligent forecasts
- Support for multiple sensor types (temperature, humidity, pressure, wind, etc.)
- Advanced weather detection algorithms using real sensor data
- Multi-language support

Author: caplaz, AndyTempel (ML Fork)
License: MIT
"""

from collections import deque
from datetime import timedelta
import logging
import os
from typing import Any

import joblib

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    CONF_ENABLE_ML,
    DOMAIN,
    KEY_HUMIDITY,
    KEY_OUTDOOR_TEMP,
    KEY_PRESSURE,
    KEY_RAIN_RATE,
    KEY_SOLAR_RADIATION,
    KEY_WIND_DIRECTION,
    KEY_WIND_SPEED,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [
    Platform.WEATHER,
    Platform.BINARY_SENSOR,
]  # Added binary_sensor and kept weather platform


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Micro Weather Station from a config entry.

    This function initializes the integration by:
    1. Creating a data coordinator for managing sensor updates
    2. Setting up the weather platform
    3. Registering update listeners for configuration changes

    Args:
        hass: Home Assistant instance
        entry: Configuration entry containing user settings

    Returns:
        bool: True if setup was successful, False otherwise

    Raises:
        ConfigEntryNotReady: If required sensors are not available
    """
    _LOGGER.info("Setting up Micro Weather Station integration")

    # Create coordinator for managing updates
    coordinator = MicroWeatherCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()

    # Store coordinator in hass data
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = coordinator

    # Set up platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register services
    async def handle_train_model(call):
        """Handle the train_model service call."""
        if not entry.options.get(CONF_ENABLE_ML, False):
            _LOGGER.warning("ML training requested but ML is disabled in configuration")
            return

        model_path = hass.config.path("custom_components/micro_weather/weather_model.joblib")
        
        from .trainer import train_model
        
        # Execute training in executor thread
        result = await hass.async_add_executor_job(
            train_model, hass, entry.options, model_path
        )

        if result.get("success"):
            _LOGGER.info("ML Model training completed successfully")
            # Store training metadata in coordinator for binary sensor attributes
            coordinator.ml_accuracy = result.get("accuracy")
            coordinator.ml_last_trained = result.get("last_trained")
            # Reload the model in the coordinator
            await coordinator.async_reload_model()
            # Fire event for success
            hass.bus.async_fire(f"{DOMAIN}_training_success", result)
        else:
            _LOGGER.error("ML Model training failed: %s", result.get("error"))
            hass.bus.async_fire(f"{DOMAIN}_training_failed", result)

    hass.services.async_register(DOMAIN, "train", handle_train_model)

    # Set up options update listener for immediate refresh
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Update options and force immediate refresh.

    Called when user modifies integration configuration. Forces an immediate
    data refresh to apply new sensor mappings and settings.

    Args:
        hass: Home Assistant instance
        entry: Updated configuration entry
    """
    _LOGGER.info("Micro Weather Station config updated, refreshing data immediately")

    # Get the coordinator and force an immediate refresh
    coordinator = hass.data[DOMAIN][entry.entry_id]
    await coordinator.async_request_refresh()


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry.

    Cleanly removes the integration by unloading all platforms and
    removing stored data from Home Assistant.

    Args:
        hass: Home Assistant instance
        entry: Configuration entry to unload

    Returns:
        bool: True if unload was successful
    """
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)

    return unload_ok


class MicroWeatherCoordinator(DataUpdateCoordinator):
    """Class to manage fetching micro weather data.

    This coordinator handles periodic updates of weather data by:
    1. Reading configured sensor entities
    2. Processing sensor data through weather detection algorithms
    3. Providing consolidated weather information to the weather platform

    The coordinator respects user-configured update intervals and handles
    errors gracefully to maintain integration stability.
    """

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the weather data coordinator.

        Args:
            hass: Home Assistant instance
            entry: Configuration entry with sensor mappings and settings
        """
        self.entry = entry
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=5),
        )
        self.ml_active = False
        self.ml_model_loaded = False
        self._ml_model = None
        self.ml_accuracy = None
        self.ml_last_trained = None

        # Historical data storage (last 48 hours, 15-minute intervals =
        # ~192 readings)
        # We store this here so it persists across coordinator updates
        self.history_maxlen = 192
        self.sensor_history: dict[str, deque[dict[str, Any]]] = {
            KEY_OUTDOOR_TEMP: deque(maxlen=self.history_maxlen),
            KEY_HUMIDITY: deque(maxlen=self.history_maxlen),
            KEY_PRESSURE: deque(maxlen=self.history_maxlen),
            KEY_WIND_SPEED: deque(maxlen=self.history_maxlen),
            KEY_WIND_DIRECTION: deque(maxlen=self.history_maxlen),
            KEY_SOLAR_RADIATION: deque(maxlen=self.history_maxlen),
            KEY_RAIN_RATE: deque(maxlen=self.history_maxlen),
            "condition_history": deque(maxlen=self.history_maxlen),
        }

        # Try to load ML model
        model_path = hass.config.path(
            "custom_components/micro_weather/weather_model.joblib"
        )
        if os.path.exists(model_path):
            try:
                self._ml_model = joblib.load(model_path)
                self.ml_model_loaded = True
                _LOGGER.info("Successfully loaded ML weather model from %s", model_path)
            except Exception as err:
                _LOGGER.error("Failed to load ML weather model: %s", err)
        else:
            _LOGGER.debug("ML weather model not found at %s", model_path)

    async def async_reload_model(self) -> None:
        """Reload the ML model from disk."""
        model_path = self.hass.config.path(
            "custom_components/micro_weather/weather_model.joblib"
        )
        if os.path.exists(model_path):
            try:
                # Run joblib.load in executor as it's a blocking IO call
                self._ml_model = await self.hass.async_add_executor_job(
                    joblib.load, model_path
                )
                self.ml_model_loaded = True
                _LOGGER.info("Successfully reloaded ML weather model from %s", model_path)
            except Exception as err:
                _LOGGER.error("Failed to reload ML weather model: %s", err)
        else:
            _LOGGER.debug("ML weather model not found at %s", model_path)

    async def _async_update_data(self) -> dict[str, Any]:
        """Update weather data from real sensors.

        Fetches current data from all configured sensors and processes it
        through the weather detection algorithms to determine current conditions.

        Returns:
            dict: Weather data including current conditions, temperature, humidity,
                  pressure, wind data, and forecast information

        Raises:
            UpdateFailed: If critical sensors are unavailable or data is invalid
        """
        from .weather_detector import WeatherDetector

        try:
            detector = WeatherDetector(
                self.hass,
                self.entry.options,
                sensor_history=self.sensor_history,
                ml_model=self._ml_model if self.ml_model_loaded else None,
                enable_ml=self.entry.options.get(CONF_ENABLE_ML, False),
            )
            # Store analyzers on coordinator for weather entity access
            self.atmospheric_analyzer = detector.atmospheric_analyzer
            self.solar_analyzer = detector.solar_analyzer
            self.trends_analyzer = detector.trends_analyzer
            self.core_analyzer = detector.core_analyzer

            data = detector.get_weather_data()
            self.ml_active = data.get("ml_active", False)
            return data
        except Exception as err:
            _LOGGER.error("Error updating weather data: %s", err)
            raise UpdateFailed(f"Failed to update weather data: {err}") from err
