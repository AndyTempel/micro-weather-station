"""Integration tests for Micro Weather Station setup and unload flows."""

from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.components.weather import (
    ATTR_CONDITION_CLOUDY,
    ATTR_CONDITION_PARTLYCLOUDY,
    ATTR_CONDITION_RAINY,
    ATTR_CONDITION_SUNNY,
)
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import UpdateFailed
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.micro_weather import (
    MicroWeatherCoordinator,
    async_setup_entry,
    async_unload_entry,
    async_update_options,
)
from custom_components.micro_weather.const import CONF_ENABLE_ML, DOMAIN, KEY_CONDITION


@pytest.mark.integration
class TestIntegrationSetup:
    """Test integration setup and unload flows."""

    async def test_async_setup_entry_success(
        self, hass: HomeAssistant, mock_config_entry: MockConfigEntry
    ):
        """Test successful setup of the integration."""
        # Mock the weather detector to return valid data
        mock_weather_data = {
            "condition": ATTR_CONDITION_SUNNY,
            "temperature": 22.5,
            "humidity": 65.0,
            "pressure": 1013.25,
            "wind_speed": 5.2,
            "wind_direction": 180.0,
            "visibility": 10.0,
        }

        with (
            patch(
                "custom_components.micro_weather.weather_detector.WeatherDetector"
            ) as mock_detector_class,
            patch(
                "homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"
            ) as mock_forward,
        ):
            mock_detector = MagicMock()
            mock_detector.get_weather_data.return_value = mock_weather_data
            mock_detector_class.return_value = mock_detector

            # Setup the entry
            mock_config_entry.mock_state(hass, ConfigEntryState.SETUP_IN_PROGRESS)
            result = await async_setup_entry(hass, mock_config_entry)

            assert result is True
            assert DOMAIN in hass.data
            assert mock_config_entry.entry_id in hass.data[DOMAIN]

            # Check coordinator was created and stored
            coordinator = hass.data[DOMAIN][mock_config_entry.entry_id]
            assert isinstance(coordinator, MicroWeatherCoordinator)

            # Verify platform setup was called
            mock_forward.assert_called_once_with(
                mock_config_entry, ["weather", "binary_sensor"]
            )

    async def test_async_setup_entry_coordinator_refresh_failure(
        self, hass: HomeAssistant, mock_config_entry: MockConfigEntry
    ):
        """Test setup failure when coordinator refresh fails."""
        from homeassistant.exceptions import ConfigEntryNotReady

        with patch(
            "custom_components.micro_weather.weather_detector.WeatherDetector"
        ) as mock_detector_class:
            mock_detector = MagicMock()
            mock_detector.get_weather_data.side_effect = Exception("Sensor unavailable")
            mock_detector_class.return_value = mock_detector

            # Setup should fail when initial refresh fails
            mock_config_entry.mock_state(hass, ConfigEntryState.SETUP_IN_PROGRESS)
            with pytest.raises(ConfigEntryNotReady):
                await async_setup_entry(hass, mock_config_entry)

    async def test_async_unload_entry_success(
        self, hass: HomeAssistant, mock_config_entry: MockConfigEntry
    ):
        """Test successful unload of the integration."""
        # First set up the integration
        mock_weather_data = {KEY_CONDITION: ATTR_CONDITION_CLOUDY, "temperature": 20.0}

        with (
            patch(
                "custom_components.micro_weather.weather_detector.WeatherDetector"
            ) as mock_detector_class,
            patch(
                "homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"
            ),
            patch(
                "homeassistant.config_entries.ConfigEntries.async_unload_platforms"
            ) as mock_unload,
        ):
            mock_detector = MagicMock()
            mock_detector.get_weather_data.return_value = mock_weather_data
            mock_detector_class.return_value = mock_detector

            # Setup first
            mock_config_entry.mock_state(hass, ConfigEntryState.SETUP_IN_PROGRESS)
            await async_setup_entry(hass, mock_config_entry)
            mock_unload.reset_mock()

            # Mock successful platform unload
            mock_unload.return_value = True

            # Now test unload
            result = await async_unload_entry(hass, mock_config_entry)

            assert result is True
            mock_unload.assert_called_once_with(
                mock_config_entry, ["weather", "binary_sensor"]
            )

            # Check data was cleaned up
            assert mock_config_entry.entry_id not in hass.data[DOMAIN]

    async def test_async_unload_entry_platform_unload_failure(
        self, hass: HomeAssistant, mock_config_entry: MockConfigEntry
    ):
        """Test unload failure when platform unload fails."""
        # First set up the integration
        mock_weather_data = {KEY_CONDITION: ATTR_CONDITION_RAINY, "temperature": 18.0}

        with (
            patch(
                "custom_components.micro_weather.weather_detector.WeatherDetector"
            ) as mock_detector_class,
            patch(
                "homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"
            ),
            patch(
                "homeassistant.config_entries.ConfigEntries.async_unload_platforms"
            ) as mock_unload,
        ):
            mock_detector = MagicMock()
            mock_detector.get_weather_data.return_value = mock_weather_data
            mock_detector_class.return_value = mock_detector

            # Setup first
            mock_config_entry.mock_state(hass, ConfigEntryState.SETUP_IN_PROGRESS)
            await async_setup_entry(hass, mock_config_entry)

            # Mock platform unload failure
            mock_unload.return_value = False

            # Test unload
            result = await async_unload_entry(hass, mock_config_entry)

            assert result is False
            # Data should NOT be cleaned up when platform unload fails
            assert mock_config_entry.entry_id in hass.data[DOMAIN]

    async def test_async_update_options(
        self, hass: HomeAssistant, mock_config_entry: MockConfigEntry
    ):
        """Test options update triggers coordinator refresh."""
        # First set up the integration
        mock_weather_data = {KEY_CONDITION: ATTR_CONDITION_SUNNY, "temperature": 25.0}

        with (
            patch(
                "custom_components.micro_weather.weather_detector.WeatherDetector"
            ) as mock_detector_class,
            patch(
                "homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"
            ),
        ):
            mock_detector = MagicMock()
            mock_detector.get_weather_data.return_value = mock_weather_data
            mock_detector_class.return_value = mock_detector

            # Setup first
            mock_config_entry.mock_state(hass, ConfigEntryState.SETUP_IN_PROGRESS)
            await async_setup_entry(hass, mock_config_entry)

            coordinator = hass.data[DOMAIN][mock_config_entry.entry_id]
            coordinator.async_request_refresh = AsyncMock()

            # Update options
            await async_update_options(hass, mock_config_entry)

            # Verify refresh was called
            coordinator.async_request_refresh.assert_called_once()


@pytest.mark.integration
class TestIntegrationServices:
    """Test integration services."""

    async def test_train_service_ml_disabled(
        self, hass: HomeAssistant, mock_config_entry: MockConfigEntry
    ):
        """Test train service when ML is disabled."""
        # Add it to HA
        mock_config_entry.add_to_hass(hass)
        # Ensure ML is disabled
        hass.config_entries.async_update_entry(
            mock_config_entry,
            options={**mock_config_entry.options, CONF_ENABLE_ML: False},
        )
        mock_config_entry.mock_state(hass, ConfigEntryState.SETUP_IN_PROGRESS)

        with (
            patch("custom_components.micro_weather.trainer.train_model") as mock_train,
            patch(
                "homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"
            ),
        ):
            await async_setup_entry(hass, mock_config_entry)
            await hass.services.async_call(DOMAIN, "train", {}, blocking=True)
            mock_train.assert_not_called()

    async def test_train_service_ml_enabled_success(
        self, hass: HomeAssistant, mock_config_entry: MockConfigEntry
    ):
        """Test train service when ML is enabled and succeeds."""
        # Add it to HA
        mock_config_entry.add_to_hass(hass)
        hass.config_entries.async_update_entry(
            mock_config_entry,
            options={**mock_config_entry.options, CONF_ENABLE_ML: True},
        )
        mock_config_entry.mock_state(hass, ConfigEntryState.SETUP_IN_PROGRESS)

        mock_result = {
            "success": True,
            "accuracy": 0.95,
            "last_trained": "2025-01-01T12:00:00",
        }

        with (
            patch(
                "custom_components.micro_weather.trainer.train_model",
                return_value=mock_result,
            ),
            patch(
                "homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"
            ),
        ):
            await async_setup_entry(hass, mock_config_entry)

            # Use a mock for async_add_executor_job to avoid actual threading
            with patch.object(
                hass, "async_add_executor_job", AsyncMock(return_value=mock_result)
            ):
                await hass.services.async_call(DOMAIN, "train", {}, blocking=True)

            coordinator = hass.data[DOMAIN][mock_config_entry.entry_id]
            assert coordinator.ml_accuracy == 0.95
            assert coordinator.ml_last_trained == "2025-01-01T12:00:00"

    async def test_train_service_ml_enabled_failure(
        self, hass: HomeAssistant, mock_config_entry: MockConfigEntry
    ):
        """Test train service when ML is enabled and fails."""
        # Add it to HA
        mock_config_entry.add_to_hass(hass)
        hass.config_entries.async_update_entry(
            mock_config_entry,
            options={**mock_config_entry.options, CONF_ENABLE_ML: True},
        )
        mock_config_entry.mock_state(hass, ConfigEntryState.SETUP_IN_PROGRESS)

        mock_result = {
            "success": False,
            "error": "Not enough data",
        }

        with (
            patch(
                "homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"
            ),
        ):
            await async_setup_entry(hass, mock_config_entry)

            with patch.object(
                hass, "async_add_executor_job", AsyncMock(return_value=mock_result)
            ):
                await hass.services.async_call(DOMAIN, "train", {}, blocking=True)

            coordinator = hass.data[DOMAIN][mock_config_entry.entry_id]
            assert coordinator.ml_accuracy is None


@pytest.mark.integration
class TestCoordinatorIntegration:
    """Test the MicroWeatherCoordinator integration."""

    async def test_coordinator_model_load_failure(
        self, hass: HomeAssistant, mock_config_entry: MockConfigEntry
    ):
        """Test coordinator handles model loading failure."""
        with (
            patch("custom_components.micro_weather.os.path.exists", return_value=True),
            patch(
                "custom_components.micro_weather.joblib.load",
                side_effect=Exception("Load error"),
            ),
        ):
            coordinator = MicroWeatherCoordinator(hass, mock_config_entry)
            assert coordinator.ml_model_loaded is False

    async def test_coordinator_model_load_not_found(
        self, hass: HomeAssistant, mock_config_entry: MockConfigEntry
    ):
        """Test coordinator handles model not found during init."""
        with patch(
            "custom_components.micro_weather.os.path.exists", return_value=False
        ):
            coordinator = MicroWeatherCoordinator(hass, mock_config_entry)
            assert coordinator.ml_model_loaded is False

    async def test_coordinator_async_reload_model(
        self, hass: HomeAssistant, mock_config_entry: MockConfigEntry
    ):
        """Test coordinator async model reloading."""
        coordinator = MicroWeatherCoordinator(hass, mock_config_entry)
        assert coordinator.ml_model_loaded is False

        with (
            patch("custom_components.micro_weather.os.path.exists", return_value=True),
            patch.object(
                hass, "async_add_executor_job", AsyncMock(return_value=MagicMock())
            ),
        ):
            await coordinator.async_reload_model()
            assert coordinator.ml_model_loaded is True

    async def test_coordinator_async_reload_model_not_found(
        self, hass: HomeAssistant, mock_config_entry: MockConfigEntry
    ):
        """Test coordinator async model reloading when file not found."""
        coordinator = MicroWeatherCoordinator(hass, mock_config_entry)

        with patch(
            "custom_components.micro_weather.os.path.exists", return_value=False
        ):
            await coordinator.async_reload_model()
            assert coordinator.ml_model_loaded is False

    async def test_coordinator_async_reload_model_failure(
        self, hass: HomeAssistant, mock_config_entry: MockConfigEntry
    ):
        """Test coordinator async model reloading failure."""
        coordinator = MicroWeatherCoordinator(hass, mock_config_entry)

        with (
            patch("custom_components.micro_weather.os.path.exists", return_value=True),
            patch.object(
                hass,
                "async_add_executor_job",
                AsyncMock(side_effect=Exception("Reload error")),
            ),
        ):
            await coordinator.async_reload_model()
            assert coordinator.ml_model_loaded is False

    async def test_coordinator_update_success(
        self, hass: HomeAssistant, mock_config_entry: MockConfigEntry
    ):
        """Test successful data update by coordinator."""
        mock_weather_data = {
            "condition": ATTR_CONDITION_PARTLYCLOUDY,
            "temperature": 23.5,
            "humidity": 70.0,
            "pressure": 1015.0,
            "wind_speed": 3.2,
            "forecast": [
                {
                    "datetime": "2025-09-30T12:00:00",
                    "temperature": 24.0,
                    "templow": 18.0,
                    "condition": ATTR_CONDITION_SUNNY,
                    "precipitation": 0.0,
                    "wind_speed": 4.0,
                    "humidity": 65.0,
                }
            ],
        }

        with patch(
            "custom_components.micro_weather.weather_detector.WeatherDetector"
        ) as mock_detector_class:
            mock_detector = MagicMock()
            mock_detector.get_weather_data.return_value = mock_weather_data
            mock_detector_class.return_value = mock_detector

            coordinator = MicroWeatherCoordinator(hass, mock_config_entry)

            # Test update
            result = await coordinator._async_update_data()

            assert result == mock_weather_data
            mock_detector_class.assert_called_once_with(
                hass,
                mock_config_entry.options,
                sensor_history=coordinator.sensor_history,
                ml_model=None,
                enable_ml=False,
            )

    async def test_coordinator_update_failure(
        self, hass: HomeAssistant, mock_config_entry: MockConfigEntry
    ):
        """Test coordinator handles update failures gracefully."""
        with patch(
            "custom_components.micro_weather.weather_detector.WeatherDetector"
        ) as mock_detector_class:
            mock_detector = MagicMock()
            mock_detector.get_weather_data.side_effect = ValueError(
                "Invalid sensor data"
            )
            mock_detector_class.return_value = mock_detector

            coordinator = MicroWeatherCoordinator(hass, mock_config_entry)

            # Test update failure
            with pytest.raises(UpdateFailed, match="Failed to update weather data"):
                await coordinator._async_update_data()

    async def test_coordinator_initialization(
        self, hass: HomeAssistant, mock_config_entry: MockConfigEntry
    ):
        """Test coordinator initialization with proper config."""
        coordinator = MicroWeatherCoordinator(hass, mock_config_entry)

        assert coordinator.entry == mock_config_entry
        assert coordinator.name == DOMAIN
        assert coordinator.update_interval is not None  # Should be set to 5 minutes
        assert coordinator.hass == hass
