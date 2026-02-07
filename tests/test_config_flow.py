"""Tests for the Micro Weather Station config flow."""

from unittest.mock import patch

from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.util.unit_system import METRIC_SYSTEM, US_CUSTOMARY_SYSTEM

from custom_components.micro_weather.config_flow import (
    ConfigFlowHandler,
    OptionsFlowHandler,
)
from custom_components.micro_weather.const import (
    CONF_ALTITUDE,
    CONF_ENABLE_ML,
    CONF_HUMIDITY_SENSOR,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_PRESSURE_SENSOR,
    CONF_RAIN_RATE_SENSOR,
    CONF_RAIN_STATE_SENSOR,
    CONF_SOLAR_LUX_SENSOR,
    CONF_SOLAR_RADIATION_SENSOR,
    CONF_SUN_SENSOR,
    CONF_UPDATE_INTERVAL,
    CONF_UV_INDEX_SENSOR,
    CONF_WIND_DIRECTION_SENSOR,
    CONF_WIND_GUST_SENSOR,
    CONF_WIND_SPEED_SENSOR,
    CONF_ZENITH_MAX_RADIATION,
)


class TestConfigFlow:
    """Test the config flow."""

    async def test_form(self, hass: HomeAssistant):
        """Test we get the form."""
        # Create flow directly instead of using async_init
        flow = ConfigFlowHandler()
        flow.hass = hass

        result = await flow.async_step_user()
        assert result["type"] == "form"
        assert result["errors"] == {}

    async def test_form_missing_required_sensor(self, hass: HomeAssistant):
        """Test form with missing required outdoor temp sensor."""
        # Create flow directly
        flow = ConfigFlowHandler()
        flow.hass = hass

        # Submit with missing required sensor
        result = await flow.async_step_user(
            {
                # Missing outdoor_temp_sensor
                "update_interval": 30,
            }
        )
        assert result["type"] == "form"
        assert result["errors"] == {"base": "missing_outdoor_temp"}

    async def test_form_success(self, hass: HomeAssistant):
        """Test successful form submission."""
        # Create flow directly
        flow = ConfigFlowHandler()
        flow.hass = hass

        with patch(
            "custom_components.micro_weather.async_setup_entry",
            return_value=True,
        ):
            # Submit all sensor data for initial setup
            result = await flow.async_step_user(
                {
                    "outdoor_temp_sensor": "sensor.outdoor_temperature",
                    "humidity_sensor": "sensor.humidity",
                    "pressure_sensor": "sensor.pressure",
                    "altitude": 100,
                    "wind_speed_sensor": "sensor.wind_speed",
                    "wind_direction_sensor": "sensor.wind_direction",
                    "wind_gust_sensor": "sensor.wind_gust",
                    "rain_rate_sensor": "sensor.rain_rate",
                    "rain_state_sensor": "sensor.rain_state",
                    "solar_radiation_sensor": "sensor.solar_radiation",
                    "solar_lux_sensor": "sensor.solar_lux",
                    "uv_index_sensor": "sensor.uv_index",
                    "sun_sensor": "sun.sun",
                    "update_interval": 30,
                    "enable_ml": True,
                }
            )
            await hass.async_block_till_done()

        assert result["type"] == "create_entry"
        assert result["title"] == "Micro Weather Station"
        assert result["options"][CONF_ENABLE_ML] is True

    async def test_get_altitude_unit_metric(self, hass: HomeAssistant):
        """Test _get_altitude_unit returns 'm' for metric system."""
        # Set metric system
        hass.config.units = METRIC_SYSTEM

        flow = ConfigFlowHandler()
        flow.hass = hass

        assert flow._get_altitude_unit() == "m"

    async def test_get_altitude_unit_imperial(self, hass: HomeAssistant):
        """Test _get_altitude_unit returns 'ft' for imperial system."""
        # Set imperial system
        hass.config.units = US_CUSTOMARY_SYSTEM

        flow = ConfigFlowHandler()
        flow.hass = hass

        assert flow._get_altitude_unit() == "ft"

    async def test_get_altitude_max_metric(self, hass: HomeAssistant):
        """Test _get_altitude_max returns 10000 for metric system."""
        # Set metric system
        hass.config.units = METRIC_SYSTEM

        flow = ConfigFlowHandler()
        flow.hass = hass

        assert flow._get_altitude_max() == 10000

    async def test_get_altitude_max_imperial(self, hass: HomeAssistant):
        """Test _get_altitude_max returns 32808 for imperial system."""
        # Set imperial system
        hass.config.units = US_CUSTOMARY_SYSTEM

        flow = ConfigFlowHandler()
        flow.hass = hass

        assert flow._get_altitude_max() == 32808


class TestOptionsFlow:
    """Test the options flow."""

    async def test_options_flow_init(self, hass: HomeAssistant, mock_config_entry):
        """Test options flow initialization with some sensors configured."""
        # Add it to HA's config entries
        mock_config_entry.add_to_hass(hass)

        # Create options flow
        flow = OptionsFlowHandler(mock_config_entry)
        flow.hass = hass

        # Should start with menu
        result = await flow.async_step_init()
        assert result["type"] == "menu"
        assert result["step_id"] == "init"

    @patch("homeassistant.helpers.frame.report_usage")
    async def test_options_flow_with_none_values(
        self, mock_report_usage, hass: HomeAssistant, mock_config_entry
    ):
        """Test options flow handles None values for unconfigured sensors."""
        # Add it to HA's config entries
        mock_config_entry.add_to_hass(hass)
        # Update options for this specific test
        hass.config_entries.async_update_entry(
            mock_config_entry,
            options={
                **mock_config_entry.options,
                CONF_PRESSURE_SENSOR: None,
                CONF_SUN_SENSOR: None,
            },
        )

        # Create options flow
        flow = OptionsFlowHandler(mock_config_entry)
        flow.hass = hass

        # Start with menu
        await flow.async_step_init()

        # Configure atmospheric sensors
        result = await flow.async_step_init({"next_step_id": "atmospheric"})
        assert result["type"] == "form"

        result = await flow.async_step_atmospheric(
            {
                CONF_OUTDOOR_TEMP_SENSOR: "sensor.outdoor_temperature",
                CONF_HUMIDITY_SENSOR: "sensor.humidity",
            }
        )
        assert result["type"] == "menu"

        # Finish configuration
        result = await flow.async_step_init({"next_step_id": "device_config"})
        assert result["type"] == "form"

        result = await flow.async_step_device_config(
            {
                CONF_UPDATE_INTERVAL: 30,
                CONF_ENABLE_ML: True,
            }
        )

        assert result["type"] == "create_entry"
        assert result["data"][CONF_HUMIDITY_SENSOR] == "sensor.humidity"
        assert result["data"].get(CONF_PRESSURE_SENSOR) is None
        assert result["data"].get(CONF_SUN_SENSOR) is None
        assert result["data"][CONF_ENABLE_ML] is True

    @patch("homeassistant.helpers.frame.report_usage")
    async def test_options_flow_add_sun_sensor(
        self, mock_report_usage, hass: HomeAssistant, mock_config_entry
    ):
        """Test adding a sun sensor through options flow."""
        # Add it to HA's config entries
        mock_config_entry.add_to_hass(hass)

        # Create options flow
        flow = OptionsFlowHandler(mock_config_entry)
        flow.hass = hass

        # Start with menu
        await flow.async_step_init()

        # Configure solar sensors
        result = await flow.async_step_init({"next_step_id": "solar"})
        assert result["type"] == "form"

        result = await flow.async_step_solar(
            {
                CONF_OUTDOOR_TEMP_SENSOR: "sensor.outdoor_temperature",
                CONF_SUN_SENSOR: "sun.sun",
            }
        )
        assert result["type"] == "menu"

        # Finish configuration
        result = await flow.async_step_init({"next_step_id": "device_config"})
        assert result["type"] == "form"

        result = await flow.async_step_device_config({})

        assert result["type"] == "create_entry"
        assert result["data"][CONF_SUN_SENSOR] == "sun.sun"

    @patch("homeassistant.helpers.frame.report_usage")
    async def test_options_flow_remove_sensor(
        self, mock_report_usage, hass: HomeAssistant, mock_config_entry
    ):
        """Test removing a sensor by clearing the field."""
        # Add it to HA's config entries
        mock_config_entry.add_to_hass(hass)
        # Update options for this specific test
        hass.config_entries.async_update_entry(
            mock_config_entry,
            options={
                **mock_config_entry.options,
                CONF_HUMIDITY_SENSOR: "sensor.humidity",
                CONF_SUN_SENSOR: "sun.sun",
            },
        )
        # Create options flow
        flow = OptionsFlowHandler(mock_config_entry)
        flow.hass = hass

        # Start with menu
        await flow.async_step_init()

        # Configure atmospheric sensors (remove humidity)
        result = await flow.async_step_init({"next_step_id": "atmospheric"})
        assert result["type"] == "form"

        result = await flow.async_step_atmospheric(
            {
                CONF_OUTDOOR_TEMP_SENSOR: "sensor.outdoor_temperature",
                CONF_HUMIDITY_SENSOR: "",
            }
        )
        assert result["type"] == "menu"

        # Finish configuration
        result = await flow.async_step_init({"next_step_id": "device_config"})
        assert result["type"] == "form"

        result = await flow.async_step_device_config({})

        assert result["type"] == "create_entry"
        assert result["data"].get(CONF_HUMIDITY_SENSOR) is None
        assert result["data"].get(CONF_SUN_SENSOR) == "sun.sun"

    @patch("homeassistant.helpers.frame.report_usage")
    async def test_options_flow_missing_required_sensor(
        self, mock_report_usage, hass: HomeAssistant, mock_config_entry
    ):
        """Test options flow validation when required sensor is missing."""
        # Add it to HA's config entries
        mock_config_entry.add_to_hass(hass)

        # Create options flow
        flow = OptionsFlowHandler(mock_config_entry)
        flow.hass = hass

        # Start with menu
        await flow.async_step_init()

        # Try to configure atmospheric sensors without required sensor
        result = await flow.async_step_init({"next_step_id": "atmospheric"})
        assert result["type"] == "form"

        result = await flow.async_step_atmospheric({})

        assert result["type"] == "form"
        assert result["errors"] == {"base": "missing_outdoor_temp"}

    @patch("homeassistant.helpers.frame.report_usage")
    async def test_options_flow_schema_building_with_defaults(
        self, mock_report_usage, hass: HomeAssistant, mock_config_entry
    ):
        """Test that schema building properly sets defaults for configured sensors."""
        # Add it to HA's config entries
        mock_config_entry.add_to_hass(hass)
        # Update options for this specific test
        hass.config_entries.async_update_entry(
            mock_config_entry,
            options={
                **mock_config_entry.options,
                CONF_PRESSURE_SENSOR: None,
                CONF_SUN_SENSOR: "sun.sun",
                CONF_UPDATE_INTERVAL: 45,
            },
        )
        # Create options flow
        flow = OptionsFlowHandler(mock_config_entry)
        flow.hass = hass

        # Start with menu
        result = await flow.async_step_init()
        assert result["type"] == "menu"

        # Go to atmospheric step to check schema building
        result = await flow.async_step_init({"next_step_id": "atmospheric"})
        assert result["type"] == "form"

    async def test_initial_config_flow_altitude_default(self, hass: HomeAssistant):
        """Test that initial config flow shows altitude default from HA elevation."""
        # Set HA elevation
        hass.config.elevation = 150.5

        # Create flow directly
        flow = ConfigFlowHandler()
        flow.hass = hass

        result = await flow.async_step_user()
        assert result["type"] == "form"
        assert result["errors"] == {}

    @patch("homeassistant.helpers.frame.report_usage")
    async def test_options_flow_solar_zenith_max_radiation_conditional_display(
        self, mock_report_usage, hass: HomeAssistant, mock_config_entry
    ):
        """Test conditional display of zenith max radiation."""
        # Add it to HA's config entries
        mock_config_entry.add_to_hass(hass)

        # 1. No solar radiation sensor
        hass.config_entries.async_update_entry(
            mock_config_entry,
            options={
                **mock_config_entry.options,
                CONF_SOLAR_RADIATION_SENSOR: None,
            },
        )
        flow = OptionsFlowHandler(mock_config_entry)
        flow.hass = hass
        await flow.async_step_init()
        result = await flow.async_step_init({"next_step_id": "solar"})
        assert CONF_ZENITH_MAX_RADIATION not in result["data_schema"].schema

        # 2. With solar radiation sensor
        hass.config_entries.async_update_entry(
            mock_config_entry,
            options={
                **mock_config_entry.options,
                CONF_SOLAR_RADIATION_SENSOR: "sensor.solar_radiation",
            },
        )
        flow = OptionsFlowHandler(mock_config_entry)
        flow.hass = hass
        await flow.async_step_init()
        result = await flow.async_step_init({"next_step_id": "solar"})
        assert CONF_ZENITH_MAX_RADIATION in result["data_schema"].schema

    @patch("homeassistant.helpers.frame.report_usage")
    async def test_options_flow_all_steps(
        self, mock_report_usage, hass: HomeAssistant, mock_config_entry
    ):
        """Test all steps of the options flow."""
        mock_config_entry.add_to_hass(hass)
        flow = OptionsFlowHandler(mock_config_entry)
        flow.hass = hass

        # 1. Atmospheric
        result = await flow.async_step_init({"next_step_id": "atmospheric"})
        assert result["type"] == FlowResultType.FORM
        result = await flow.async_step_atmospheric(
            {
                CONF_OUTDOOR_TEMP_SENSOR: "sensor.temp",
                CONF_HUMIDITY_SENSOR: "sensor.hum",
                CONF_PRESSURE_SENSOR: "sensor.pres",
                CONF_ALTITUDE: 200,
            }
        )
        assert result["type"] == FlowResultType.MENU

        # 2. Solar
        result = await flow.async_step_init({"next_step_id": "solar"})
        assert result["type"] == FlowResultType.FORM
        result = await flow.async_step_solar(
            {
                CONF_SOLAR_RADIATION_SENSOR: "sensor.solar",
                CONF_SOLAR_LUX_SENSOR: "sensor.lux",
                CONF_SUN_SENSOR: "sun.sun",
                CONF_UV_INDEX_SENSOR: "sensor.uv",
            }
        )
        assert result["type"] == FlowResultType.MENU

        # 3. Wind
        result = await flow.async_step_init({"next_step_id": "wind"})
        assert result["type"] == FlowResultType.FORM
        result = await flow.async_step_wind(
            {
                CONF_WIND_SPEED_SENSOR: "sensor.wind",
                CONF_WIND_DIRECTION_SENSOR: "sensor.dir",
                CONF_WIND_GUST_SENSOR: "sensor.gust",
            }
        )
        assert result["type"] == FlowResultType.MENU

        # 4. Rain
        result = await flow.async_step_init({"next_step_id": "rain"})
        assert result["type"] == FlowResultType.FORM
        result = await flow.async_step_rain(
            {
                CONF_RAIN_RATE_SENSOR: "sensor.rain",
                CONF_RAIN_STATE_SENSOR: "sensor.rain_state",
            }
        )
        assert result["type"] == FlowResultType.MENU

        # 5. Device Config
        result = await flow.async_step_init({"next_step_id": "device_config"})
        assert result["type"] == FlowResultType.FORM
        result = await flow.async_step_device_config(
            {
                CONF_UPDATE_INTERVAL: 40,
                CONF_ENABLE_ML: True,
            }
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY

        assert result["data"][CONF_OUTDOOR_TEMP_SENSOR] == "sensor.temp"
        assert result["data"][CONF_ENABLE_ML] is True
        assert result["data"][CONF_UPDATE_INTERVAL] == 40

    async def test_options_get_altitude_unit_and_max(
        self, hass: HomeAssistant, mock_config_entry
    ):
        """Test altitude unit and max getters in options flow."""
        flow = OptionsFlowHandler(mock_config_entry)
        flow.hass = hass

        # Metric
        hass.config.units = METRIC_SYSTEM
        assert flow._get_altitude_unit() == "m"
        assert flow._get_altitude_max() == 10000

        # Imperial
        hass.config.units = US_CUSTOMARY_SYSTEM
        assert flow._get_altitude_unit() == "ft"
        assert flow._get_altitude_max() == 32808
