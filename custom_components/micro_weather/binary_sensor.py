"""Binary sensor platform for Micro Weather Station."""

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .version import __version__


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Micro Weather Station binary sensor entities."""
    coordinator = hass.data[DOMAIN][config_entry.entry_id]

    async_add_entities([MicroWeatherAISensor(coordinator, config_entry)])


class MicroWeatherAISensor(CoordinatorEntity, BinarySensorEntity):
    """Binary sensor to indicate if AI model is active."""

    _attr_has_entity_name = True
    _attr_name = "AI Active"
    _attr_device_class = BinarySensorDeviceClass.RUNNING

    def __init__(self, coordinator, config_entry):
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._config_entry = config_entry
        self._attr_unique_id = f"{config_entry.entry_id}_ai_active"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, config_entry.entry_id)},
            "name": "Micro Weather Station",
            "manufacturer": "Micro Weather",
            "model": "MWS-1",
            "sw_version": __version__,
        }

    @property
    def is_on(self) -> bool:
        """Return true if the AI model is active."""
        return self.coordinator.ml_active

    @property
    def extra_state_attributes(self):
        """Return extra state attributes."""
        return {
            "model_loaded": self.coordinator.ml_model_loaded,
            "training_accuracy": self.coordinator.ml_accuracy,
            "last_trained": self.coordinator.ml_last_trained,
        }
