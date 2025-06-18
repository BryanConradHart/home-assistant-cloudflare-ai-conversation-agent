"""Integration for Cloudflare AI as a Home Assistant conversation agent."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv

from .config_flow import CONF_ECHO_CASE, ECHO_CASE_UPPER
from .const import DOMAIN

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
_PLATFORMS: list[Platform] = [Platform.CONVERSATION]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the Cloudflare AI conversation agent config entry."""
    hass.data.setdefault(DOMAIN, {})[CONF_ECHO_CASE] = entry.data.get(
        CONF_ECHO_CASE, ECHO_CASE_UPPER
    )
    await hass.config_entries.async_forward_entry_setups(entry, _PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload the Cloudflare AI conversation agent config entry."""
    return await hass.config_entries.async_unload_platforms(entry, _PLATFORMS)
