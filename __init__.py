"""Integration for Cloudflare AI as a Home Assistant conversation agent."""

from __future__ import annotations

import cloudflare

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv

from .const import CONF_API_TOKEN, DOMAIN

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
_PLATFORMS: list[Platform] = [Platform.CONVERSATION]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the Cloudflare AI conversation agent config entry."""
    # Construct and store the Cloudflare client for this entry
    user_token = entry.data.get(CONF_API_TOKEN) or entry.options.get(CONF_API_TOKEN)
    if user_token:
        client = cloudflare.AsyncCloudflare(api_token=user_token)
        hass.data.setdefault(DOMAIN, {})[entry.entry_id] = client
    else:
        hass.data[DOMAIN][entry.entry_id] = None
    await hass.config_entries.async_forward_entry_setups(entry, _PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload the Cloudflare AI conversation agent config entry."""
    hass.data[DOMAIN].pop(entry.entry_id, None)
    return await hass.config_entries.async_unload_platforms(entry, _PLATFORMS)
