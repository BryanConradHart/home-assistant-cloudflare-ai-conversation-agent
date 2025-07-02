"""Integration for Cloudflare AI as a Home Assistant conversation agent."""

from __future__ import annotations

import cloudflare
import openai

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_TOKEN, Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.httpx_client import get_async_client

from .const import CONF_ACCOUNT_ID, CONF_GATEWAY_ID, CONF_MODEL, DOMAIN

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
_PLATFORMS: list[Platform] = [Platform.CONVERSATION]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the Cloudflare AI conversation agent."""
    api_token = entry.data.get(CONF_API_TOKEN)
    account_id = entry.data.get(CONF_ACCOUNT_ID)
    gateway_id = entry.data.get(CONF_GATEWAY_ID)
    model = entry.data.get(CONF_MODEL)

    if api_token and account_id and gateway_id and model:
        client = await hass.async_add_executor_job(
            lambda: cloudflare.AsyncCloudflare(api_token=api_token)
        )
        try:
            gateway_url_result = await client.ai_gateway.urls.get(
                account_id=account_id, gateway_id=gateway_id, provider="workers-ai"
            )
            hass.data.setdefault(DOMAIN, {})[entry.entry_id] = openai.AsyncOpenAI(
                api_key=api_token,
                http_client=get_async_client(hass),
                base_url=gateway_url_result + "/v1",
            )
        except cloudflare.CloudflareError:
            hass.data.setdefault(DOMAIN, {})[entry.entry_id] = None
            return False
    else:
        hass.data.setdefault(DOMAIN, {})[entry.entry_id] = None
        return False
    await hass.config_entries.async_forward_entry_setups(entry, _PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload the Cloudflare AI conversation agent config entry."""
    hass.data[DOMAIN].pop(entry.entry_id, None)
    return await hass.config_entries.async_unload_platforms(entry, _PLATFORMS)
