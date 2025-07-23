"""Integration for Cloudflare AI as a Home Assistant conversation agent."""

from __future__ import annotations

import cloudflare
import openai

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_TOKEN, Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.httpx_client import get_async_client

from .const import CONF_ACCOUNT_ID, CONF_GATEWAY_ID, DOMAIN

type CloudflareEntry = ConfigEntry[cloudflare.AsyncCloudflare]

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
_PLATFORMS: list[Platform] = [Platform.CONVERSATION]  # TODO add Platform.AI_TASK


async def async_setup_entry(hass: HomeAssistant, entry: CloudflareEntry) -> bool:
    """Set up the Cloudflare AI conversation agent."""
    api_token = entry.data.get(CONF_API_TOKEN)
    if api_token:
        entry.runtime_data = await hass.async_add_executor_job(
            lambda: cloudflare.AsyncCloudflare(api_token=api_token)
        )
        for subentry in entry.subentries.values():
            if subentry.subentry_type == "conversation":
                account_id = subentry.data[CONF_ACCOUNT_ID]
                gateway_id = subentry.data[CONF_GATEWAY_ID]
                try:
                    gateway_url_result = await entry.runtime_data.ai_gateway.urls.get(
                        account_id=account_id,
                        gateway_id=gateway_id,
                        provider="workers-ai",
                    )
                except cloudflare.CloudflareError:
                    return False
                else:
                    hass.data.setdefault(DOMAIN, {})[
                        entry.entry_id + subentry.subentry_id
                    ] = openai.AsyncOpenAI(
                        api_key=api_token,
                        http_client=get_async_client(hass),
                        base_url=gateway_url_result + "/v1",
                    )
    else:
        return False
    await hass.config_entries.async_forward_entry_setups(entry, _PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(async_update_options))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload the Cloudflare AI conversation agent config entry."""
    return await hass.config_entries.async_unload_platforms(entry, _PLATFORMS)


async def async_update_options(hass: HomeAssistant, entry: CloudflareEntry) -> None:
    """Update options."""
    await hass.config_entries.async_reload(entry.entry_id)
