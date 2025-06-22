"""Cloudflare AI conversation agent for Home Assistant using the Cloudflare python client."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

import cloudflare

from homeassistant.components import conversation
from homeassistant.components.conversation import ChatLog, ConversationResult
from homeassistant.components.conversation.models import ConversationInput
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.intent import IntentResponse

from .const import CONF_MODEL, DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = CloudflareAIConversationEntity(config_entry)
    async_add_entities([agent])


class CloudflareAIConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Cloudflare AI conversation agent using the Cloudflare python client."""

    _attr_has_entity_name = True
    _attr_supports_streaming = False

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the agent with the desired config."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_name = entry.title + " " + entry.data.get(CONF_MODEL, "model")
        if entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        await hass.config_entries.async_reload(entry.entry_id)

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """Relay the user phrase to Cloudflare AI and return the completion."""
        settings = {**self.entry.data, **self.entry.options}
        model = settings.get(CONF_MODEL, "@cf/meta/llama-3-8b-instruct")
        prompt = user_input.text
        account_id = settings.get("account_id")
        if not account_id:
            response = IntentResponse(language=user_input.language)
            response.async_set_speech("Cloudflare account ID is not configured")
            return ConversationResult(
                response=response,
                conversation_id=chat_log.conversation_id,
                continue_conversation=False,
            )

        # Get the stored client from hass.data
        cloudflare_client = self.hass.data[DOMAIN].get(self.entry.entry_id)
        if cloudflare_client is None:
            response = IntentResponse(language=user_input.language)
            response.async_set_speech("Cloudflare AI client is not configured")
            return ConversationResult(
                response=response,
                conversation_id=chat_log.conversation_id,
                continue_conversation=False,
            )

        try:
            # The cloudflare client is sync, so use run_in_executor for the call
            ai_response = await cloudflare_client.ai.run(
                model, prompt=prompt, account_id=account_id
            )
            if isinstance(ai_response, dict):
                response_text = ai_response.get("response") or str(ai_response)
            else:
                response_text = str(ai_response)
        except cloudflare.APIConnectionError:
            response_text = "The server could not be reached"
        except cloudflare.RateLimitError:
            response_text = "A 429 status code was received; we should back off a bit."
        except cloudflare.APIStatusError as e:
            response_text = str(e.status_code) + ":" + str(e.response)

        response = IntentResponse(language=user_input.language)
        response.async_set_speech(response_text)
        return ConversationResult(
            response=response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=False,
        )
