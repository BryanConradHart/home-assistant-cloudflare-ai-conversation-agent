"""Cloudflare AI conversation agent for Home Assistant using the Cloudflare python client."""

from __future__ import annotations

import logging
from typing import Literal

from homeassistant.components import conversation
from homeassistant.components.conversation import ChatLog, ConversationResult
from homeassistant.components.conversation.models import ConversationInput
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.intent import IntentResponse

from .const import CONF_API_TOKEN, CONF_MODEL, DOMAIN

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

    def _format_history(self, chat_log: ChatLog) -> str:
        """Format the conversation history for Cloudflare AI context."""

        # Follows OpenAI/Anthropic/Ollama style: role: content\n\n
        def format_message(msg):
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                # If there are tool calls, include them
                tool_str = "\n".join(
                    f"[tool_call] {tool_call.tool_name}: {tool_call.tool_args}"
                    for tool_call in msg.tool_calls
                )
                return f"{role}: {content}\n{tool_str}"
            return f"{role}: {content}"

        # Exclude system prompt if present (Cloudflare may not need it, but you can adjust)
        history = [format_message(msg) for msg in chat_log.content]
        return "\n\n".join(history)

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """Relay the user phrase to Cloudflare AI and return the completion, with history as context."""
        settings = {**self.entry.data, **self.entry.options}
        model = settings.get(CONF_MODEL, "@cf/meta/llama-3-8b-instruct")
        if not model:
            response = IntentResponse(language=user_input.language)
            response.async_set_speech("Cloudflare model is not configured")
            return ConversationResult(
                response=response,
                conversation_id=chat_log.conversation_id,
                continue_conversation=False,
            )

        client = self.hass.data[DOMAIN].get(self.entry.entry_id)
        if client is None:
            response = IntentResponse(language=user_input.language)
            response.async_set_speech("Cloudflare AI client is not configured")
            return ConversationResult(
                response=response,
                conversation_id=chat_log.conversation_id,
                continue_conversation=False,
            )

        # Build the full prompt with history
        prompt = self._format_history(chat_log)

        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                extra_headers={
                    "cf-aig-authorization": f"Bearer {settings.get(CONF_API_TOKEN)}"
                },
            )
            response_text = completion.choices[0].message.content
        except Exception as e:
            response_text = f"Error communicating with Cloudflare AI: {e}"

        response = IntentResponse(language=user_input.language)
        response.async_set_speech(response_text)
        return ConversationResult(
            response=response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=False,
        )
