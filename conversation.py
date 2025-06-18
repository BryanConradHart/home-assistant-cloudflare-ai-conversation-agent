"""Cloudflare AI conversation agent that echoes the prompt in upper or lower case."""

from __future__ import annotations

from typing import Literal

from homeassistant.components import assist_pipeline, conversation
from homeassistant.components.conversation import (
    AssistantContent,
    ChatLog,
    ConversationResult,
)
from homeassistant.components.conversation.models import ConversationInput
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.intent import IntentResponse

from .config_flow import CONF_ECHO_CASE, ECHO_CASE_LOWER
from .const import CONF_PROMPT, DOMAIN


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
    """Cloudflare AI conversation agent that echoes the prompt in upper or lower case."""

    _attr_has_entity_name = True
    _attr_supports_streaming = False

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the agent with the desired echo case."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_name = entry.title + " " + entry.data.get(CONF_ECHO_CASE)
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
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self._attr_unique_id, self.entity_id
        )
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
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """Call the API."""
        settings = {**self.entry.data, **self.entry.options}

        try:
            await chat_log.async_update_llm_data(
                DOMAIN,
                user_input,
                settings.get(CONF_LLM_HASS_API),
                settings.get(CONF_PROMPT),
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        if self.hass.data[DOMAIN][CONF_ECHO_CASE] == ECHO_CASE_LOWER:
            response_text = user_input.text.lower()
        else:
            response_text = user_input.text.upper()
        # Add the response to the chat log.
        # chat_log.async_add_assistant_content_without_tools(
        #     AssistantContent(
        #         agent_id=user_input.agent_id,
        #         content=response_text,
        #     )
        # )
        response = IntentResponse(language=user_input.language)
        response.async_set_speech(response_text)
        return ConversationResult(
            response=response,
            conversation_id="conversationId",
            continue_conversation=False,
        )
