"""Cloudflare AI conversation agent for Home Assistant using the Cloudflare python client."""

from __future__ import annotations

from collections.abc import Callable
import json
import logging
from typing import Any, Literal

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
    Function as ToolParamFunction,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params.function_definition import FunctionDefinition
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.components.conversation import ChatLog, ConversationResult
from homeassistant.components.conversation.chat_log import (
    AssistantContent,
    SystemContent,
    ToolResultContent,
    UserContent,
)
from homeassistant.components.conversation.models import ConversationInput
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_TOKEN, CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.intent import IntentResponse

from .const import (
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
)

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
        self._entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_name = entry.title
        if entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    @property
    def llm_hass_api(self) -> bool:
        """Return whether to enable Home Assistant API tool."""
        return self._entry.options.get(CONF_LLM_HASS_API)

    @property
    def instruction(self) -> str | None:
        """Return the instruction prompt."""
        return self._entry.options.get(CONF_PROMPT)

    @property
    def max_tokens(self) -> int | None:
        """Return the max tokens setting."""
        return self._entry.options.get(CONF_MAX_TOKENS) or None

    @property
    def top_p(self) -> float | None:
        """Return the top-p setting."""
        return self._entry.options.get(CONF_TOP_P) or None

    @property
    def temperature(self) -> float | None:
        """Return the temperature setting."""
        return self._entry.options.get(CONF_TEMPERATURE) or None

    @property
    def api_token(self) -> str | None:
        """Return the API token from config entry data."""
        return self._entry.data.get(CONF_API_TOKEN)

    @property
    def model(self) -> str | None:
        """Return the model from config entry data."""
        return self._entry.data.get(CONF_MODEL)

    @property
    def client(self) -> ChatCompletion | None:
        """Return the OpenAI client for this config entry."""
        return self.hass.data[DOMAIN].get(self._entry.entry_id)

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self._entry, self)
        self._entry.async_on_unload(
            self._entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self._entry)
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

    def _format_tool(
        self, tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
    ) -> ChatCompletionToolParam:
        """Format tool specification."""
        tool_spec = FunctionDefinition(
            name=tool.name,
            description=tool.description,
            parameters=convert(tool.parameters, custom_serializer=custom_serializer),
        )
        return ChatCompletionToolParam(function=tool_spec, type="function")

    def _format_tool_call_param(
        self, tool: llm.ToolInput
    ) -> ChatCompletionMessageToolCallParam:
        """Format tool call specification."""
        tool_spec = ToolParamFunction(
            name=tool.name,
            parameters=json.dumps(tool.parameters),
        )
        return ChatCompletionMessageToolCallParam(function=tool_spec, type="function")

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """Relay the user phrase to Cloudflare AI and return the completion, with history as context. Supports function calling."""
        if not self.api_token:
            response = IntentResponse(language=user_input.language)
            response.async_set_speech("Cloudflare API token is not configured")
            return ConversationResult(
                response=response,
                conversation_id=chat_log.conversation_id,
                continue_conversation=False,
            )

        if not self.model:
            response = IntentResponse(language=user_input.language)
            response.async_set_speech("Cloudflare model is not configured")
            return ConversationResult(
                response=response,
                conversation_id=chat_log.conversation_id,
                continue_conversation=False,
            )

        if self.client is None:
            response = IntentResponse(language=user_input.language)
            response.async_set_speech("Cloudflare AI client is not configured")
            return ConversationResult(
                response=response,
                conversation_id=chat_log.conversation_id,
                continue_conversation=False,
            )
        try:
            await chat_log.async_update_llm_data(
                DOMAIN,
                user_input,
                self.llm_hass_api,
                self.instruction,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        # Build the full chat history as a list of OpenAI messages
        messages: ChatCompletionMessageParam = [
            ChatCompletionSystemMessageParam(content=chat.content, role="system")
            if isinstance(chat, SystemContent)
            else ChatCompletionAssistantMessageParam(
                content=chat.content,
                role="assistant",
                tool_calls=[
                    self._format_tool_call_param(tool) for tool in chat.tool_calls
                ]
                if chat.tool_calls
                else None,
            )
            if isinstance(chat, AssistantContent)
            else ChatCompletionToolMessageParam(
                content=chat.content, role="tool", tool_call_id=chat.tool_call_id
            )
            if isinstance(chat, ToolResultContent)
            else ChatCompletionUserMessageParam(content=chat.content, role="user")
            if isinstance(chat, UserContent)
            else None
            for chat in chat_log.content
            if isinstance(
                chat, (SystemContent, UserContent, AssistantContent, ToolResultContent)
            )
        ]

        # Function calling support
        tools: list[ChatCompletionToolParam] | None = None
        if chat_log.llm_api:
            tools = [
                self._format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        # Only include top_p, temperature, max_tokens if not None
        create_kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "tools": tools,
            "tool_choice": "auto",
            "extra_headers": {"cf-aig-authorization": f"Bearer {self.api_token}"},
        }
        if self.top_p is not None:
            create_kwargs["top_p"] = self.top_p
        if self.temperature is not None:
            create_kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            create_kwargs["max_tokens"] = self.max_tokens

        try:
            # extra headers required for Cloudflare AI Gateway
            completion: ChatCompletion = await self.client.chat.completions.create(
                **create_kwargs
            )
            # Handle function call responses if present
            choice: Choice = completion.choices[0]
            if choice.message and choice.message.tool_calls:
                # Add function call as tool call to chat log
                chat_log.async_add_assistant_content(
                    AssistantContent(
                        content=None,
                        agent_id=user_input.agent_id,
                        tool_calls=[
                            llm.ToolInput(
                                id=tool_call.id,
                                tool_name=tool_call.function.name,
                                tool_args=json.loads(tool_call.function.arguments),
                            )
                            for tool_call in choice.message.tool_calls
                            if tool_call.id
                            and tool_call.type == "function"
                            and tool_call.function
                        ],
                    )
                )
                response_text = choice.message.tool_calls[0].function.name
            else:
                response_text = choice.message.content
                chat_log.async_add_assistant_content_without_tools(
                    AssistantContent(
                        content=response_text,
                        agent_id=user_input.agent_id,
                    )
                )
        except Exception as e:
            response_text = f"Error communicating with Cloudflare AI: {e}"

        response = IntentResponse(language=user_input.language)
        response.async_set_speech(response_text)
        return ConversationResult(
            response=response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=False,
        )
