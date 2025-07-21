"""Cloudflare AI conversation agent for Home Assistant using the Cloudflare python client."""

from __future__ import annotations

from collections.abc import AsyncIterable, Callable
import json
import logging
from typing import Any, Literal

from openai import AsyncOpenAI, Stream
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice
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
    AssistantContentDeltaDict,
    Content,
    SystemContent,
    ToolResultContent,
    UserContent,
)
from homeassistant.components.conversation.models import ConversationInput
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_TOKEN, CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
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

_MAX_TOOL_ITERATIONS = 10


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
    _attr_supports_streaming = True

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
    def client(self) -> AsyncOpenAI | None:
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
            name=tool.tool_name,
            arguments=json.dumps(tool.tool_args),
        )
        return ChatCompletionMessageToolCallParam(
            id=tool.id, function=tool_spec, type="function"
        )

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """Relay the user phrase to Cloudflare AI and return the completion, with history as context. Supports function calling."""
        try:
            await chat_log.async_update_llm_data(
                DOMAIN,
                user_input,
                self.llm_hass_api,
                self.instruction,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        # Function calling support
        tools: list[ChatCompletionToolParam] | None = None
        if chat_log.llm_api:
            tools = [
                self._format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        # Build the full chat history as a list of OpenAI messages
        messages: ChatCompletionMessageParam = [
            self._convert_content(chat) for chat in chat_log.content
        ]

        for _iteration in range(_MAX_TOOL_ITERATIONS):
            # Only include top_p, temperature, max_tokens if not None
            create_kwargs = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "tools": tools,
                "tool_choice": "auto",
                "extra_headers": {
                    "cf-aig-authorization": f"Bearer {self.api_token}"
                },  # extra headers required for Cloudflare AI Gateway
            }
            if self.top_p is not None:
                create_kwargs["top_p"] = self.top_p
            if self.temperature is not None:
                create_kwargs["temperature"] = self.temperature
            if self.max_tokens is not None:
                create_kwargs["max_tokens"] = self.max_tokens
            try:
                result: Stream[
                    ChatCompletionChunk
                ] = await self.client.chat.completions.create(**create_kwargs)
            except Exception as e:
                raise HomeAssistantError("Error talking to Cloudflare") from e

            # Handle function call responses if present
            messages.extend(
                [
                    self._convert_content(content)
                    async for content in chat_log.async_add_delta_content_stream(
                        user_input.agent_id,
                        self._transform_stream(result, user_input),
                    )
                ]
            )

            if not chat_log.unresponded_tool_results:
                break

        if chat_log.unresponded_tool_results:
            raise HomeAssistantError("Exceeded maxmimum tool uses for a single request")
        if (
            hasattr(chat_log.content[-1], "content")
            and not chat_log.content[-1].content
        ):
            raise HomeAssistantError("Assistant did not respond with text content")
        # If there are no unresponded tool results, we can return the response
        intent_response = IntentResponse(language=user_input.language)
        intent_response.async_set_speech(chat_log.content[-1].content)
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

    def _convert_content(self, chat: Content) -> ChatCompletionMessageParam:
        """Convert a Home Assistant conversation content object to an OpenAI message param using pattern matching."""
        match chat:
            case SystemContent(role="system", content=content):
                return ChatCompletionSystemMessageParam(content=content, role="system")
            case AssistantContent(
                role="assistant", content=content, tool_calls=tool_calls
            ):
                return ChatCompletionAssistantMessageParam(
                    content=content
                    or "",  # should be allowed to be null, but cloudflare seems to do extra erroneous validation
                    role="assistant",
                    tool_calls=[
                        self._format_tool_call_param(tool) for tool in tool_calls
                    ]
                    if tool_calls
                    else None,
                )
            case ToolResultContent(
                role="tool_result", tool_result=tool_result, tool_call_id=tool_call_id
            ):
                return ChatCompletionToolMessageParam(
                    content=json.dumps(tool_result),
                    role="tool",
                    tool_call_id=tool_call_id,
                )
            case UserContent(role="user", content=content):
                return ChatCompletionUserMessageParam(content=content, role="user")
            case _:
                raise TypeError(
                    f"Unknown Content type: {type(chat).__module__}.{type(chat).__qualname__}"
                )

    async def _transform_stream(
        self,
        result: Stream[ChatCompletionChunk],
        user_input: ConversationInput,
    ) -> AsyncIterable[AssistantContentDeltaDict]:
        new_msg: bool = True
        async for chunk in result:
            if len(chunk.choices) != 1:
                raise HomeAssistantError(
                    f"AI produced an unexpected number of choices: {len(chunk.choices)}"
                )
            choice: Choice = chunk.choices[0]
            if choice.delta.refusal:
                raise HomeAssistantError("Request refused: " + choice.delta.refusal)
            delta: AssistantContentDeltaDict = {}
            if new_msg:
                delta["role"] = "assistant"
                current_tool_calls = []
                new_msg = False
            if (content := choice.delta.content) and isinstance(content, str):
                delta["content"] = content
            if tool_calls := choice.delta.tool_calls:
                for tool_call in tool_calls:
                    # Grow the list if needed
                    if tool_call.index >= len(current_tool_calls):
                        current_tool_calls.extend(
                            [{}] * (tool_call.index - len(current_tool_calls) + 1)
                        )
                    # Only populate if empty
                    if not current_tool_calls[tool_call.index]:
                        current_tool_calls[tool_call.index] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }
                    # tool calls are build in parts, but HA doesn't support partial tool calls
                    if tool_call.id:
                        current_tool_calls[tool_call.index]["id"] += tool_call.id
                    if tool_call.function.name:
                        current_tool_calls[tool_call.index]["name"] += (
                            tool_call.function.name
                        )
                    if tool_call.function.arguments:
                        current_tool_calls[tool_call.index]["arguments"] += (
                            tool_call.function.arguments
                        )
            if choice.finish_reason:
                new_msg = True
                if current_tool_calls:
                    # If there are tool calls, add them to the delta
                    delta["tool_calls"] = [
                        llm.ToolInput(
                            id=tool_call["id"],
                            tool_name=tool_call["name"],
                            tool_args=json.loads(tool_call["arguments"]),
                        )
                        for tool_call in current_tool_calls
                    ]
                if choice.finish_reason not in {"stop", "tool_calls"}:
                    raise HomeAssistantError(
                        f"Unexpected finish reason: {choice.finish_reason}"
                    )
            if any(key in delta for key in ("role", "content", "tool_calls")):
                # Only yield if there is content to yield
                yield delta
