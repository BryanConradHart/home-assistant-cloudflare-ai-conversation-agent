"""Config flow for the cloudflare ai conversation assistant integration."""

from __future__ import annotations

from typing import Any

import cloudflare
from cloudflare import AsyncCloudflare
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_TOKEN, CONF_LLM_HASS_API
from homeassistant.core import callback
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)

from .const import (
    CONF_ACCOUNT_ID,
    CONF_GATEWAY_ID,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
)

# Constant for the default Cloudflare gateway name
HA_CLOUDFLARE_GATEWAY_NAME: str = "ha-cloudflare-conversation-agent"
DEFAULT_MODEL: str = "@hf/nousresearch/hermes-2-pro-mistral-7b"

STEP_USER_DATA_SCHEMA: vol.Schema = vol.Schema({vol.Required(CONF_API_TOKEN): str})


class CloudflareAiConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for cloudflare ai conversation assistant."""

    async def async_step_user(
        self, user_input: dict[str, str] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step (API token)."""
        errors: dict[str, str] = {}
        description_placeholders: dict[str, str] = {}
        if user_input is not None:
            self._async_abort_entries_match(user_input)
            try:
                client: AsyncCloudflare = await self.hass.async_add_executor_job(
                    lambda: AsyncCloudflare(api_token=user_input[CONF_API_TOKEN])
                )
                await client.accounts.list()  # Test the API connection
            except OSError:
                errors[CONF_API_TOKEN] = "cannot_connect"
            else:
                self.context[CONF_API_TOKEN] = user_input[CONF_API_TOKEN]
                return self.async_create_entry(
                    title="Cloudflare AI",
                    data={CONF_API_TOKEN: user_input[CONF_API_TOKEN]},
                )  # TODO fetch and pass in subentry data

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
            description_placeholders=description_placeholders,
        )

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {"conversation": ConversationSubentryFlowHandler}


# TODO figure out why none of the steps labels are displaying
class ConversationSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing conversation subentries."""

    def __init__(self) -> None:
        """Initialize options flow."""
        self._is_new: bool = True

    @property
    def client(self) -> AsyncCloudflare:
        """Return the Cloudflare client."""
        return self._get_entry().runtime_data

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """New subentry."""
        # abort if entry is not loaded
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")
        self._is_new = True
        return await self.async_step_account()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure existing subentry."""
        if self.source == "user":
            return await self.async_step_user(user_input)
        # abort if entry is not loaded
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")
        self._is_new = False
        self.load_existing_values()
        return await self.async_step_model()

    def load_existing_values(self) -> None:
        """Load existing subentry values into the context."""
        self.context.setdefault(
            CONF_ACCOUNT_ID, self._get_reconfigure_subentry().data.get(CONF_ACCOUNT_ID)
        )
        self.context.setdefault(
            CONF_GATEWAY_ID, self._get_reconfigure_subentry().data.get(CONF_GATEWAY_ID)
        )
        self.context.setdefault(
            CONF_MODEL, self._get_reconfigure_subentry().data.get(CONF_MODEL)
        )
        self.context.setdefault(
            CONF_PROMPT, self._get_reconfigure_subentry().data.get(CONF_PROMPT)
        )
        self.context.setdefault(
            CONF_LLM_HASS_API,
            self._get_reconfigure_subentry().data.get(CONF_LLM_HASS_API),
        )
        self.context.setdefault(
            CONF_MAX_TOKENS, self._get_reconfigure_subentry().data.get(CONF_MAX_TOKENS)
        )
        self.context.setdefault(
            CONF_TOP_P, self._get_reconfigure_subentry().data.get(CONF_TOP_P)
        )
        self.context.setdefault(
            CONF_TEMPERATURE,
            self._get_reconfigure_subentry().data.get(CONF_TEMPERATURE),
        )

    async def async_step_account(
        self, user_input: dict[str, str] | None = None
    ) -> SubentryFlowResult:
        """Let the user pick a Cloudflare account."""
        errors: dict[str, str] = {}
        description_placeholders: dict[str, str] = {}
        try:
            # Workaround: Only use the first page due to bug in auto-pagination: https://github.com/cloudflare/cloudflare-python/issues/2584
            # account_choices = {
            #     account.id: account.name
            #     async for account in self.client.accounts.list()
            #     # async for account in self.client.accounts.list()
            # }
            first_page = await self.client.accounts.list()
            account_choices = {
                account.id: account.name for account in first_page.result
            }
            account_choices = dict(
                sorted(account_choices.items(), key=lambda account: account[1].lower())
            )
        except cloudflare.APIConnectionError:
            errors[CONF_ACCOUNT_ID] = "cannot_connect"
        except cloudflare.RateLimitError:
            errors[CONF_ACCOUNT_ID] = "rate_limit_exceeded"
        except cloudflare.PermissionDeniedError:
            errors[CONF_ACCOUNT_ID] = "insufficient_permission_account"
        except cloudflare.APIStatusError as e:
            errors[CONF_ACCOUNT_ID] = "unknown_api_error"
            description_placeholders["code"] = e.status_code
            description_placeholders["message"] = e.response
        if not account_choices:
            errors[CONF_ACCOUNT_ID] = "no_accounts"
        if user_input is not None and not errors:
            self.context[CONF_ACCOUNT_ID] = user_input[CONF_ACCOUNT_ID]
            return await self.async_step_gateway()
        return self.async_show_form(
            step_id="account",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_ACCOUNT_ID,
                        default=self.context.get(CONF_ACCOUNT_ID),
                    ): vol.In(account_choices)
                }
            ),
            errors=errors,
        )

    async def async_step_gateway(
        self, user_input: dict[str, str] | None = None
    ) -> SubentryFlowResult:
        """Let the user pick or create an AI Gateway."""
        errors: dict[str, str] = {}
        description_placeholders: dict[str, str] = {}
        account_id = self.context[CONF_ACCOUNT_ID]
        gateway_choices = {}
        try:
            gateway_choices = {
                g.id async for g in self.client.ai_gateway.list(account_id=account_id)
            }
        except cloudflare.APIConnectionError:
            errors[CONF_GATEWAY_ID] = "cannot_connect"
        except cloudflare.RateLimitError:
            errors[CONF_GATEWAY_ID] = "rate_limit_exceeded"
        except cloudflare.PermissionDeniedError:
            errors[CONF_GATEWAY_ID] = "insufficient_permission_gateway_read"
        except cloudflare.APIStatusError as e:
            errors[CONF_GATEWAY_ID] = "unknown_api_error"
            description_placeholders["code"] = e.status_code
            description_placeholders["message"] = e.response
        # Add 'Create new gateway' option at the top, then sort gateways alphabetically
        CREATE_NEW = "Create new gateway..."
        gateway_choices = [CREATE_NEW, *sorted(gateway_choices, key=str.lower)]
        # Set default to HA_CLOUDFLARE_GATEWAY_NAME if it exists, else None
        if user_input is not None and not errors:
            if user_input[CONF_GATEWAY_ID] == CREATE_NEW:
                return await self.async_step_new_gateway()
            self.context[CONF_GATEWAY_ID] = user_input[CONF_GATEWAY_ID]
            return await self.async_step_model()
        return self.async_show_form(
            step_id="gateway",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_GATEWAY_ID,
                        default=self.context.get(
                            CONF_GATEWAY_ID, HA_CLOUDFLARE_GATEWAY_NAME
                        ),
                    ): vol.In(gateway_choices)
                }
            ),
            errors=errors,
        )

    async def async_step_new_gateway(
        self, user_input: dict[str, str] | None = None
    ) -> SubentryFlowResult:
        """Step to create a new AI Gateway."""
        errors: dict[str, str] = {}
        description_placeholders: dict[str, str] = {}
        if user_input is not None:
            gateway_id = user_input[CONF_GATEWAY_ID]
            try:
                # Create the gateway with authentication enabled
                await self.client.ai_gateway.create(
                    account_id=self.context[CONF_ACCOUNT_ID],
                    id=gateway_id,
                    authentication=True,
                    cache_invalidate_on_update=True,
                    cache_ttl=0,
                    collect_logs=True,
                    rate_limiting_interval=0,
                    rate_limiting_limit=0,
                    rate_limiting_technique="fixed",
                    log_management_strategy="DELETE_OLDEST",
                )
                self.context[CONF_GATEWAY_ID] = gateway_id
                return await self.async_step_model()
            except cloudflare.APIConnectionError:
                errors[CONF_GATEWAY_ID] = "cannot_connect"
            except cloudflare.RateLimitError:
                errors[CONF_GATEWAY_ID] = "rate_limit_exceeded"
            except cloudflare.PermissionDeniedError:
                errors[CONF_GATEWAY_ID] = "insufficient_permission_gateway_write"
            except cloudflare.APIStatusError as e:
                errors[CONF_GATEWAY_ID] = "unknown_api_error"
                description_placeholders["code"] = e.status_code
                description_placeholders["message"] = e.response
        return self.async_show_form(
            step_id="new_gateway",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_GATEWAY_ID,
                        default=HA_CLOUDFLARE_GATEWAY_NAME,
                    ): str
                }
            ),
            errors=errors,
        )

    async def async_step_model(
        self, user_input: dict[str, str] | None = None
    ) -> SubentryFlowResult:
        """Let the user pick a model from the Cloudflare API."""
        errors: dict[str, str] = {}
        description_placeholders: dict[str, str] = {}
        model_choices: list[str] = []
        try:
            model_choices = [
                SelectOptionDict(value=m.get("name"), label=m.get("name"))
                async for m in self.client.ai.models.list(
                    account_id=self.context[CONF_ACCOUNT_ID], task="Text Generation"
                )
                if "id" in m and self.supports_function_calling(m)
            ]
            model_choices = sorted(model_choices, key=lambda m: m["label"].lower())
        except cloudflare.APIConnectionError:
            errors[CONF_MODEL] = "cannot_connect"
        except cloudflare.RateLimitError:
            errors[CONF_MODEL] = "rate_limit_exceeded"
        except cloudflare.PermissionDeniedError:
            errors[CONF_MODEL] = "insufficient_permission_models"
        except cloudflare.APIStatusError as e:
            errors[CONF_MODEL] = "unknown_api_error"
            description_placeholders["code"] = e.status_code
            description_placeholders["message"] = e.response
        if not model_choices:
            errors[CONF_MODEL] = "no_models"
        if user_input is not None and not errors:
            self.context[CONF_MODEL] = user_input[CONF_MODEL]
            return await self.async_step_advanced_options()
        return self.async_show_form(
            step_id="model",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_MODEL,
                        default=self.context.get(CONF_MODEL, DEFAULT_MODEL),
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=model_choices,
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    )
                }
            ),
            errors=errors,
        )

    def supports_function_calling(self, model: dict) -> bool:
        """Return True if the model supports function calling."""
        for prop in model.get("properties", []):
            if (
                prop.get("property_id") == "function_calling"
                and prop.get("value") == "true"
            ):
                return True
        return False

    async def async_step_advanced_options(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Show the advanced options form for updating configuration."""
        errors: dict[str, str] = {}
        llm_api_options = [
            SelectOptionDict(value=api.id, label=api.name)
            for api in llm.async_get_apis(self.hass)
        ]
        options_schema = vol.Schema(
            {
                vol.Required(
                    CONF_PROMPT,
                    default=self.context.get(
                        CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                    ),
                ): TemplateSelector(),
                vol.Required(
                    CONF_LLM_HASS_API,
                    default=self.context.get(CONF_LLM_HASS_API, [llm.LLM_API_ASSIST]),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=llm_api_options,
                        multiple=True,
                    )
                ),
            }
        )
        # schema doesnt like default None, unions types arent supported
        # TODO use add_suggested_values_to_schema
        # TODO check if advanced options are visible
        if self.context.get(CONF_MAX_TOKENS):
            options_schema = options_schema.extend(
                {
                    vol.Optional(
                        CONF_MAX_TOKENS,
                        default=self.context.get(CONF_MAX_TOKENS),
                    ): int
                }
            )
        else:
            options_schema = options_schema.extend(
                {
                    vol.Optional(
                        CONF_MAX_TOKENS,
                    ): int
                }
            )
        if self.context.get(CONF_TOP_P):
            options_schema = options_schema.extend(
                {
                    vol.Optional(
                        CONF_TOP_P,
                        default=self.context.get(CONF_TOP_P),
                    ): int
                }
            )
        else:
            options_schema = options_schema.extend(
                {
                    vol.Optional(
                        CONF_TOP_P,
                    ): int
                }
            )
        if self.context.get(CONF_TEMPERATURE):
            options_schema = options_schema.extend(
                {
                    vol.Optional(
                        CONF_TEMPERATURE,
                        default=self.context.get(CONF_TEMPERATURE),
                    ): int
                }
            )
        else:
            options_schema = options_schema.extend(
                {
                    vol.Optional(
                        CONF_TEMPERATURE,
                    ): int
                }
            )
        if user_input is not None:
            self.context.update(user_input)
            if self._is_new:
                return self.async_create_entry(
                    title=f"Cloudflare {self.context[CONF_MODEL]}", data=self.context
                )
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                title=f"Cloudflare {self.context[CONF_MODEL]}",
                data=self.context,
            )
        return self.async_show_form(
            step_id="advanced_options",
            data_schema=options_schema,
            errors=errors,
            description_placeholders={
                "prompt": "Instruction Prompt",
                "llm_haas_api": "LLM Home Assistant API",
                "max_tokens": "Max Tokens",
                "top_p": "Top-P",
                "temperature": "Temperature",
            },
            last_step=True,
        )
