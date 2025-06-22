"""Config flow for the cloudflare ai conversation assistant integration."""

from __future__ import annotations

import asyncio

import cloudflare
from cloudflare import AsyncCloudflare
import voluptuous as vol

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult

from .const import CONF_ACCOUNT_ID, CONF_API_TOKEN, CONF_MODEL, DOMAIN

STEP_USER_DATA_SCHEMA = vol.Schema({vol.Required(CONF_API_TOKEN): str})

STEP_ACCOUNT_DATA_SCHEMA = vol.Schema({vol.Required(CONF_ACCOUNT_ID): str})


class ConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for cloudflare ai conversation assistant."""

    def __init__(self) -> None:
        """Initialize the config flow and create the AsyncCloudflare client once."""
        self.client = None

    async def async_step_user(
        self, user_input: dict[str, str] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step (API token)."""
        errors: dict[str, str] = {}
        description_placeholders: dict[str, str] = {}
        if user_input is not None:
            api_token = user_input[CONF_API_TOKEN]
            try:
                # get a warning if we try call the constructor synchronously
                self.client = await self.hass.async_add_executor_job(
                    lambda: AsyncCloudflare(api_token=api_token)
                )
                self.context[CONF_API_TOKEN] = user_input[CONF_API_TOKEN]
                return await self.async_step_account()
            except OSError:
                errors[CONF_API_TOKEN] = "cannot_connect"
        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
            description_placeholders=description_placeholders,
        )

    async def async_step_account(
        self, user_input: dict[str, str] | None = None
    ) -> ConfigFlowResult:
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
            errors[CONF_API_TOKEN] = "cannot_connect"
        except cloudflare.RateLimitError:
            errors[CONF_API_TOKEN] = "rate_limit_exceeded"
        except cloudflare.PermissionDeniedError:
            errors[CONF_API_TOKEN] = "insufficient_permission_account"
        except cloudflare.APIStatusError as e:
            errors[CONF_API_TOKEN] = "unknown_api_error"
            description_placeholders["code"] = e.status_code
            description_placeholders["message"] = e.response
        if not account_choices:
            errors[CONF_ACCOUNT_ID] = "no_accounts"
        if user_input is not None and not errors:
            self.context[CONF_ACCOUNT_ID] = user_input[CONF_ACCOUNT_ID]
            return await self.async_step_model()
        return self.async_show_form(
            step_id="account",
            data_schema=vol.Schema(
                {vol.Required(CONF_ACCOUNT_ID): vol.In(account_choices)}
            ),
            errors=errors,
        )

    async def async_step_model(
        self, user_input: dict[str, str] | None = None
    ) -> ConfigFlowResult:
        """Let the user pick a model from the Cloudflare API."""
        errors: dict[str, str] = {}
        description_placeholders: dict[str, str] = {}
        api_token = self.context[CONF_API_TOKEN]
        account_id = self.context[CONF_ACCOUNT_ID]
        try:
            model_choices = {
                m.get("name", m["id"])
                async for m in self.client.ai.models.list(account_id=account_id)
                if "id" in m
            }
            model_choices = sorted(model_choices, key=lambda m: m.lower())
            # Set default model if available
            default_model = (
                "@cf/meta/llama-3.1-8b-instruct-fp8"
                if "@cf/meta/llama-3.1-8b-instruct-fp8" in model_choices
                else None
            )
        except cloudflare.APIConnectionError:
            errors[CONF_API_TOKEN] = "cannot_connect"
        except cloudflare.RateLimitError:
            errors[CONF_API_TOKEN] = "rate_limit_exceeded"
        except cloudflare.PermissionDeniedError:
            errors[CONF_API_TOKEN] = "insufficient_permission_models"
        except cloudflare.APIStatusError as e:
            errors[CONF_API_TOKEN] = "unknown_api_error"
            description_placeholders["code"] = e.status_code
            description_placeholders["message"] = e.response
        if not model_choices:
            errors[CONF_MODEL] = "no_models"
        if user_input is not None and not errors:
            return self.async_create_entry(
                title="Cloudflare AI Conversation Agent",
                data={
                    CONF_API_TOKEN: api_token,
                    CONF_ACCOUNT_ID: account_id,
                    CONF_MODEL: user_input[CONF_MODEL],
                },
            )
        return self.async_show_form(
            step_id="model",
            data_schema=vol.Schema(
                {vol.Required(CONF_MODEL, default=default_model): vol.In(model_choices)}
            ),
            errors=errors,
        )
