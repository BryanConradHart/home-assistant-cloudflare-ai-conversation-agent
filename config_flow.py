"""Config flow for the cloudflare ai conversation assistant integration."""

from __future__ import annotations

import voluptuous as vol

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult

from .const import DOMAIN

CONF_ECHO_CASE = "echo_case"
ECHO_CASE_UPPER = "uppercase"
ECHO_CASE_LOWER = "lowercase"

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_ECHO_CASE, default=ECHO_CASE_UPPER): vol.In(
            [ECHO_CASE_UPPER, ECHO_CASE_LOWER]
        ),
    }
)


class ConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for cloudflare ai conversation assistant."""

    async def async_step_user(
        self, user_input: dict[str, str] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            return self.async_create_entry(
                title="Cloudflare AI Conversation Agent", data=user_input
            )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )
