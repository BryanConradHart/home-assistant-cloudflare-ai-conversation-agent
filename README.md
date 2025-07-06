# Home Assistant Cloudflare AI Conversation Agent

Create a [Home Assistant Conversation Agent](https://www.home-assistant.io/voice_control/) powered by [Cloudflare hosted LLM models](https://developers.cloudflare.com/workers-ai/models/), enabling advanced conversation and automation features.

## Key Features
- **Cost:**
  - Cloudflare offers a [free usage tier](https://developers.cloudflare.com/workers-ai/platform/pricing/) for AI models.
  - Different models consume usage at different rates. [See usage rates here](https://developers.cloudflare.com/workers-ai/platform/usage/).
- **Cloudflare-Hosted Models:**
  - Use a wide variety of AI models directly from [Cloudflare's platform](https://developers.cloudflare.com/workers-ai/models/).
- **AI Gateway Integration:**
  - All requests are routed through your [Cloudflare AI Gateway](https://developers.cloudflare.com/ai-gateway/), allowing you to track usage, set custom restrictions, and enforce limits on your AI consumption.
- **Ideal for Cloudflared Users:**
  - If you already use [Cloudflare tunnels](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/) for [Home Assistant](https://github.com/brenner-tobias/addon-cloudflared), then you already have a Cloudflare account anyways.

## Setup Instructions

1. Create a Cloudflare [API Token](https://developers.cloudflare.com/fundamentals/api/get-started/create-token/) for secure API access.
   - **Required permissions:**
     - `Account Settings:Read` — to list the available accounts (Optional: If you're using a User Token instead of an Account Token)
     - `Workers AI:Read` — to list the available LLM models, during setup
     - `AI Gateway:Edit` — to create a new AI gateway during setup (optional: if you already have one)
     - `AI Gateway:Read` — to get the list of existing gateways, and get the gateway ID
     - `AI Gateway:Run` — use the AI, to get completions.
2. Configure the integration in Home Assistant, providing your token and selecting your desired [Cloudflare AI model](https://developers.cloudflare.com/workers-ai/models/).
3. Optionally, adjust advanced options such as prompt, temperature, top-p, and max tokens. [See usage rates for your chosen model](https://developers.cloudflare.com/workers-ai/platform/usage/).
4. Go to `Settings > Voice assistants`, and [configure an assistant](https://www.home-assistant.io/voice_control/voice_remote_local_assistant/) to use the model you just selected as its Conversation Agent.

---

**Note:** Cloudflare's free tier and model availability may change. Always check the official documentation for the latest details.
