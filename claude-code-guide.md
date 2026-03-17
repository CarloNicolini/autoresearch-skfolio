## `autoresearch-skfolio` with Claude Code and free OpenRouter credits

This guide shows you how to "trick" Claude Code into using **OpenRouter** as its backend, unlocking a world of alternative models, including the highly intelligent (and often free tier) `nemotron-3-super-120b-a12b:free`.

---

### Get Your OpenRouter API Key

If you don't already have one, creating an OpenRouter API key is quick.

1.  **Visit OpenRouter:** Go to [https://openrouter.ai/](https://openrouter.ai/).
2.  **Sign In/Up:** You can sign in using your Google, GitHub, or Discord account.
3.  **Navigate to API Keys:** Once logged in, go to your dashboard and find the "Keys" or "API Keys" section. This is usually under your profile dropdown or a dedicated "Settings" page.
4.  **Create a New Key:** Click "Create New Key." Give it a descriptive name (e.g., "Claude Code CLI").
5.  **Copy Your Key:** Your API key will be displayed. **Copy it immediately** as you won't be able to see it again. It will look something like `sk-or-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`.

Now you have two ways to proceed. Local installation or execution via Docker. If you have Docker available I strongly suggest using that solution to sandbox Claude Code.

---

### Install Claude Code CLI

If you haven't already, install the Claude Code CLI. This requires Node.js (v18 or higher) and npm/yarn, on MacOs and Linux it's easy to do.

1.  **Install Node.js:** If you don't have Node.js, download it from [nodejs.org](https://nodejs.org/) or use a version manager like `nvm`.
2.  **Install Claude Code:**
    ```bash
    npm install -g @anthropic-ai/claude-cli
    # or if you prefer yarn
    yarn global add @anthropic-ai/claude-cli
    ```

### Point Claude Code to OpenRouter

This is where we tell Claude Code to route its requests through OpenRouter. We'll use environment variables.

Add the following lines to your shell's configuration file (e.g., `~/.zshrc`, `~/.bashrc`, `~/.config/fish/config.fish`).

```bash
# --- OpenRouter Configuration for Claude Code ---
export OPENROUTER_API_KEY="sk-or-YOUR_OPENROUTER_KEY_HERE"
export ANTHROPIC_BASE_URL="[https://openrouter.ai/api](https://openrouter.ai/api)" # Important: Use /api, NOT /api/v1
export ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY"
export ANTHROPIC_API_KEY="" # Ensure this is empty to avoid conflicts
``` 


then run

```bash
source ~/.zshrc
```

and the environment variables will be ready.

The last step is to edit the `~/.claude/settings.json` and add the following field

```json
{
    "hasCompletedOnboarding": true
}
```


This will let you avoid the initial onboarding that requires you to sign in to Claude either with Anthropic sdk or via APIs.

Now you can run Claude code using the above mentioned free model.

```bash
claude --model nvidia/nemotron-3-super-120b-a12b:free
```

Happy autoresearch!
