# Use Astral's official uv Docker image as base
FROM ghcr.io/astral-sh/uv:debian-slim AS base

# Install Node.js, npm, and certificates (vital for SSL to OpenRouter)
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs \
    npm \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

# Create a non-root user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Pre-configure Claude Code to skip onboarding and login
# This prevents the "freeze" by telling the CLI you are already set up.
RUN mkdir -p /home/appuser/.claude && \
    echo '{"hasCompletedOnboarding": true}' > /home/appuser/.claude.json && \
    chown -R appuser:appuser /home/appuser

# Switch to non-root user
USER appuser

# Default command: Start a shell
CMD ["/bin/sh"]