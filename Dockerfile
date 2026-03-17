# Use Astral's official uv Docker image
FROM ghcr.io/astral-sh/uv:debian-slim AS base

# Install Git and basic dependencies as root
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs \
    npm \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

# Create a non-root user
RUN useradd -m -u 1000 appuser

# --- CRITICAL STEP ---
# Explicitly set the PATH to include the standard binary locations.
# This ensures appuser always knows where 'git' (usually in /usr/bin) is.
ENV PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
ENV DISABLE_TELEMETRY=1

WORKDIR /app

# Pre-configure Claude Code
RUN mkdir -p /home/appuser/.claude && \
    echo '{"hasCompletedOnboarding": true}' > /home/appuser/.claude.json && \
    chown -R appuser:appuser /home/appuser

# Switch to non-root user
USER appuser

# Final sanity check: if this fails, the build stops.
RUN which git && git --version

CMD ["/bin/sh"]