FROM rocm/dev-ubuntu-22.04:6.2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    python3.10 \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Install ROCm SDK and PyTorch with native gfx1151 (Strix) support from TheRock nightlies
RUN pip install --upgrade pip && \
    pip install 'numpy<2' && \
    pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries,devel]" && \
    pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchvision torchaudio

# Install Node.js (required for Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt-get update && apt-get install -y gh && \
    rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Install Python-based AI CLIs
RUN pip install scipy google-generativeai openai

# Install Gemini CLI (if available) and OpenAI Codex CLI
RUN npm install -g @google/generative-ai 2>/dev/null || true

# Set working directory
WORKDIR /workspace

# Default command
CMD ["bash"]
