FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

# Enable AOTriton for Strix Halo (gfx1151) support
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Node.js (required for Claude Code)
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
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
