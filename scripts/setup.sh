
#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root
cd "$(dirname "$0")/.."

echo "Setting up distributed parameter server..."

# Install uv if it is missing
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# Install locked dependencies and create .venv
echo "Installing Python dependencies..."
uv sync --locked

# Generate RPC stubs, if proto definitions exist
if compgen -G "proto/*.proto" > /dev/null; then
    echo "Generating gRPC stubs..."

    # Generate Python RPC stubs using buf.gen.yaml
    buf generate

    # Ensure generated code is a Python package
    mkdir -p src/generated
    touch src/generated/__init__.py
fi

echo "Environment setup complete!"
echo "Run commands using: uv run python ..."
