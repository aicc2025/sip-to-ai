#!/bin/bash

# Mock Bridge Runner Script
# Demonstrates running the SIP-to-AI bridge in mock mode for testing

set -e

echo "=== SIP-to-AI Bridge Mock Demo ==="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies if needed
if [ ! -f ".venv/pyvenv.cfg" ] || ! uv pip list | grep -q "anyio"; then
    echo "Installing dependencies..."
    uv sync
fi

# Set mock configuration
export AI_VENDOR=mock
export FRAME_MS=20
export SIP_SR=8000
export AI_SR=16000
export LOG_LEVEL=INFO
export LOG_FORMAT=text

echo "Configuration:"
echo "  AI Vendor: $AI_VENDOR"
echo "  Audio: ${SIP_SR}Hz (SIP) ↔ ${AI_SR}Hz (AI)"
echo "  Frame Size: ${FRAME_MS}ms"
echo ""

echo "Starting mock bridge (Press Ctrl+C to stop)..."
echo ""

# Run the bridge in mock mode
python -m app.main --mode mock

echo ""
echo "Mock bridge stopped."