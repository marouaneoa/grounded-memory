#!/bin/bash
#
# Full Pipeline Test Script for Grounded Memory System
#
# This script tests ALL key features:
# - LLM Entity Extraction
# - Candidate Fact Creation  
# - Constraint Validation (Approval & Rejection)
# - Fact Supersession
# - Temporal Management
#
# Usage: ./scripts/test_full_pipeline.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "==========================================="
echo "Grounded Memory System - Full Pipeline Test"
echo "==========================================="
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Resolve Python interpreter
echo "Resolving Python interpreter..."
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
    echo "✓ using .venv interpreter: $PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    echo "⚠ using system interpreter: $PYTHON_BIN"
else
    echo "✗ No Python interpreter found"
    exit 1
fi

# Set PYTHONPATH (safe under set -u)
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# Check if required packages are available
echo ""
echo "Checking dependencies..."
"$PYTHON_BIN" -c "import pydantic; print('✓ pydantic')" 2>/dev/null || { echo "✗ pydantic not installed"; exit 1; }
"$PYTHON_BIN" -c "import rich; print('✓ rich')" 2>/dev/null || { echo "✗ rich not installed"; exit 1; }
"$PYTHON_BIN" -c "import httpx; print('✓ httpx')" 2>/dev/null || { echo "✗ httpx not installed"; exit 1; }

# Check if .env exists for LLM configuration
if [ ! -f .env ]; then
    echo ""
    echo "⚠ Warning: .env file not found"
    echo "  The test requires LLM configuration (OPENROUTER_API_KEY or similar)"
    echo "  Create a .env file with your API configuration"
    echo ""
fi

# Run the test
echo ""
echo "Running full pipeline test..."
echo "==========================================="
echo ""

if [ -f demos/demo_openrouter.py ]; then
    PIPELINE_ENTRY="demos/demo_openrouter.py"
else
    echo "✗ No pipeline entrypoint found. Expected:"
    echo "  - demos/demo_openrouter.py"
    exit 1
fi

echo "Using pipeline entry: $PIPELINE_ENTRY"

"$PYTHON_BIN" "$PIPELINE_ENTRY"

echo ""
echo "==========================================="
echo "Test completed!"
echo "==========================================="
