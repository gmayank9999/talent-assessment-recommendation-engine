#!/usr/bin/env bash
# Render build script
# installs CPU-only torch first (200MB vs 2GB full torch),
# then the rest of the dependencies, then builds the embedding index

set -e

echo "Installing CPU-only PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "Installing remaining dependencies..."
pip install -r requirements.txt

echo "Building embedding index..."
python scripts/build_index.py

echo "Build complete."
