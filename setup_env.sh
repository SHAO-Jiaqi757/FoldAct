#!/bin/bash

# Setup script for Search_R1 environment
# This script sets up the necessary environment variables

echo "Setting up Search_R1 environment..."

# Set library path for flash-attn compatibility
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Activate conda environment
# Use the correct conda initialization for Zsh
source /datapool/miniconda3/etc/profile.d/conda.sh
conda activate verl-agent

echo "Environment setup complete!"
echo "You can now run your training scripts."
echo ""
echo "To make this permanent, add the following to your ~/.bashrc or ~/.zshrc:"
echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH"
echo "source /datapool/miniconda3/etc/profile.d/conda.sh" 