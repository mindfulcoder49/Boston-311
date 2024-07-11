#!/bin/bash

# Deactivate and remove the current virtual environment if it exists
if [ -d ".venv" ]; then
    deactivate
    rm -rf .venv
fi

# Create a new virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Upgrade specific packages to ensure compatibility
pip install --upgrade numpy scipy scikit-learn