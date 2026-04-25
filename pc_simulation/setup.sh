#!/bin/bash
echo "Setting up Waste Classifier..."

python3 -m venv venv
source venv/bin/activate

echo "Installing PyTorch (CPU)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete! Run the classifier with:"
echo "  bash run.sh"
