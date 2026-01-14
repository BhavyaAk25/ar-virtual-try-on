#!/bin/bash

# AR Virtual Shirt Try-On - Setup Script
# This script automates the installation process

echo "======================================"
echo "AR Virtual Shirt Try-On - Setup"
echo "======================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed!"
    echo "Please install Python 3.7 or higher and try again."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -eq 0 ]; then
    echo "✓ Virtual environment created"
else
    echo "✗ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

if [ $? -eq 0 ]; then
    echo "✓ Virtual environment activated"
else
    echo "✗ Failed to activate virtual environment"
    exit 1
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import cv2; import mediapipe; import numpy; import PIL; print('All packages imported successfully!')"

if [ $? -eq 0 ]; then
    echo "✓ Installation verified"
else
    echo "✗ Installation verification failed"
    exit 1
fi

echo ""
echo "======================================"
echo "Setup Complete! 🎉"
echo "======================================"
echo ""
echo "To run the application:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the app:"
echo "     python3 main.py"
echo ""
echo "  3. When done, deactivate the environment:"
echo "     deactivate"
echo ""
echo "For more information, see README.md"
echo ""
