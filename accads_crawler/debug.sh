#!/bin/bash

echo "Starting debug script..."
echo "Current directory: $(pwd)"
echo "Listing contents:"
ls -la

echo "Checking npm installation:"
npm --version

echo "Checking node installation:"
node --version

echo "Checking Python script:"
cat /home/chromiumuser/docker_in.py

echo "Running Python script with debug output:"
python3 -u /home/chromiumuser/docker_in.py "$@"

# Keep container running for debugging
echo "Script completed. Press Ctrl+C to exit."
tail -f /dev/null 