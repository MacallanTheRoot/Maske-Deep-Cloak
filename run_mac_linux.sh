#!/bin/bash

echo "======================================================="
echo "          MASKE SECURITY SYSTEMS // INITIALIZING"
echo "======================================================="
echo ""

# Check Python
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    echo "[ERROR] Python 3 not found!"
    echo "Please install Python 3.10+ and try again."
    read -p "Press Enter to exit..."
    exit 1
fi

echo "[+] Checking Verification Protocols..."
echo "[+] Updating Neural Dependencies..."
$PIP_CMD install -r requirements.txt

echo ""
echo "[+] Dependencies Synchronized."
echo "[+] Launching Cloak Engine..."
echo ""

$PYTHON_CMD maske_app.py

read -p "Press Enter to exit..."
