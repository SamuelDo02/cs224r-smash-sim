#!/bin/sh
set -e

Yellow='\033[0;33m'

sudo apt-get update
sudo apt-get install p7zip-full libasound2 libegl1 libgl1 libusb-1.0-0 libglib2.0-0 libgdk-pixbuf2.0-0 libpangocairo-1.0-0 libasound2-dev pkg-config libegl-dev libusb-1.0-0-dev -y

PROJECT_DIR="/opt/projects"
EMULATOR_FILE_PATH="$PROJECT_DIR/emulator/Slippi_Online-Ubuntu20.04-Exi-x86_64.AppImage"
cd $PROJECT_DIR
git clone git@gitlab.com:ericyuegu/hal2.git
cd hal2
chmod +x $EMULATOR_FILE_PATH
$EMULATOR_FILE_PATH --appimage-extract
echo "${Yellow}Extracted emulator"

if [ ! -d ".venv" ]; then
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
fi
echo "${Yellow}Installed venv"

DATA_DIR="/opt/slippi"
mkdir -p $DATA_DIR
# Download ISO from env var
aws s3 cp $SSBM_ISO_PATH $DATA_DIR/ssbm.ciso

echo "${Yellow}Downloaded ISO"
