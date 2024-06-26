#!/bin/sh
set -e

Yellow='\033[0;33m'

notebook="notebook_eric_llms"
PROJECT="hal2"
SRC_DIR="/Users/ericgu/src"
LOCAL_PROJ_DIR="$SRC_DIR/$PROJECT"
REMOTE_DIR="/opt/projects"
REMOTE_PROJ_DIR="$REMOTE_DIR/$PROJECT"

rsync -avz --delete --filter=":- .gitignore" -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" $LOCAL_PROJ_DIR/ $notebook:$REMOTE_PROJ_DIR/
ssh $notebook /bin/bash << EOF
  cd $REMOTE_PROJ_DIR
  if [ ! -d ".venv" ]; then
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
  fi
EOF
echo "${Yellow}Cloned hal2 repo and installed requirements"

EMULATOR_DIR="/Users/ericgu/data/ssbm"
REMOTE_EMULATOR_DIR="/opt/slippi"
EMULATOR_FILE_NAME="ssbm_ntsc_1.02.7z"
CISO_NAME="Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso"
rsync -avz --delete --filter=":- .gitignore" --exclude=".DS_Store" --exclude=".localized" -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" $EMULATOR_DIR/ $notebook:$REMOTE_EMULATOR_DIR/
ssh $notebook /bin/bash << EOF
  sudo apt-get update
  sudo apt-get install p7zip-full libasound2 libegl1 libgl1 libusb-1.0-0 libglib2.0-0 libgdk-pixbuf2.0-0 libpangocairo-1.0-0 -y
  cd $REMOTE_EMULATOR_DIR
  7za x $EMULATOR_FILE_NAME
  mv "$CISO_NAME" ssbm.ciso
  ./Slippi_Online-Ubuntu20.04-x86_64.AppImage --appimage-extract
EOF

# Create a Python file with the remote paths
cat << PYTHON_EOF > $LOCAL_PROJ_DIR/hal/emulator_paths.py
from typing import Final

REMOTE_EMULATOR_PATH: Final[str] = "$REMOTE_EMULATOR_DIR/squashfs-root/AppRun.wrapped"
REMOTE_CISO_PATH: Final[str] = "$REMOTE_EMULATOR_DIR/ssbm.ciso"
PYTHON_EOF
rsync -avz --delete --filter=":- .gitignore" -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" $LOCAL_PROJ_DIR/ $notebook:$REMOTE_PROJ_DIR/

echo "${Yellow}Synced emulator & iso and created emulator_paths.py"
