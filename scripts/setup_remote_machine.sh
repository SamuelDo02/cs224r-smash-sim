#!/bin/sh
set -e

Yellow='\033[0;33m'

notebook="notebook_eric_llms"
PROJECT="hal2"
SRC_DIR="/Users/ericgu/src"
LOCAL_PROJ_DIR="$SRC_DIR/$PROJECT"
REMOTE_DIR="/opt/projects"
REMOTE_PROJ_DIR="$REMOTE_DIR/$PROJECT"
EMULATOR_FILE_PATH="$REMOTE_PROJ_DIR/emulator/Slippi_Online-Ubuntu20.04-Exi-x86_64.AppImage"

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

LOCAL_ISO_PATH="/Users/ericgu/data/ssbm/ssbm.ciso"
REMOTE_ISO_PATH="/opt/slippi/ssbm.ciso"
rsync -avz --delete --filter=":- .gitignore" --exclude=".DS_Store" --exclude=".localized" -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" $LOCAL_ISO_PATH/$ISO_FILE_NAME $notebook:$REMOTE_ISO_PATH
ssh $notebook /bin/bash << EOF
  sudo apt-get update
  sudo apt-get install p7zip-full libasound2 libegl1 libgl1 libusb-1.0-0 libglib2.0-0 libgdk-pixbuf2.0-0 libpangocairo-1.0-0 libasound2-dev pkg-config libegl-dev libusb-1.0-0-dev -y
  cd $REMOTE_PROJ_DIR/emulator
  chmod +x $EMULATOR_FILE_PATH
  $EMULATOR_FILE_PATH --appimage-extract
EOF

echo "${Yellow}Synced iso & extracted emulator"
