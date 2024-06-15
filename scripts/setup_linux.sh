#!/bin/sh
set -e

Yellow='\033[0;33m'
notebook="notebook_eric_llms"
PROJECT="hal2"
LOCAL_DIR="/Users/ericgu/src/./$PROJECT/"  # . and trailing slash important
REMOTE_DIR="/opt/projects"
REMOTE_PROJ_DIR="$REMOTE_DIR/$PROJECT"

rsync -avzR --delete --filter=":- .gitignore" -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" $LOCAL_DIR $notebook:$REMOTE_DIR
ssh $notebook /bin/bash << EOF
  cd $REMOTE_PROJ_DIR
  if [ ! -d ".venv" ]; then
    python -m venv .venv
  fi
  source .venv/bin/activate
  pip install -r requirements.txt
EOF
echo "${Yellow}Cloned hal2 repo and installed requirements"
