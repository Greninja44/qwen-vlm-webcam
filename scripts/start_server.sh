#!/usr/bin/env bash
# Run the model server by hand (app.py also starts it automatically).
cd "$(dirname "$0")/.." || exit 1
exec python3 -c "import subprocess,sys,server;sys.exit(subprocess.call(server.server_cmd()))"
