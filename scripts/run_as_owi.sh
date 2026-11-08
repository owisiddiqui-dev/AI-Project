#!/usr/bin/env bash
# Launcher: run as root. Ensures the swapfile and the /mnt/d_ara bind mount exist,
# then runs the given script as user owi. The mount lives only while WSL is running, so every
# long job must be launched through this wrapper.
# Usage (from Windows): wsl -d Ubuntu-22.04 -u root --cd /tmp -- bash "/mnt/d/AI Thesis/scripts/run_as_owi.sh" <script>
swapon /swapfile_extra 2>/dev/null
mkdir -p /mnt/d_ara
mountpoint -q /mnt/d_ara || mount --bind "/mnt/d/AI Thesis/ara" /mnt/d_ara
exec runuser -u owi -- bash "$@"
