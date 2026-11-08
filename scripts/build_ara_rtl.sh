#!/usr/bin/env bash
# Fetch IP deps with Bender, compile a test app, and build the Verilator model of CVA6+Ara.
set -u
git config --global --add safe.directory '*'
cd /mnt/d_ara/hardware
echo "== bender =="
make bender || exit 1
echo "== bender checkout (IP dependencies) =="
# Use the committed Bender.lock (do NOT run `bender update`: it re-resolves and hits axi version conflicts)
cd /mnt/d_ara && git checkout -- Bender.lock 2>/dev/null
./hardware/bender checkout || exit 1
cd /mnt/d_ara/hardware
echo "== compile hello_world =="
cd /mnt/d_ara/apps
make bin/hello_world > /mnt/d_ara/app_build.log 2>&1; echo "app exit: $?"; tail -n 5 /mnt/d_ara/app_build.log
echo "== verilate =="
cd /mnt/d_ara/hardware
make verilate > /mnt/d_ara/verilate_build.log 2>&1; echo "verilate exit: $?"
tail -n 15 /mnt/d_ara/verilate_build.log
