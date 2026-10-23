#!/bin/bash
# Run on the Lawrence LOGIN node:  bash cluster/discover.sh  (or paste it).  Read-only; prints facts we need.
echo "== user / home";        whoami; echo "$HOME"; df -h "$HOME" | tail -1
echo "== lab dir";            ls -ld ~/santosh_lab; readlink -f ~/santosh_lab; ls -la "$(readlink -f ~/santosh_lab)" | head -20
df -h "$(readlink -f ~/santosh_lab)" | tail -1
echo "== partitions (name, time limit, nodes, gres)"
sinfo -o "%P %l %D %G %N" 2>/dev/null
echo "== gpu nodes (gres labels + state)"
sinfo -p gpu -N -o "%N %G %t %c %m" 2>/dev/null
echo "== my limits / associations"
sacctmgr -n show assoc user="$USER" format=account,partition,qos 2>/dev/null | head
echo "== conda"
ls ~/miniforge3/bin/conda ~/miniforge3/bin/mamba 2>/dev/null; ~/miniforge3/bin/conda --version 2>/dev/null
echo "== modules mentioning cuda"
module avail cuda 2>&1 | head -20
