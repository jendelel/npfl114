#!/bin/bash
if [ -z "$1" ]; then
    echo "./script (script_to_deploy)"
    exit 1
fi

user="oskopek"
local="$1"
file="`basename "$1"`"
sshaddr="$user@skirit.ics.muni.cz"
scp "$file" "$sshaddr:~/$file"
ssh "$sshaddr" "qsub $file"

