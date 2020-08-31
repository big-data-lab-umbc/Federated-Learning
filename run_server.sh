#!/bin/bash

oldifs="$IFS"

IFS=$'\n'


until [ python3 seq_server.py | $(while read line) ];
do
    echo "@@@";
done



for line in $(python3 seq_server.py); do
    if [ ${line:0:1} = "l" ]; then echo $line > /dev/null; 
    else echo $line;
    fi
done



IFS="$oldifs"