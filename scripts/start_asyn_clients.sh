#!/bin/bash

python3 ./asyn_client.py 0& sleep 1 
python3 ./asyn_client.py 1& sleep 1 
python3 ./asyn_client.py 2& sleep 1 
