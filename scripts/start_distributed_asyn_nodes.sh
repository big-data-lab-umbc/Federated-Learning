#!/bin/bash

python3 ./distributed_asyn_node.py 0& sleep 1 
python3 ./distributed_asyn_node.py 1& sleep 1 
python3 ./distributed_asyn_node.py 2& sleep 1 
