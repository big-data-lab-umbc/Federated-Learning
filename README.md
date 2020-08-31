# Federated-Learning-With_Tree-Structure

# Requirements  

pip3 install treelib  
pip3 install Keras  
pip3 install flask-socketio 
pip3 install python-socketio
pip3 install python-engineio 
pip3 install numpy  
pip3 install msgpack-numpy  
pip3 install socketIO-client  
pip3 install tensorflow  
(pip3 install tensorflow==1.5.0)  

# How to run  

[Sequential_Mode]  
python3 seq_server.py  

(bash ./scripts/start_seq_clients.sh) 
python3 seq_client.py 0 
python3 seq_client.py 1  
python3 seq_client.py 2   

[Asynchronous_Mode]  
python3 asyn_server.py 
  
(bash ./scripts/start_asyn_clients.sh)  
python3 asyn_client.py 0  
python3 asyn_client.py 1
python3 asyn_client.py 2

[Distributed_Sequential_Mode]  
(bash ./scripts/start_distributed_seq_nodes.sh)  
   
python3 distributed_seq_node.py 0  
python3 distributed_seq_node.py 1  
python3 distributed_seq_node.py 2  
  
[Distributed_Asynchronous_Mode]  
(bash ./scripts/start_distributed_asyn_nodes.sh)  
  
python3 distributed_asyn_node.py 0  
python3 distributed_asyn_node.py 1  
python3 distributed_asyn_node.py 2  
  
# Kill process  
  
sudo pkill -9 python  