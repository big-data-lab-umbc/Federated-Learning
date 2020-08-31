# Federated-Learning-With-Tree-Structure

# Requirements  
  
python3.5  
treelib  
Keras==2.2.3  
flask-socketio  
python-socketio
python-engineio 
numpy==1.18.0
scipy==1.4.0
msgpack-numpy  
socketIO-client  
tensorflow==1.10.0  

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
