## Introduction 
Hello! 
  
This is the example code I wrote to implement ring communication using ***send*** and ***recv*** in ***torch.distributed***. 
  
Here is a brief description.
  
model is cnn, dataset is mnist(`needn't download by yourself`. It will download automatically when you run the code, its code located in ***52-57 lines*** in ***train.py***)
 
## where is the core code of communication
It is ***33-57 lines*** in the ***train_model()*** function in ***run.py*** file under ***utiles*** folder, between 'loss.backward()' and 'optimizer.step()'.   
  
There are detailed comments to explain my ideas.
 
## Installing
 
Just download this folder in all servers directly.
 
## Env
* You should install **Anaconda** in all servers if you haven't it.
* You should schdule Configure conda environment in all servers.**I have pack conda env for you.** 
    
    * In all your servers, you can download the **"environment.yaml"**  and run ***"conda env create -f environment.yaml"***, then you will get the env "fl38" which contains all the required packages.
      
    * you can activate this env by ***"conda activate fl38"***
      
* You need to have several servers that can communicate with each other, and corresponding aliases in their hosts. 
    * For example, they should contain each other in their ***/etc/hosts*** :
    
         192.168.0.1 server1
      
         192.168.0.2 server2

* Find your **gloo_socket_ifname**：
    * run ***"ifconfig"*** on the command line. 
    * If server1'ip is 192.168.0.1, then the name whose ip is 192.168.0.1 is to find gloo_ socket_ name. 
    * It will be used when we run train.py. (different server may have different)
 
## Running
 
* When you complete the above environment configuration, you need to do is：
    * 1.enter the "./ring-reduce" directory, the directory where "train.py" is located   
    * 2.Check the port to ensure that it is not used or occupied. For example, you choose port 23456, run ***"netstat anp | grep 23456"*** on the command line to check.
    * 3.To avoid communication problems, you can try to temporarily close the firewall.
    * 4.Choose use the trained model--"model_state_dict.pth" or not. If you use it, you just need to add "-- model_path  [the path of model_state_dict.pth]" at the end of the nohup directive
    * 5.**Train experience**：Accuracy will be very low before about 300-600 epoches,and will jump to 98%-99.99% at 400-800 epoches. You can set the epoch to 1200, which may be better if you need to train for high accuracy. You can also directly load the trained model **"model_state_dict.pth"** or half-trained model **"model_state_dict_300.pth"**
    * 6.enter "nohup python3 -u train.py --init-method tcp://[your_master-server_ip] --rank [the_rank] --world-size [world_size] --gloo_socket_ifname[current-server_gloo-socket-ifname]" on the command line. For example, their are two servers:
     
     * **In server1(master):**
     
     \>\> ***conda activate fl38***
     
     \>\> ***cd /home/mxy01/ring-reduce***  (the code path)
     
     \>\> ***nohup python3 -u train.py --init-method tcp://192.168.0.1:23456 --epoch 1200 --rank 0 --world-size 2  --gloo_socket_ifname eth0  （--model_path "/home/mxy01/fl/ring-reduce/model_state_dict.pth"）***
     
    * **In server2:**
     
     \>\> ***conda activate fl38***
     
     \>\> ***cd /home/maxvyang02/ring-reduce***
     
     \>\> ***nohup python3 -u train.py --init-method tcp://192.168.0.1:23456 --epoch 1200 --rank 1 --world-size 2  --gloo_socket_ifname enps1 （--model_path "/home/mxy01/fl/ring-reduce/model_state_dict.pth"***
     
     
    * explain
    
     **--world_size 2** means two servers(world_size) 
    
     **192.168.0.1** is the ip address of the server1 in **/ect/hosts**.
     
     **--gloo_socket_ifname enps1** find by **ifconfig**
  
* When the training reaches about 90 epochs, the accuracy of the test_set will reach more than 90%

## Code structure explanation
 * "Train.py"  is the main entrance of the program.    
    * At the bottom, "if __name__=="__ main__ ":" start multi-threaded task. Then, in "init_process()",  carry out initialization configuration, load data and models,and make each worker enter training through "utils. run()" in 106 lines
* "run.py"    in the "utils" folder is used to train and test the model, and the communication code is written in the "train_model()" function.
* "data" folder stores the dataset
* "Usage.py"   is not used temporarily
* "model_state_dict.pth" is the model parameter after training with lr=0.9 0.93 0.97, epoch=1500
 
## Communication ideas
We need two ring communications：
  
The first ring:
1. Begin from rank 0，rank 0 send its grad to rank 1(its right worker).
2. Others：
   1. firstly, recive its right worker; 
   2. secondely, add its grad and updata itself; 
   3. thirdly, send the new grad to its right worker
3. After all workers done ,the last rank（rank size-1） aggregate all worker's grad ,and send its grad to rank 0
  
The second ring:
  
1. Now, begin from rank 0, spread the aggregated grad ring one-to-one
2. Because the grad of rank size-1 is the grad that aggregates all the rank, so rank size-2 does not need to send its grad to rank size-1,just stop at rank size-2
3. Finally, divide grad by "world_size" to get the average gradient and complete communication
 
## Authors
 
* **Xvyang Ma** - [Homepage](https://github.com/serein-dream)
* Main Reference: ***Mixed-Compression***  -[Project Page](https://github.com/ZhangQBx/Mixed-Compression/tree/zhangqinbo)  -[**Qinbo Zhang**](https://github.com/ZhangQBx)
* 
    .
    .
    (waiting for you to add :P)
 
## License
 
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
 
