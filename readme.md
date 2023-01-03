## Introduction 
Hello! 
  
This is the example code I wrote to implement ring communication using ***send*** and ***recv*** in ***torch.distributed***. 
  
Here is a brief description.
  
model is cnn, dataset is mnist(`needn't download by yourself`. It will download automatically when you run the code, its code located in ***52-57 lines*** in ***train.py***)
 
## where is the core code of communication
It is ***33-57 lines*** in the ***train_model()*** function in ***run.py*** file under ***utiles*** folder, between 'loss.backward()' and 'optimizer.step()'.   
  
There are detailed comments to explain my ideas.
 
## Installing
 
Just download this folder directly
 
## Env
* Firstly, you should have installed **anconda**
* Secondly ,you should install the some packages. We import them in ***train.py*** and ***run.py***.When you debug, it will remind you which versions need to be installed or changed
* Thirdly, you need to have several servers that can communicate with each other, and corresponding aliases in their hosts. 
    * For example, in execute.sh, ***machines={01 02}***, ***$cluster_ name=ring***, then alias the two servers as ***ring01*** and ***ring02***:
    * 192.168.0.1 server1 ring01
    * 192.168.0.2 server2 ring02
    * If **RuntimeError: The server socket has failed to listen on any local network address** occurs, you can try to close the port firewall.
    * For example, in execute.sh, if you set "$master_ip" to 192.168.0.1:2345, you need to ensure that port 2345 is open. If an error is still reported, try to close the firewall on port 2345
* Set the three folder paths as follows：
    * home_path="/home/maxvyang"    the root
    * atcivate_path="${home_path}/anaconda3/bin/activate"   when you install conda,you will find the **/anaconda3**
    * env_path="${home_path}/anaconda3/envs/fl38"           **fl38** is my conda env name,you can create your conda env with any name 
    * code_path="${home_path}/fl/ring-reduce"             **/ring-reduce**is the **/RingCommunication** on my github
* Finally，Set these three directories as shared folders(atcivate_path, env_path, code_path). 
    * If you don't know how to do it, you can refer to: -[Here] (https://blog.csdn.net/qq_41853833/article/details/126479493)  
## Running
 
* When you complete the above environment configuration, you need to do is：
    * 1.enter the "./ring-reduce" directory, the directory where "extract. sh" is located   
    * 2. enter "bash excute.sh ${world_size} $cluster-name $master_ip" on the command line,for example:
    * bash excute.sh 2 ring 192.168.0.1:2345   
    * **2** means two servers(world_size) 
    * **ring** is cluster_name,Corresponding to the alias in **/ect/hosts**  
    * **192.168.0.1** is the ip address of the server1(where the code located in) in **/ect/hosts**, and **2345** is the port  refer **env** above
  
* If you want to change the number of workers for parallel training, you can change the value of "world_size" in "execute_cpu. sh", which represents the number of workers
  
* When the training reaches about 90 epochs, the accuracy of the test_set will reach more than 90%

## Code structure explanation
 * "Train.py"  is the main entrance of the program.    
    * At the bottom, "if __name__=="__ main__ ":" start multi-threaded task. Then, in "init_process()",  carry out initialization configuration, load data and models,and make each worker enter training through "utils. run()" in 106 lines
* "run.py"    in the "utils" folder is used to train and test the model, and the communication code is written in the "train_model()" function.
* "data" folder stores the dataset
* "Usage.py"   is not used temporarily
 
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
* Main Reference: ***Mixed-Compression***  -[Project Page](https://github.com/ZhangQBx/Mixed-Compression/tree/zhangqinbo)  -[**Qingbo Zhang**](https://github.com/ZhangQBx)
* 
    .
    .
    (waiting for you to add :P)
 
## License
 
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
 
