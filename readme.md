## Introduction 
Hello! 
  
This is the example code I wrote to implement ring communication using ***send*** and ***recv*** in ***torch.distributed***. 
  
Here is a brief description.
  
model is cnn, dataset is mnist(`needn't download by yourself`It will download automatically when you run the code，its code located in ***52-57 lines*** in ***train.py***)
 
## where is the core code of communication
It is ***33-57 lines*** in the ***train_model()*** function in ***run.py*** file under ***utiles*** folder, between 'loss.backward()' and 'optimizer.step()'.   
  
There are detailed comments to explain my ideas.
 
## Installing
 
Just download this folder directly
 
## Running
 
* All you need to do is：
    * 1.enter the "./ring reduce" directory   
    * 2. enter "sh execute_cpu.sh" on the command line
  
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
 
* **Xvyang Ma** - *Initial work* - [Homepage](https://github.com/serein-dream)
    .
    .
    .
    (waiting for you to add :P)
 
## License
 
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
 
