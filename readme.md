### Introduction
 
    Hello! 
    This is the example code I wrote to implement ring communication using send and recv in torch.distributed. 
    Here is a brief description.
    model is cnn, dataset is mnist(needn't download by yourself)
 
### where is the core code of communication
 
    It is 33-57 lines in the "train_model()" function in "run.py" file under "utiles" folder
, between "loss.backward()" and "optimizer.step()". 
    There are detailed comments to explain my ideas.
 
### Installing
 
    Just download this folder directly
 
### Running
 
    All you need to do is enter "sh execute_cpu.sh" on the command line
    If you want to change the number of workers for parallel training, 
    you can change the value of "world_size" in "execute_cpu. sh"
, which represents the number of workers

### Code structure explanation
 
    "Train.py"  is the main entrance of the program.    
At the bottom, "if __name__=="__ main__ ":" start multi-threaded task. 
Then, in "init_process()",  carry out initialization configuration, load data and models,
and make each worker enter training through "utils. run()" in 106 lines
    "run.py"    in the "utils" folder is used to train and test the model
, and the communication code is written in the "train_model()" function.
    "data" folder stores the dataset
    "Usage.py"   is not used temporarily
 
### Communication ideas
    Begin from rank 0，rank 0 send its grad to rank 1(its right worker).
    Others, firstly, recive its right worker; secondely, add its grad and updata itself; thirdly, send the new grad to its right worker
    After all workers done ,the last rank（rank size-1） aggregate all worker's grad ,and send its grad to rank 0
    
    Now, begin from rank 0, spread the aggregated grad ring one-to-one
    The grad of rank size-1 is the grad that aggregates all the rank, so rank size-2 does not need to be sent to rank size-1 again
    Finally, divide grad by "world_size" to get the average gradient and complete communication
 
### Authors
 
* **Xvyang Ma** - *Initial work* - [Homepage](https://github.com/serein-dream)
 
 
### License
 
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
 
