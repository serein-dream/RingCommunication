    Hello! 
    This is the example code I wrote to implement ring communication using send and recv in torch.distributed. Here is a brief description:
    Firsly, the core code of communication is 33-57 lines in the train_model() function in utiles.run, between loss. backward () and optimizer. step (). There are detailed comments to explain my ideas