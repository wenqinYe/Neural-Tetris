# Neural Tetris

This is the code I used to complete my grade 12 culminating assignment in computer science. 

The idea was to combine  q-learning with the computer "imagining" a goal state. Ultimately it didn't play very well, but this code is a good starting point for anybody interested in building their own neural network tetris AI.



The neural network, optimizer and q learning class are coded from scratch using numpy. 

 ## Class Structure

Layers.py

- Houses the code for a general neural network. Includes feedforward, backpropagation (adagrad) as well as gradient caching for mini batch runs.
- Allows parameters (weights and biases) of the neural network to be saved.

Optimizer.py

- Trains the neural network given some data, and expected outputs. 
- Has an error visualization plot to view how well the neural network is training over time.

Encoder.py

- Autoencoder trained on my friend's tetris gameplay (he's a level 20 grandmaster on tetris friends - thank you Tim!).

Deep_q.py

- Where the game play and q-learning happens.
- Goal value returns how far away the current encoded tetris board is away from an ideal tetris board (where there is a full line). This implements the idea of the computer imagining a goal state.



## Getting Startd

Go to deep_q.py and call run_deepq_script() to start the neural network! 





​	



