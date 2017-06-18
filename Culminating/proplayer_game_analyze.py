""""
Analyze Tim's Gameplay for Tetris
"""
# save_object(Game(), '/Users/wenqin/Documents/GitHub/grade-12-assignments-wenqinYe/Culminating/memory.pkl')a
import cPickle as pickle
import layers
import numpy as np
import deepq
import optimizer
import MaTris.matris as matris
from MaTris.matris import *
import pygame

reload(layers)
reload(deepq)
reload(matris)
reload(optimizer)

        
def open_object(filename):
    episodes = []
    with open(filename, 'rb') as input:
        i = 0
        done_loading = False
        while not done_loading:
            try:
                episodes.append(pickle.load(input))
                i+=1 
                if i > 2000:
                    done_loading=True
            except:
                done_loading = True
    return episodes
    
def shape_to_vector(shape):
    """
    Shape is expected to be a series of tuples
    """
    
    


episodes = open_object('/Users/wenqin/Documents/GitHub/grade-12-assignments-wenqinYe/Culminating/memory_v2.pkl')
m = matris.Matris()

action_number = {'left': 0, 'right': 1, 'rotate0': 2, 'rotate1': 3, 'rotate2': 4, 'rotate3': 5, 'hard_drop': 6}
action_word = ['left', 'right', 'rotate0', 'rotate1', 'rotate2', 'rotate3', 'hard_drop']

# input: 22 * 10 grid + 7 tetrominoes * 4 rotations
# output: left, right, rotate, hard_drops

column_neurons = []

#n.append(layers.NeuralNet(layers=[['input', 20], ['tanh', 10], ['tanh', 4]]))
n = layers.NeuralNet(layers=[['input', {"size": (1,220)}], ['tanh', {"size":(1,50)}],
                            ['tanh', {"size":(1,20)}], ['linear', {"size":(1,7)}]])
o = optimizer.Optimizer(n)
n.alpha_weight = 1e-4
n.alpha_bias = 1e-4
n.adagrad_decay = 0.9
"""
Error = (Q(s', a') + reward) - Q(s, a))
Error is prediction of neural network subtracted from
the actual reward and then teh prediction from completing optimaly in the new state
"""
X = []
Y = []
rewards = []

for episode in episodes:
    """
    Each episode is consists of:
    (current_state, move, reward, new_state, rot_before, rot_after, current_tetromino.shape, next_tetromino.shape)
    
    Moves consist of (in order):
    left, right, r1, r2, r3, r4, hard_drop
    """
    state, move, reward, new_state, rot_before, rot_after, shape, next_shape = episode
    state_with_active_piece = m.dict_to_matrix(m.blend(shape=shape, matrix=state)).ravel()
    if not reward > 0:
        continue
    """
    Predicts a bunch of actions. The best move is the one that the player actually does
    """
    if move == "request_rotation":
        move = "rotate"+str(rot_after)
    action_index = action_number[move]
        
    q_sa = n.forward(state_with_active_piece)
    # q_sa[:, 0: action_index] = 0
    # q_sa[:, action_index+1:4] = 0
    
    new_state_with_new_piece = m.dict_to_matrix(m.blend(shape=shape, matrix=new_state)).ravel()
    q_sa_prime = n.forward(new_state_with_new_piece)
    
    q_sa_prime[:, 0: action_index] = 0
    q_sa_prime[:, action_index+1:6] = 0
    q_sa_prime[:, action_index] += ((reward/500.0)) 
    
    
    error = q_sa_prime - q_sa
    if np.count_nonzero(error) == 0:
        continue
        
    
    X.append(np.matrix(state_with_active_piece))
    Y.append(np.matrix(error))

o.start_error_plot()
i = 0
for epoch in range(2000):
    o.run_minibatch(X, Y, batch_size=10)
    i += 1
    
    if i % 25 == 0:  
        o.update_error_plot()
    


def callback(matris, time):
    current_shape = matris.current_tetromino.shape
    current_game_state = matris.dict_to_matrix(matris.blend(shape=current_shape, matrix=matris.matrix)).ravel()
    q_scores = n.forward(current_game_state)
    best_action = np.argmax(q_scores)
    
    move = action_word[best_action]
    matris.execute_move(move) 
    
if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MaTris")
    game = matris.Game()
    while(1):
        game.main(screen, callback)
            