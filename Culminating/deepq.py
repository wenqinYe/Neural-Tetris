import optimizer
import layers

import sys
import MaTris.matris as matris_module
import random
import time
reload(layers)
reload(optimizer)
reload(matris_module)
import pdb
import scipy.ndimage 
import scipy.signal as signal

from MaTris.matris import *
from numpy import linalg as la


class DeepQ():
    def __init__(self, matris=None, optimizer=None, neural_net=None, encoder_net=None):
        """
        Construct a DeepQ object.
        
        :param matris: A Matris game object
        :param optimizer: A neural network optimizer
        :param neural_net: The neural net that the deep_q algorithm trains on
        :encoder_net: The autoencoder to represent the high level game state
        """
        self.matris = matris
        self.optimizer = optimizer
        if not neural_net:
            self.net = self.optimizer.net
        else:
            self.net = neural_net
            
        self.encoder_net = encoder_net
            
        self.goal = np.ones((10, 22))
        
        self.replay_memory = []
        
        self.encoded_goal_state = np.zeros((22, 10))
        self.encoded_goal_state[21] =  np.ones(10)
        self.encoded_goal_state = self.encoded_goal_state.ravel()

    def encoded_state(self, state):
        """
        Creates an encoded version of the tetris matrix that represents
        high level features of the game.
        
        :param state: State is a matrix dictionary object
        :return: An encoded version of the tetris game state
        """
        self.encoder_net.forward(self.matris.dict_to_matrix(state).ravel())
        return self.encoder_net.outputs_cache[2]
    
    def Q(self, state, piece):
        """
        Finds the 
        :param state: A dict that represents the tetris board (with shadow
               piece)
               
        :return: An array representing the scores of each move 
                 (Will be none if another move is not possible)
        """
        try:  
            game_matrix = self.matris.matrix
            game_matrix = self.matris.blend(shape=piece.shape,matrix=game_matrix)
            self.encoder_net.forward(self.encoded_state(game_matrix))
            encoded_state = self.encoder_net.outputs_cache[1]
            return self.net.forward(encoded_state)
        except:
            return None
        
        
    def Q_max(self, state, piece):
        """
        State - a matris board dict
        Piece - a tetromino object (not the shape)
        
        Given a game state, this function
        will compute (from the neural network)
        and return the best moves to make (argmax).
        
        Also returns the score for the best move (amax)
        """
        next_states = self.matris.generate_next_states(state, piece)

        scores = []
        for next_state in next_states:
            next_state, moves  = next_state
            
            score = self.Q(next_state) 
            scores.append(scores)
            
        best_state = np.argmax(np.array(scores))
        
        """
        execute the actions to reach
        the best state
        """
        
        moves = next_states[best_state][1]
        
        return (moves, scores[best_state])
        
    def Q_max_moves(self, state, piece):
        """
        Same as Q_max but deals with raw actions rather
        than final piece locations
        
        Returns the index of the highest value action
        
        This will return None if no moves can be made.
        """
        scores = self.Q(state, piece)
        if scores == None:
            """
            gameover state
            """
            return None, None
        max_index = np.argmax(np.array(scores))
        return max_index, scores[0][max_index]
        
        
    def execute_moves(self, moves):
        """
        Moves is a tuple of (deltaX_movement, rotation_number)
        After the moves are executed the block is automatically
        hard dropped.
        """
        self.matris.execute_moves(moves)
        
    def intermediary_reward(self, state):
        goal_value = self.goal_value(state)
        return  -1 * goal_value * 0.01
        
    def play(self, time):
        """
        Plays one iteration of Q-learning
        
        :param time: The current time step
        """
        state = self.matris.matrix
        tetromino = self.matris.current_tetromino

        move, score = self.Q_max_moves(state, tetromino) 
        if move == None:
            print "WARNING: Deep Q could not find a move (probably because \
                   the tetris game board is invalid."
            print move
            return
        if np.random.rand() < (1-(time/100)):
            move = np.random.randint(0, 7)
        original_score = self.matris.score
        self.matris.execute_move_index(move) 
        reward = self.matris.score - original_score
        new_state = self.matris.matrix
        
        reward = self.intermediary_reward(new_state)
        if len(self.replay_memory) > 1 and reward == self.replay_memory[-1][3]:
            return
        self.replay_memory.append((state, tetromino, move, reward, new_state))
        
        
    def learn(self):
        """
        Runs one interation where the neural network learns from its replay
        memory.
        """   
        X = []
        Y = []
        for i in range(len(self.replay_memory)):
            if i > 10:
                break
            state, tetromino, moves, reward, new_state = random.choice(self.replay_memory)
            self.encoder_net.forward(self.matris.dict_to_matrix(state))
            encoded_state = self.encoder_net.outputs_cache[1]
            
            max_score = self.Q_max_moves(state, tetromino)[1]
            if max_score == None:
                continue
            X.append(encoded_state)

            expected = reward + self.Q_max_moves(state, tetromino)[1]
                
            Y.append(expected)
            
        self.optimizer.run_minibatch(X, Y)
        
        
        
    def goal_value(self, state):
        """
        Determines how close the given state is to reaching the goal.
        
        :param state: The current game state represented by a matris game dict
        """

        return la.norm(self.encoded_state(state) - self.encoded_goal_state)
            
    def open_object(self, filename):
        """
        Loads an episode file
        
        :param filename: Path to the pickle file
        """
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
         
def run_deepq_script():
    """
    Runs the deepq learning script 
    """   
    autoencoder = layers.NeuralNet([['input', {"size": (1, 220)}], 
    ['relu', {"size": (1, 15)}],
    ['relu', {"size": (1, 220)}]
    ])
    autoencoder.load_parameters('/Users/wenqin/Documents/GitHub/grade-12-assignments-wenqinYe/Culminating/parameters/encoder')
    
    q_net = layers.NeuralNet([['input', {"size": (1, 15)}], 
    ['relu', {"size": (1, 8)}], 
    ['relu', {"size": (1, 7)}]
    ])
    opter = optimizer.Optimizer(q_net)
    deep_q = DeepQ(matris=None, optimizer=opter, neural_net=q_net, encoder_net=autoencoder)
    
    episodes = deep_q.open_object('/Users/wenqin/Documents/GitHub/grade-12-assignments-wenqinYe/Culminating/parameters/memory_v2.pkl')
    
    def callback(matris, time):    
        deep_q.matris = matris
        deep_q.play(time)
        deep_q.learn()
        
    if __name__ == '__main__':
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("MaTris")
        game = Game()
        while(1):
            game.main(screen, callback)
            
run_deepq_script()