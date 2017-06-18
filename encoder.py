import cPickle as pickle
import layers
import numpy as np
import optimizer
import MaTris.matris as matris
from MaTris.matris import *
import pygame

from numpy import linalg as la
import sys

reload(layers)
reload(matris)
reload(optimizer)

class AutoEncoder():
    """
    A tailor made autoencoder class for Deep Q learning on a tetris board.
    """
    def __init__(self):
        self.episodes = []
        self.X = []
        self.Y = []
        self.autoencoder = layers.NeuralNet([
            ['input', {"size": (1, 220)}], 
            ['relu', {"size": (1, 15)}],
            ['relu', {"size": (1, 220)}]
        ])
        self.optimizer = optimizer.Optimizer(self.autoencoder)
        self.autoencoder.alpha_weight = 1e-3
        self.autoencoder.alpha_bias = 1e-3
        self.autoencoder.adagrad_decay = 0.9
        
        self.matris = Matris()
        
        
    def open_episode_file(self, filename):
        """
        Loads episodes from human player gameplay
        
        :return: An array of episodes from a human player.
        """
        self.episodes = []
        with open(filename, 'rb') as input:
            i = 0
            done_loading = False
            while not done_loading:
                try:
                    self.episodes.append(pickle.load(input))
                    i+=1 
                    if i > 2000:
                        done_loading=True
                except:
                    done_loading = True
                    
        """
        Filter episodes that are zero reward, and sort by increasing reward
        to improve training.
        """
        self.episodes = self.search_rewards_greater_than(0, self.episodes)
        self.episodes = self.sort_episodes(self.episodes)
        return self.episodes

    
    def load_episodes(self):
        """
        Loads the episodes from the specified folder
        """
        self.episodes = self.open_episode_file('/Users/wenqin/Documents/GitHub/grade-12-assignments-wenqinYe/Culminating/parameters/memory_v2.pkl')
        for episode in self.episodes:
            state, move, reward, new_state, rot_before, rot_after, shape, next_shape = episode
            if reward > 0:
                matrix = self.matris.dict_to_matrix(state)
                self.X.append(matrix.ravel())
                self.Y.append(matrix.ravel())
                
    def load_params(self):
        """
        Loads pretrained auto encoder parameters from file
        """
        self.autoencoder.load_parameters('/Users/wenqin/Documents/GitHub/grade-12-assignments-wenqinYe/Culminating/parameters/encoder')
    
    def save_params(self):
        """
        Saves the autoencoder parameters to a file
        """
        self.autoencoder.save_parameters('/Users/wenqin/Documents/GitHub/grade-12-assignments-wenqinYe/Culminating/parameters/encoder')

    def search_rewards_greater_than(self, number, episodes, index=0, results=[]):
        """
        Searches for episodes that have a reward greater than a certain value
        (RECURSIVE LINEAR SEARCH)
        
        :param: number: The number that the reward has to be greater than
        :return: Retruns a list of episodes with a reward greater than the specified 
                number.
        """
        sys.setrecursionlimit(5000)
        if index > len(episodes)-1:
            return results
        
        if self.reward_from(episodes[index]) > number:
            results.append(episodes[index])
        return self.search_rewards_greater_than(number, episodes, index+1, results)
        
    def reward_from(self, episode):
        """
        :return: Returns the reward from an episode
        """
        return episode[2]
        
    def sort_episodes(self, episodes):
        """
        Sorts the episodes (using bubble sort) in order of increasing rewards.
        In theory this could make the neural network train better 
        because it trains on high quality data first.
        
        :param episodes: A list of episodes that represent game states
        :return: Return a list of episodes in descending order
        """
        for i in range(len(episodes)):
            for j in range(len(episodes)-i-1):
                if i == j:
                    continue   
                if self.reward_from(episodes[j]) > self.reward_from(episodes[j+1]):
                    tmp = episodes[j+1]
                    episodes[j+1] = episodes[j]
                    episodes[j] = tmp
                    
        return episodes
                    
    def train(self, epochs):
        """
        Trains the autoencoder.
        
        :param epochs: Trains the neural network for a specifid number of iterations
        """
        self.optimizer.start_error_plot()    
        for i in range(epochs):
            self.optimizer.run_minibatch(self.X, self.Y, batch_size=100)
            if i % 10 == 0:
                self.optimizer.update_error_plot()
            
a = AutoEncoder()
a.load_episodes()
episodes = a.episodes
