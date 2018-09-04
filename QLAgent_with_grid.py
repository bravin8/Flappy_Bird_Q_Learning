# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 09:51:43 2018

@author: Bravin
"""
import math
import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
import time

class QLAgent():
    def __init__(self,flappy_actions,grid_size):
        
        #intializing state variables array length 
        #grid sixe is used to reduce the size of the variables into 10th of original size
        #array length to store horizontal distance to the next pipe
        horizontal_distance_next = math.ceil(350/grid_size)
        #array length to store horizontal distance to next next pipe 
        horizontal_distance_next_next = math.ceil(700/grid_size)
        #array length to store horizontal distance ro the next pipe
        vertical_distance_lower = math.ceil(1024/grid_size)
        
        #initializing the q-matrix with 4d array.3 of them are to hold the state values one to hold the action value
        self.Q_values = np.zeros((horizontal_distance_next,horizontal_distance_next_next,vertical_distance_lower,2))
        #initializing the action variable
        self.flappy_actions = flappy_actions
        #initializing the grid size variable 
        self.grid_size = grid_size
        #initializing the learning rate variable 
        self._alpha = 0.1
        #inizializing the discount factor variable
        self._gamma = 1
        
    #function to get the state of the agent and store it in a state array
    def get_current_state(self,flappy_observations):
        game_state = np.zeros((3,), dtype=int)
        game_state[0] = flappy_observations['next_pipe_dist_to_player'] // self.grid_size
        game_state[1] = flappy_observations['next_next_pipe_dist_to_player'] // self.grid_size
        #game_state[1] = (flappy_observations['next_pipe_top_y'] - flappy_observations['player_y'] + 512) // self.grid_size
        game_state[2] = (flappy_observations['next_pipe_bottom_y'] - flappy_observations['player_y'] + 512) // self.grid_size
        #game_state[1] = flappy_observations['next_pipe_bottom_y']
        #return the state array
        return game_state
        
    #definfing function to perform an action and retun values based on the rewards
    def perform_action(self,p,action):
        reward = p.act(self.flappy_actions[action])
        #if reward is greater than or equal to 0 return 1 or else return -1000
        if reward >= 0:
            return 1
        else:
            return -1000
    
    #method to get the optimal action based on the Q value matrix
    def get_action(self,game_state):
        flappy_jump = self.Q_values[game_state[0],game_state[1],game_state[2],0]
        flappy_not_jump = self.Q_values[game_state[0],game_state[1],game_state[2],1]
        
        if flappy_jump > flappy_not_jump:
            return 0
        else:
            return 1
    
    #method to update the Q values using the equation  
    def update_Q_values(self,game_current_state,game_next_state,reward,action):
        self.Q_values[game_current_state[0], game_current_state[1], game_current_state[2], action] = (1-self._alpha) * self.Q_values[game_current_state[0], 
                     game_current_state[1], game_current_state[2], action] + self._alpha * (reward + self._gamma * np.max(self.Q_values[game_next_state[0],
                                       game_next_state[1],game_next_state[2]]))
                         
            
if __name__ == "__main__":
    game = FlappyBird()
    p = PLE(game, fps=30, display_screen=True)
    #creating a QlAgent class object
    agent = QLAgent(flappy_actions=p.getActionSet(),grid_size=10)

    p.init()
    
    #get the current state values(state array)
    game_current_state = agent.get_current_state(game.getGameState())
    #initializing the episode to 0
    number_of_episods = 0
    #initializing the maximum score variable to 0
    maximum_score = 0
    
    #creating a while loop to itaraqte through the episodes
    while True:
        #get the optimal action to the current state and store in in the variable 
        maximum_action = agent.get_action(game_current_state)
        #get the score in the current episode
        current_score = p.score()
        #get the maximim score by comparing with the current acore 
        maximum_score = max(current_score, maximum_score)
        #get the reward value by performing the above action (rward is either 1 or -1000)
        reward = agent.perform_action(p, maximum_action)
        #get the next state values (state array)
        game_next_state = agent.get_current_state(game.getGameState())
        
        #update the Q values by calling the update Q function
        agent.update_Q_values(game_current_state, game_next_state, reward, maximum_action)
        
        #set the next state as current state 
        game_current_state = game_next_state
        
        time.sleep(0.01)
        
        #after each and every episode print the episode score and the maximum score achived 
        if p.game_over():
            number_of_episods += 1
            print('Episode number: %s, Current episode score score: %s, Maximun score: %s' % (number_of_episods,
                                                                                              current_score, 
                                                                                              maximum_score))
            p.reset_game()