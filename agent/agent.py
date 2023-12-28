import json
import os
import random

from .state import State


class Q_State(State):
    '''Augments the game state with Q-learning information'''

    def __init__(self, string):
        super().__init__(string)

        # key stores the state's key string (see notes in _compute_key())
        self.key = self._compute_key()

    def _compute_key(self):
        '''
        Returns a key used to index this state.

        The key should reduce the entire game state to something much smaller
        that can be used for learning. When implementing a Q table as a
        dictionary, this key is used for accessing the Q values for this
        state within the dictionary.
        '''

        # this simple key uses the 3 object characters above the frog
        # and combines them into a key string
        return ''.join([
            self.get(self.frog_x - 1, self.frog_y - 1) or '_',
            self.get(self.frog_x, self.frog_y - 1) or '_',
            self.get(self.frog_x + 1, self.frog_y - 1) or '_',
            self.get(self.frog_x - 2, self.frog_y - 1) or '_',
            self.get(self.frog_x, self.frog_y - 2) or '_',
            self.get(self.frog_x + 2, self.frog_y - 1) or '_',
            self.get(self.frog_x - 3, self.frog_y - 1) or '_',
            self.get(self.frog_x, self.frog_y - 3) or '_',
            self.get(self.frog_x + 3, self.frog_y - 1) or '_',
            self.get(self.frog_x - 4, self.frog_y - 1) or '_',
            self.get(self.frog_x + 4, self.frog_y - 1) or '_',
            self.get(self.frog_x - 2, self.frog_y + 1) or '_',
            self.get(self.frog_x, self.frog_y + 1) or '_',
            self.get(self.frog_x + 2, self.frog_y + 1) or '_',
            self.get(self.frog_x - 1, self.frog_y + 1) or '_',
            self.get(self.frog_x, self.frog_y + 1) or '_',
            self.get(self.frog_x + 1, self.frog_y + 1) or '_',
                   ])

    def reward(self):
        '''Returns a reward value for the state.'''

        if self.at_goal:
            return self.score
        elif self.is_done:
            return -10
        else:
            return 0


class Agent:

    def __init__(self, train=None):

        # train is either a string denoting the name of the saved
        # Q-table file, or None if running without training
        self.train = train

        # q is the dictionary representing the Q-table
        self.q = {}
        self.oldstate = None
        self.oldaction = None

        # name is the Q-table filename
        # (you likely don't need to use or change this)
        self.name = train or 'q'

        # path is the path to the Q-table file
        # (you likely don't need to use or change this)
        self.path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'train', self.name + '.json')

        self.load()

    def load(self):
        '''Loads the Q-table from the JSON file'''
        try:
            with open(self.path, 'r') as f:
                self.q = json.load(f)
            if self.train:
                print('Training {}'.format(self.path))
            else:
                print('Loaded {}'.format(self.path))
        except IOError:
            if self.train:
                print('Training {}'.format(self.path))
            else:
                raise Exception('File does not exist: {}'.format(self.path))
        return self

    def save(self):
        '''Saves the Q-table to the JSON file'''
        with open(self.path, 'w') as f:
            json.dump(self.q, f)
        return self
    
    def temporal_diff(self, oldstate, oldaction, newstate):
        discount = 0.95
        return oldstate.reward() + discount * max(self.q[newstate.key]) - self.q[oldstate.key][oldaction]
    
    def bellman(self, oldstate, oldaction, newstate):
        lr = 0.1
        return self.q[oldstate.key][oldaction] + lr * self.temporal_diff(oldstate, oldaction, newstate)

    def bestactionindex(self, state):
        if self.q.__contains__(state.key) == False:
            self.q[state.key] = [0, 0, 0, 0, 0] 
            return 4
        else:     
            array = self.q[state.key]
            max_val = max(array)
            index = array.index(max_val)
        return index

    def choice(self, state):
        epsilon = random.randint(1,10)
        if epsilon <= 2:
            action = random.choice(state.ACTIONS)
        else:
            i = self.bestactionindex(state)
            action = state.ACTIONS[i]
        return action


    def updateqt(self, state):
        if self.q.__contains__(state.key) == False:
            self.q[state.key] = [0, 0, 0, 0, 0]
        self.save()

    def choose_action(self, state_string):
        '''
        Returns the action to perform.

        This is the main method that interacts with the game interface:
        given a state string, it should return the action to be taken
        by the agent.

        The initial implementation of this method is simply a random
        choice among the possible actions. You will need to augment
        the code to implement Q-learning within the agent.
        '''
        state = Q_State(state_string)
        if self.train and self.oldstate is None:
            self.updateqt(state)
            action = self.choice(state)
            self.oldstate = state
            self.oldaction = action
            return action
        elif self.train and self.oldstate:
            if state.is_done:
                pass
            self.updateqt(state)
            action = self.choice(state)
            oldactioni = state.ACTIONS.index(self.oldaction)
            self.q[self.oldstate.key][oldactioni] = self.bellman(self.oldstate, oldactioni, state)
            self.oldstate = state
            self.oldaction = action
            return action
        else:
            if state.is_done:
                pass
            i = self.bestactionindex(state)
            action = state.ACTIONS[i]
            return action

        


        


