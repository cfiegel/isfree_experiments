import numpy as np
import math
from numba import njit

from open_spiel.python import policy
import pyspiel

from open_spiel.python.algorithms.exploitability import nash_conv
from agents.omd import OMDBase
from agents.balanced_ftrl import BalancedFTRL
from agents.utils import sample_from_weights
from agents.utils import compute_log_sum_from_logit
from tqdm import tqdm

class ISFree(BalancedFTRL):

  def __init__(
    self,
    game,
    budget,
    base_constant=1.0,
    lr_constant=1.0,
    ix_constant=1.0,
    name=None
  ):

    OMDBase.__init__(
      self,
      game,
      budget,
      base_constant=base_constant,
      lr_constant=lr_constant,
      ix_constant=ix_constant,
      )
    
    
    self.name = 'ISFree'
    if name:
      self.name = name
    
    #Balanced policy
    self.compute_balanced()
    
    #Set rates
    self.learning_rates = self.base_learning_rate * np.ones(self.policy_shape[0])
    self.n_visit=np.zeros(self.policy_shape[0])
    
    #Set policy
    self.current_policy.action_probability_array=self.balanced_policy.copy()
    self.sampling_policy=self.balanced_policy.copy()
    
    self.current_logit=np.log(self.current_policy.action_probability_array,where=self.legal_actions_indicator)
    self.initial_logit=self.current_logit.copy()
    
  def update_sampling_policy(self):
    '''Update the sampling policy to something more suited for actual learning.
    ''' 
    self.sampling_policy = self.cumulative_plan+self.eps*self.legal_actions_mask
    self.sampling_policy /= self.average_policy.action_probability_array.sum(axis=-1, keepdims=True)

  def sample_action_from_idx_from_sampling(self, state_idx, return_idx=False):
    '''Sample an action from the current policy at a state.
    '''
    probs = self.sampling_policy[state_idx,:]
    action_idx = sample_from_weights(list(range(probs.shape[0])), probs)
    action=action_idx
    if return_idx:
      return action, action_idx
    return action
    
  def sample_trajectory(self, step):
    plans =  np.ones(self.num_players)
    cum_plans = np.ones(self.num_players)*(step+1.0)
    trajectory = []
    state = self.game.new_initial_state()
    self.current_learning_player=step%self.num_players
    while not state.is_terminal():
      if state.is_chance_node():
        #Chance state
        outcomes_with_probs = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes_with_probs)
        action = sample_from_weights(action_list, prob_list)
        state.apply_action(action)
      else:
        #Current state
        current_player = state.current_player() 
        state_idx = self.state_index(state)
        #Sample action
        if self.current_learning_player==current_player:
          action, action_idx = self.sample_action_from_idx_from_sampling(state_idx, return_idx=True)
          #action, action_idx = self.sample_action_from_idx(state_idx, return_idx=True)
          #Update cumulative plans
          policy = self.get_current_policy(state_idx)
          self.cumulative_plan[state_idx,:] += (cum_plans[current_player]-self.cumulative_plan[state_idx,:].sum())* policy
          cum_plans[current_player] = self.cumulative_plan[state_idx, action_idx]
          #Update plans
          plans[current_player] *= policy[action_idx]
          #Record transition
          transition = {
            'player': current_player,
            'state_idx': state_idx,
            'action_idx': action_idx,
            'plan': plans[current_player],
            'loss': 0.0
          }
          trajectory += [transition]
        else:
          action, action_idx = self.sample_action_from_idx(state_idx, return_idx=True)
        #Apply action
        state.apply_action(action)

    #Compute loss
    losses = self.reward_to_loss(np.asarray(state.returns()))
    trajectory[-1]['loss'] = losses[trajectory[-1]['player']] 

    return trajectory

  def update(self, trajectory):

    #Initialize values
    value =  0

    for transition in reversed(trajectory):
        player, state_idx, action_idx, plan, loss = transition.values()
      
        policy = self.current_policy.action_probability_array[state_idx,:]  
      
        #Update lr:
        self.n_visit[state_idx]+=1
        lr=self.learning_rates[state_idx]/math.sqrt(self.n_visit[state_idx])
        alpha=math.sqrt(1-1/self.n_visit[state_idx])
  
        #Compute new policy 
        legal_actions=self.legal_actions_indicator[state_idx,:]
        adjusted_loss = (loss - value)/self.sampling_policy[state_idx,action_idx]
        #adjusted_loss = (loss - value)/policy[action_idx]
        self.current_logit[state_idx,:]=alpha*self.current_logit[state_idx,:]+(1-alpha)*self.initial_logit[state_idx,:]
        self.current_logit[state_idx,action_idx]-=lr*adjusted_loss
        logz=compute_log_sum_from_logit(self.current_logit[state_idx,:],legal_actions)
        self.current_logit[state_idx,:]-=logz*legal_actions
        value = logz/lr
        new_policy=np.exp(self.current_logit[state_idx,:],where=legal_actions)*legal_actions
  
        #Update new policy 
        self.set_current_policy(state_idx, new_policy)