import os
import math
import pdb

import torch
import pickle
import logging
import numpy as np
import torch.nn as nn
from torch.nn import init, Parameter
import torch.nn.functional as F
import torch.optim as optimizer
from torch.autograd import Variable

from .ouprocess import OUProcess
from .prioritized_replay_memory import PrioritizedReplayMemory
# from autotune.knobs import logger


# code from https://github.com/Kaixhin/NoisyNet-A3C/blob/master/model.py
class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.05, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
        # µ^w and µ^b reuse self.weight and self.bias
        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
        self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
            init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.constant(self.sigma_weight, self.sigma_init)
            init.constant(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), self.bias + self.sigma_bias * Variable(self.epsilon_bias))

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)


class Normalizer(object):

    def __init__(self, mean, variance):
        if isinstance(mean, list):
            mean = np.array(mean)
        if isinstance(variance, list):
            variance = np.array(variance)

        # Fix: Sanitizar mean y variance
        self.mean = np.nan_to_num(mean, nan=0.0)
        variance = np.nan_to_num(variance, nan=1.0)
        variance[variance < 1e-6] = 1.0  # Evitar división por 0
        self.std = np.sqrt(variance + 0.00001)

    def update(self, mean, variance):
        if isinstance(mean, list):
            mean = np.array(mean)
        if isinstance(variance, list):
            variance = np.array(variance)
        
        self.mean = np.nan_to_num(mean, nan=0.0)
        variance = np.nan_to_num(variance, nan=1.0)
        variance[variance < 1e-6] = 1.0
        self.std = np.sqrt(variance + 0.00001)

    def normalize(self, x):
        if isinstance(x, list):
            x = np.array(x)
        x = x - self.mean
        x = x / self.std
        return Variable(torch.FloatTensor(x))

    def __call__(self, inputs, inverse=False):
        # IMPORTANTE: Manejar tanto numpy arrays como Tensors
        is_tensor = torch.is_tensor(inputs)
        is_list = isinstance(inputs, list)

        # Convertir a numpy si es necesario
        if is_list:
            inputs = np.array(inputs)
        if is_tensor:
            device = inputs.device
            inputs = inputs.cpu().detach().numpy()

        # Normalización
        if inverse:
            result = inputs * self.std + self.mean
        else:
            result = (inputs - self.mean) / self.std
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        # Devolver como Tensor SIEMPRE (el Actor lo necesita)
        return torch.FloatTensor(result)

class Actor(nn.Module):

    def __init__(self, n_states, n_actions, noisy=False):
        super(Actor, self).__init__()
        # Edge-DDPG: Boosted architecture + LayerNorm (Stability)
        # LayerNorm is superior to BatchNorm for RL with small batches
        self.layers = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.LayerNorm(32),
            nn.Linear(32, n_actions)
        )

        self.act = nn.Sigmoid()
        self._init_weights()    

    def _init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)

    def sample_noise(self):
        self.out.sample_noise()

    def forward(self, states):
        actions = self.act(self.layers(states))
        return actions


class Critic(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Critic, self).__init__()
        self.act = nn.Tanh()
        
        # Edge-DDPG: Boosted architecture + LayerNorm
        self.state_input = nn.Linear(n_states, 128)
        self.action_input = nn.Linear(n_actions, 128)
        self.layers = nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.LayerNorm(64),
            nn.Linear(64, 1)
        )
        self._init_weights()

    def _init_weights(self):
        self.state_input.weight.data.normal_(0.0, 1e-2)
        self.state_input.bias.data.uniform_(-0.1, 0.1)

        self.action_input.weight.data.normal_(0.0, 1e-2)
        self.action_input.bias.data.uniform_(-0.1, 0.1)

        for m in self.layers:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)

    def forward(self, states, actions):
        states = self.act(self.state_input(states))
        actions = self.act(self.action_input(actions))

        _input = torch.cat([states, actions], dim=1)
        value = self.layers(_input)
        return value


class DDPG(object):

    def __init__(self, n_states, n_actions, opt, mean=None, var=None, ouprocess=True, supervised=False, debug=False):
        """ DDPG Algorithms
        Args:
            n_states: int, dimension of states
            n_actions: int, dimension of actions
            opt: dict, params
            supervised, bool, pre-train the actor with supervised learning
        """
        self.n_states = n_states
        self.n_actions = n_actions

        # Params
        self.alr = opt['alr']
        self.clr = opt['clr']
        self.model_name = opt['model']
        self.batch_size = opt['batch_size']
        self.gamma = opt['gamma']
        self.tau = opt['tau']
        self.ouprocess = ouprocess
        self.debug=debug
        self.logger = logging.getLogger(self.__class__.__name__)

        # Edge-DDPG: Optimized for resource-constrained fog computing
        self.logger.info("=== Edge-DDPG Configuration ===")
        self.logger.info(f"Actor: {n_states}→128→64→32→{n_actions}")
        self.logger.info(f"Critic: ({n_states}+{n_actions})→256→64→1")
        self.logger.info(f"Memory size: {opt['memory_size']}")
        self.logger.info(f"Batch size: {opt['batch_size']}")

        if mean is None:
            mean = np.zeros(n_states)
        if var is None:
            var = np.zeros(n_states)

        self.normalizer = Normalizer(mean, var)

        if supervised:
            self._build_actor()
            self.logger.info("Supervised Learning Initialized")
        else:
            # Build Network
            self._build_network()
            self.logger.info('Finish Initializing Networks')

        self.replay_memory = PrioritizedReplayMemory(capacity=opt['memory_size'])
        self.noise = OUProcess(n_actions)
        self.debug = debug
        self.debug = debug
        self.logger.info('DDPG Initialzed!')

    def update_normalization(self, mean, var):
        self.normalizer.update(mean, var)

    @staticmethod
    def totensor(x):
        # Fix: Convert list to numpy array first to avoid PyTorch slow-conversion warning
        if isinstance(x, list):
            x = np.array(x)
        return Variable(torch.FloatTensor(x))

    def _build_actor(self):
        if self.ouprocess:
            noisy = False
        else:
            noisy = True
        self.actor = Actor(self.n_states, self.n_actions, noisy=noisy)
        self.actor_criterion = nn.MSELoss()
        self.actor_optimizer = optimizer.Adam(lr=self.alr, params=self.actor.parameters())

    def _build_network(self):
        if self.ouprocess:
            noisy = False
        else:
            noisy = True
        self.actor = Actor(self.n_states, self.n_actions, noisy=noisy)
        self.target_actor = Actor(self.n_states, self.n_actions)
        self.critic = Critic(self.n_states, self.n_actions)
        self.target_critic = Critic(self.n_states, self.n_actions)

        # if model params are provided, load them
        if len(self.model_name):
            self.load_model(model_name=self.model_name)
            self.logger.info("Loading model from file: {}".format(self.model_name))

        # Copy actor's parameters
        DDPG._update_target(self.target_actor, self.actor, tau=1.0)

        # Copy critic's parameters
        DDPG._update_target(self.target_critic, self.critic, tau=1.0)

        self.loss_criterion = nn.MSELoss()
        self.actor_optimizer = optimizer.Adam(lr=self.alr, params=self.actor.parameters(), weight_decay=1e-5)
        self.critic_optimizer = optimizer.Adam(lr=self.clr, params=self.critic.parameters(), weight_decay=1e-5)

    @staticmethod
    def _update_target(target, source, tau):
        for (target_param, param) in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1 - tau) + param.data * tau
            )

    def reset(self, sigma):
        self.noise.reset(sigma)

    def _sample_batch(self):
        batch, idx = self.replay_memory.sample(self.batch_size)
        # batch = self.replay_memory.sample(self.batch_size)
        states = list(map(lambda x: x[0], batch))
        actions = list(map(lambda x: x[1].tolist(), batch))
        rewards = list(map(lambda x: x[2], batch))
        next_states = list(map(lambda x: x[3], batch))
        terminates = list(map(lambda x: x[4], batch))

        return idx, states, next_states, actions, rewards, terminates

    def _resolve_reward(self, reward_data):
        """Resolve reward_data to a scalar, re-labeling with current baseline if needed.
        reward_data can be:
          - float: legacy scalar reward (used as-is)
          - (tps_raw, penalty_sum): re-label with self.baseline_tps
        """
        if isinstance(reward_data, tuple):
            tps_raw, penalty_sum = reward_data
            return (tps_raw / self.baseline_tps) * 100.0 + penalty_sum
        return float(reward_data)

    def add_sample(self, state, action, reward_data, next_state, terminate):
        self.critic.eval()
        self.actor.eval()
        self.target_critic.eval()
        self.target_actor.eval()
        batch_state = self.normalizer([state])
        batch_next_state = self.normalizer([next_state])
        current_value = self.critic(batch_state, DDPG.totensor([action.tolist()]))
        target_action = self.target_actor(batch_next_state)
        reward_scalar = self._resolve_reward(reward_data)
        target_value = DDPG.totensor([reward_scalar]) \
            + DDPG.totensor([0 if x else 1 for x in [terminate]]) \
            * self.target_critic(batch_next_state, target_action) * self.gamma
        error = float(torch.abs(current_value - target_value).data.numpy()[0])

        self.target_actor.train()
        self.actor.train()
        self.critic.train()
        self.target_critic.train()
        self.replay_memory.add(error, (state, action, reward_data, next_state, terminate))

    def delta_action_value(self, action0, action1, current_value0, current_value1):
        delta_action = abs(action1 - action0).sum(axis=0)
        delta_action=delta_action / delta_action.shape[0]
        delta_value=((current_value1 - current_value0) * (current_value1 - current_value0)).mean()
        self.logger.info('[ddpg] delta_action: {}'.format(delta_action))
        self.logger.info('[ddpg] critic_MSE: {}'.format(delta_value))
        return delta_action,delta_value

    def update(self, elite_sample=None, phase_step=0, bc_lambda=0.3):
        """ Update the Actor and Critic with a batch data
        """
        idxs, states, next_states, actions, rewards, terminates = self._sample_batch()
        batch_states = self.normalizer(states)
        batch_next_states = self.normalizer(next_states)
        batch_actions = DDPG.totensor(actions)
        # Re-label rewards with current baseline_tps (PopArt-style non-stationary correction).
        # Transitions storing (tps_raw, penalty_sum) are recalculated; legacy scalar rewards pass through.
        rewards_resolved = [self._resolve_reward(r) for r in rewards]
        batch_rewards = DDPG.totensor(rewards_resolved)

        mask = [0 if x else 1 for x in terminates]
        mask = DDPG.totensor(mask)
        if self.debug:
            action0= self.target_actor(batch_states).detach().data.numpy()
            current_value0 = self.target_critic(batch_states, batch_actions).data.numpy()

        # Target nets must be in eval() to disable Dropout — stochastic targets add variance to TD-error.
        self.target_actor.eval()
        self.target_critic.eval()
        target_next_actions = self.target_actor(batch_next_states).detach()
        target_next_value = self.target_critic(batch_next_states, target_next_actions).detach().squeeze(1)

        current_value = self.critic(batch_states, batch_actions)
        # Clamp target Q to preserve the penalty hierarchy:
        #   swap=-1000 > mem=-500*alpha > cpu (small) > normal=[50..130]
        # Upper bound 200 blocks runaway positive Q. Lower bound -1100 lets the
        # Critic distinguish "swap config" from "memory at 100%" — losing that
        # distinction made the swap hard-limit ineffective with the old -200 bound.
        target_next_value = torch.clamp(target_next_value, -1500.0, 500.0)
        next_value = batch_rewards + mask * target_next_value * self.gamma

        # update prioritized memory
        error = torch.abs(current_value - next_value).data.numpy()
        for i in range(self.batch_size):
            idx = idxs[i]
            self.replay_memory.update(idx, error[i][0])

        # Update Critic
        loss = self.loss_criterion(current_value.squeeze(1), next_value)
        self.critic_optimizer.zero_grad()
        loss.backward()
        # Raised from 0.5 to 1.0: with reward range [-1100..200] (swap=-1000, mem=-500*alpha),
        # 0.5 was too aggressive and slowed Critic convergence. The clamp on target_next_value
        # already blocks true outliers; 1.0 lets the Critic learn from large penalties properly.
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Update Actor
        # Both critic and actor in eval() to disable Dropout during policy gradient:
        # - Critic.eval(): Q-values used as targets must be deterministic
        # - Actor.eval(): with batch_size=8, Dropout(0.3) adds too much stochastic noise
        #   to gradients, harming convergence. Entropy regularization already provides
        #   sufficient exploration incentive without Dropout noise.
        self.critic.eval()
        self.actor.eval()
        actor_out = self.actor(batch_states)
        policy_loss = -self.critic(batch_states, actor_out).mean()
        # Entropy regularization: penalize deterministic outputs (Sigmoid near 0 or 1).
        # Coefficient decays exponentially so early training stays exploratory but
        # later steps can converge (fixes C4 90-96% entropy plateau in v3).
        # floor=0.005 ensures a minimal exploration incentive is always present.
        # phase_step (resets each tier) — not the cumulative global step — so the
        # exploration ramp restarts per tier instead of staying pinned at the floor.
        entropy = -(actor_out * torch.log(actor_out + 1e-8) +
                    (1 - actor_out) * torch.log(1 - actor_out + 1e-8))
        entropy_coef = max(0.005, 0.02 * (0.995 ** phase_step))
        policy_loss = policy_loss - entropy_coef * entropy.mean()
        
        # Guided Guided Exploration (Behavior Cloning)
        # If we have an elite sample (known high-reward config), anchor the actor to it.
        # This prevents catastrophic forgetting/drift.
        if elite_sample is not None:
             elite_state = self.normalizer([elite_sample[0]])
             elite_action_target = DDPG.totensor([elite_sample[1]])
             # Clamp target to the Actor's feasible output range (Sigmoid clips to [0.02,0.98]).
             # Without this, targets at exactly 0.0 or 1.0 push the last linear layer toward
             # saturation, causing large weights and entropy collapse.
             elite_action_target = torch.clamp(elite_action_target, 0.02, 0.98)

             # Calculate what the actor would do in that elite state
             actor_action_pred = self.actor(elite_state)

             # MSE Loss: Force actor to mimic the elite action for that state
             bc_loss = nn.MSELoss()(actor_action_pred, elite_action_target)

             # BC active: lambda decays by cycle (v6: C1=0.5, C2=0.3, C3+=0.1).
             policy_loss = policy_loss + bc_lambda * bc_loss
             self.logger.info(f'[BC] Added BC Loss: {bc_loss.item():.4f}')

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  # looser than Critic (0.5) — Actor sees no outlier rewards
        self.actor_optimizer.step()
        self.critic.train()
        self.actor.train()

        DDPG._update_target(self.target_critic, self.critic, tau=self.tau)
        DDPG._update_target(self.target_actor, self.actor, tau=self.tau)

        if self.debug:    
            action1 = self.target_actor(batch_states).detach().data.numpy()
            current_value1 = self.target_critic(batch_states, batch_actions).data.numpy()
            self.delta_action_value(action0,action1,current_value0,current_value1)

        return loss.item(), policy_loss.item()

    def choose_action(self, x, coff=1):
        """ Select Action according to the current state
        Args:
            x: np.array, current state
        """
        self.actor.eval()
        act = self.actor(self.normalizer([x])).squeeze(0)
        self.actor.train()
        action = act.data.numpy()
        # self.logger.info('Action before OUProcess: {}'.format(action))
        if self.ouprocess:
            action += self.noise.noise()  * coff
        return action.clip(0.02, 0.98)

    def sample_noise(self):
        self.actor.sample_noise()

    def load_model(self, model_name):
        """ Load Torch Model from files
        Args:
            model_name: str, model path
        """
        try:
            actor_path = '{}_actor.pth'.format(model_name)
            critic_path = '{}_critic.pth'.format(model_name)
            
            if os.path.exists(actor_path):
                self.actor.load_state_dict(torch.load(actor_path))
                self.logger.info(f"Loaded Actor from {actor_path}")
            else:
                self.logger.warning(f"Actor model not found at {actor_path}. Starting fresh.")
                
            if os.path.exists(critic_path):
                self.critic.load_state_dict(torch.load(critic_path))
                self.logger.info(f"Loaded Critic from {critic_path}")
            else:
                self.logger.warning(f"Critic model not found at {critic_path}. Starting fresh.")
                
        except Exception as e:
            self.logger.warning(f"Failed to load model {model_name}: {e}")

    def save_model(self, model_dir, title):

        import os
        os.makedirs(model_dir, exist_ok=True)
        """ Save Torch Model from files
        Args:
            model_dir: str, model dir
            title: str, model name
        """
        torch.save(
            self.actor.state_dict(),
            '{}/{}_actor.pth'.format(model_dir, title)
        )

        torch.save(
            self.critic.state_dict(),
            '{}/{}_critic.pth'.format(model_dir, title)
        )

    def save_actor(self, path):
        """ save actor network
        Args:
             path, str, path to save
        """
        torch.save(
            self.actor.state_dict(),
            path
        )

    def load_actor(self, path):
        """ load actor network
        Args:
             path, str, path to load
        """
        self.actor.load_state_dict(
            torch.load(path)
        )

    def train_actor(self, batch_data, is_train=True):
        """ Train the actor separately with data
        Args:
            batch_data: tuple, (states, actions)
            is_train: bool
        Return:
            _loss: float, training loss
        """
        states, action = batch_data

        if is_train:
            self.actor.train()
            pred = self.actor(self.normalizer(states))
            action = DDPG.totensor(action)

            _loss = self.actor_criterion(pred, action)

            self.actor_optimizer.zero_grad()
            _loss.backward()
            self.actor_optimizer.step()

        else:
            self.actor.eval()
            pred = self.actor(self.normalizer(states))
            action = DDPG.totensor(action)
            _loss = self.actor_criterion(pred, action)

        return _loss.item()
