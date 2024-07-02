from dqn import DQN
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple

import math
import random
import numpy as np
from utils import *
from itertools import count, chain

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward', 'next_cand'))

class Agent(object):
    def __init__(self, device, memory, state_size, action_size, hidden_size, gcn_net, learning_rate, l2_norm, PADDING_ID, EPS_START = 0.9, EPS_END = 0.1, EPS_DECAY = 0.0001, tau=0.01):
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.steps_done = 0
        self.device = device
        self.gcn_net = gcn_net
        self.policy_net = DQN(state_size, action_size, hidden_size).to(device)
        self.target_net = DQN(state_size, action_size, hidden_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(chain(self.policy_net.parameters(),self.gcn_net.parameters()), lr=learning_rate, weight_decay = l2_norm)
        self.memory = memory
        self.loss_func = nn.MSELoss()
        self.PADDING_ID = PADDING_ID
        self.tau = tau

    def select_action(self, state, cand, action_space, is_test=False, is_last_turn=False):
        #action_space[0]=feature, action_space[1]=item
        state_emb = self.gcn_net([state])
        cand = torch.LongTensor([cand]).to(self.device)
        cand_emb = self.gcn_net.embedding(cand)
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if is_test or sample > eps_threshold:
            if is_test and (len(action_space[1]) <= 10 or is_last_turn):
                return torch.tensor(action_space[1][0], device=self.device, dtype=torch.long), action_space[1]
            with torch.no_grad():
                actions_value = self.policy_net(state_emb, cand_emb)
                # print(sorted(list(zip(cand[0].tolist(), actions_value[0].tolist())), key=lambda x: x[1], reverse=True))
                action = cand[0][actions_value.argmax().item()]
                sorted_actions = cand[0][actions_value.sort(1, True)[1].tolist()]
                return action, sorted_actions.tolist()
        else:
            # print('Number of action space', len(action_space))
            shuffled_cand = action_space[0]+action_space[1] #feature + item
            # random.shuffle(shuffled_cand)
            return torch.tensor(shuffled_cand[0], device=self.device, dtype=torch.long), shuffled_cand
    
    def update_target_model(self):
        #soft assign
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + target_param.data * (1.0 - self.tau))

    def optimize_model(self, BATCH_SIZE, GAMMA):
        if len(self.memory) < BATCH_SIZE:
            return
        
        self.update_target_model()
        
        idxs, transitions, is_weights = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_emb_batch = self.gcn_net(list(batch.state))

        # Convert each tensor in the batch to a NumPy array after moving it to the CPU
        action_batch_cpu = [tensor.cpu().numpy() for tensor in batch.action]
        action_batch_np = np.array(action_batch_cpu).astype(int).reshape(-1, 1)
        # Convert the NumPy array back to a tensor and move it to the appropriate device
        action_batch = torch.LongTensor(action_batch_np).to(self.device)
        # action_batch = torch.LongTensor(np.array(batch.action).astype(int).reshape(-1, 1)).to(self.device) #[N*1]
        action_emb_batch = self.gcn_net.embedding(action_batch)

        # Convert each tensor in batch.reward to a NumPy array after moving it to the CPU
        reward_batch_cpu = [tensor.cpu().numpy() for tensor in batch.reward]
        reward_batch_np = np.array(reward_batch_cpu).astype(float).reshape(-1, 1)
        
        # Convert the NumPy array back to a tensor and move it to the appropriate device
        reward_batch = torch.FloatTensor(reward_batch_np).to(self.device)
        # reward_batch = torch.FloatTensor(np.array(batch.reward).astype(float).reshape(-1, 1)).to(self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        n_states = []
        n_cands = []
        for s, c in zip(batch.next_state, batch.next_cand):
            if s is not None:
                n_states.append(s)
                n_cands.append(c)
        next_state_emb_batch = self.gcn_net(n_states)
        next_cand_batch = self.padding(n_cands)
        next_cand_emb_batch = self.gcn_net.embedding(next_cand_batch)

        q_eval = self.policy_net(state_emb_batch, action_emb_batch, choose_action=False)

        # Double DQN
        best_actions = torch.gather(input=next_cand_batch, dim=1, index=self.policy_net(next_state_emb_batch, next_cand_emb_batch).argmax(dim=1).view(len(n_states),1).to(self.device))
        best_actions_emb = self.gcn_net.embedding(best_actions)
        q_target = torch.zeros((BATCH_SIZE,1), device=self.device)
        q_target[non_final_mask] = self.target_net(next_state_emb_batch,best_actions_emb,choose_action=False).detach()
        q_target = reward_batch + GAMMA * q_target

        # prioritized experience replay
        errors = (q_eval - q_target).detach().cpu().squeeze().tolist()
        self.memory.update(idxs, errors)

        # mean squared error loss to minimize
        loss = (torch.FloatTensor(is_weights).to(self.device) * self.loss_func(q_eval, q_target)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.data
    
    def save_model(self, data_name, filename, epoch_user):
        save_rl_agent(
            dataset=data_name, 
            model={'policy': self.policy_net.state_dict(), 'gcn': self.gcn_net.state_dict()}, 
            filename=filename, 
            epoch_user=epoch_user
            )
    def load_model(self, data_name, filename, epoch_user):
        model_dict = load_rl_agent(dataset=data_name, filename=filename, epoch_user=epoch_user, device=self.device)
        self.policy_net.load_state_dict(model_dict['policy'])
        self.gcn_net.load_state_dict(model_dict['gcn'])
    
    def padding(self, cand):
        pad_size = max([len(c) for c in cand])
        padded_cand = []
        for c in cand:
            cur_size = len(c)
            new_c = np.ones((pad_size)) * self.PADDING_ID
            new_c[:cur_size] = c
            padded_cand.append(new_c)
        return torch.LongTensor(padded_cand).to(self.device)