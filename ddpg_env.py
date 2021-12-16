import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import random
from environment.portfolio import PortfolioEnv


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        """
        :param state:
        :return:
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        :param state:
        :param action:
        :return:
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class DDPG(object):
    def __init__(self, args):
        self.args = args
        self.actor = Actor(args.state_dim, args.hidden_size, args.action_dim)
        self.actor_target = Actor(args.state_dim, args.hidden_size, args.action_dim)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic = Critic(args.state_dim + args.action_dim, args.hidden_size, args.action_dim)
        self.critic_target = Critic(args.state_dim + args.action_dim, args.hidden_size, args.action_dim)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_critic)
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        self.actor_loss = 0
        self.critic_loss = 0
        self.actor_loss_list = []
        self.critic_loss_list = []
        self.replay_buffer = ReplayBuffer(args.buffer_size)
        self.writer = SummaryWriter('./ddpg_log')

    def soft_update(self, target, source, tau):
        """
        :param target:
        :param source:
        :param tau:
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        """
        :param target:
        :param source:
        :return:
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def train(self):
        """
        :return:
        """
        if self.replay_buffer.size() > self.args.batch_size:
            state, action, next_state, reward, not_done = self.replay_buffer.sample(self.args.batch_size)
            # train critic
            with torch.no_grad():
                next_action = self.actor_target(next_state)
                q_target_next = self.critic_target(next_state, next_action)
                q_target = reward + (not_done * self.args.gamma * q_target_next)
                q_target = q_target.detach()
            self.critic_loss = F.mse_loss(self.critic(state, action), q_target)
            self.critic_optim.zero_grad()
            self.critic_loss.backward()
            self.critic_optim.step()
            # train actor
            self.actor_loss = -self.critic(state, self.actor(state)).mean()
            self.actor_optim.zero_grad()
            self.actor_loss.backward()
            self.actor_optim.step()
            # update target network
            self.soft_update(self.actor_target, self.actor, self.args.tau)
            self.soft_update(self.critic_target, self.critic, self.args.tau)
            self.actor_loss_list.append(self.actor_loss.item())
            self.critic_loss_list.append(self.critic_loss.item())
            self.writer.add_scalar('actor_loss', self.actor_loss.item(), self.args.global_step)
            self.writer.add_scalar('critic_loss', self.critic_loss.item(), self.args.global_step)
            self.args.global_step += 1

    def save(self):
        """
        :return:
        """
        torch.save(self.actor.state_dict(), os.path.join(self.args.save_dir, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(self.args.save_dir, 'critic.pth'))
        print('Models saved')

    def load(self):
        """
        :return:
        """
        self.actor.load_state_dict(torch.load(os.path.join(self.args.save_dir, 'actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(self.args.save_dir, 'critic.pth')))
        print('Models loaded')


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, next_state, reward, not_done):
        """
        :param state:
        :param action:
        :param next_state:
        :param reward:
        :param not_done:
        :return:
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, next_state, reward, not_done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        :param batch_size:
        :return:
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, not_done = map(np.stack, zip(*batch))
        return state, action, next_state, reward, not_done

    def __len__(self):
        return len(self.buffer)


def main(args):
    """
    :param args:
    :return:
    """
    env = PortfolioEnv(args.window_size, args.asset_num, args.sample_days, args.init_balance, args.trading_cost)
    ddpg = DDPG(args)
    ddpg.load()
    for i in range(args.episode):
        state = env.reset()
        episode_reward = 0
        for t in count():
            action = ddpg.actor(torch.from_numpy(state).float())
            action = action.detach().numpy()
            next_state, reward, done, info = env.step(action)
            ddpg.replay_buffer.push(state, action, next_state, reward, not done)
            state = next_state
            episode_reward += reward
            if done:
                break
        print('Episode: {}\tReward: {}'.format(i, episode_reward))
        ddpg.train()
        if i % args.save_interval == 0:
            ddpg.save()
    ddpg.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--asset_num', type=int, default=5)
    parser.add_argument('--sample_days', type=int, default=252)
    parser.add_argument('--init_balance', type=int, default=1000000)
    parser.add_argument('--trading_cost', type=float, default=0.0025)
    parser.add_argument('--episode', type=int, default=10000)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--lr_actor', type=float, default=1e-4)
    parser.add_argument('--lr_critic', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--save_dir', type=str, default='./ddpg_models')
    parser.add_argument('--global_step', type=int, default=0)
    args = parser.parse_args()
    main(args)