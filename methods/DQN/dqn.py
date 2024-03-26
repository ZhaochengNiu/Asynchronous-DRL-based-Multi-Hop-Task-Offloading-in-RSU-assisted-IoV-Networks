#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:50:49
@LastEditor: John
LastEditTime: 2022-07-13 00:08:18
@Discription: 
@Environment: python 3.7.7
'''
'''off-policy
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np

class MLP(nn.Module):
    def __init__(self, n_states,n_actions,hidden_dim=128):
        """ 初始化q网络，为全连接网络
            n_states: 输入的特征数即环境的状态维度
            n_actions: 输出的动作维度
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim) # 输入层
        self.fc2 = nn.Linear(hidden_dim,hidden_dim) # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, n_actions) # 输出层
        
    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        # 初始化经验回放的容量、缓冲区和位置
        self.capacity = capacity  # 经验回放的容量
        self.buffer = []  # 缓冲区
        self.position = 0  # 初始化位置变量，表示下一个转移应该存储的位置。
    
    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        # 将一个转移存入缓冲区。
        # 参数 state、action、reward、next_state 和 done 分别表示当前状态、动作、奖励、下一个状态和完成标志。
        # 如果缓冲区未满，则直接添加到缓冲区末尾；如果缓冲区已满，则覆盖最早的转移。
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # 如果缓冲区未满，添加空位
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity  # 更新位置，实现循环缓冲区的效果
    
    def sample(self, batch_size):
        # 使用 random.sample 方法从缓冲区中随机采样指定数量的转移。
        # 随机采出小批量转移
        batch = random.sample(self.buffer, batch_size)
        # 将采样得到的转移解压成状态、动作、奖励、下一个状态和完成标志。
        state, action, reward, next_state, done = zip(*batch)
        #  返回解压后的转移。
        return state, action, reward, next_state, done
    
    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)

class DQN:
    def __init__(self, n_states,n_actions,cfg):

        self.n_actions = n_actions  # 存储动作空间的大小，即可供智能体选择的动作数量。
        self.device = cfg.device  # 存储设备信息，如 CPU 或 GPU，用于指定模型在哪个设备上运行。
        self.gamma = cfg.gamma  # 存储折扣因子，用于平衡当前奖励和未来奖励的重要性。
        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数 初始化 ε-greedy 策略的衰减计数器为 0。
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
            (cfg.epsilon_start - cfg.epsilon_end) * \
            math.exp(-1. * frame_idx / cfg.epsilon_decay) # 定义了一个匿名函数 epsilon，用于计算 ε-greedy 策略中的 ε 值。
        # cfg.epsilon_end、cfg.epsilon_start 和 cfg.epsilon_decay
        # 是从配置对象 cfg 中获取的参数，分别表示 ε 的最终值、初始值和衰减率。
        self.batch_size = cfg.batch_size # 存储批量大小，即每次从经验回放缓冲区中采样的样本数量。
        # 创建策略网络，该网络使用 MLP (多层感知器) 模型，
        # 输入大小为状态空间大小 n_states，输出大小为动作空间大小 n_actions，并将其发送到指定的设备上。
        self.policy_net = MLP(n_states,n_actions).to(self.device)
        # 创建目标网络，与策略网络具有相同的架构，并将其发送到指定的设备上。
        self.target_net = MLP(n_states,n_actions).to(self.device)
        #  遍历目标网络和策略网络的参数，并将策略网络的参数复制到目标网络中，以确保它们具有相同的初始参数。
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()): # 复制参数到目标网路targe_net
            target_param.data.copy_(param.data)
        # 初始化优化器，使用 Adam 算法来优化策略网络的参数，学习率为配置对象 cfg 中的 lr 参数。
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)  # 优化器
        self.memory = ReplayBuffer(cfg.memory_capacity)  # 创建经验回放缓冲区，用于存储智能体与环境交互的经验转移数据。

    def choose_action(self, state):
        ''' 选择动作
        '''
        # 增加帧索引。这可能跟踪代理在环境中走过的步数或帧数。
        self.frame_idx += 1
        # 这个条件决定了是探索还是利用。如果随机生成的数大于epsilon值（控制探索-利用权衡的参数），
        # 代理根据当前策略选择具有最高Q值的动作。否则，它会探索，选择一个随机动作。
        if random.random() > self.epsilon(self.frame_idx):
            # 这是PyTorch的上下文管理器，用于在推断期间禁用梯度计算。
            # 在此处使用它可以节省内存和计算资源，因为在动作选择过程中不需要梯度。
            with torch.no_grad():
                # 将输入状态转换为PyTorch张量，发送到指定设备（如GPU），
                # 并添加一个批处理维度。这准备好将状态输入神经网络模型。
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                # 将状态张量传递给神经网络（policy_net），该网络输出当前状态下每个动作的Q值。
                q_values = self.policy_net(state)
                # 选择具有最高Q值的动作。q_values.max(1)返回最大Q值及其对应的索引。
                # [1]提取索引，并.item()将其转换为表示动作的Python标量。
                action = q_values.max(1)[1].item() # 选择Q值最大的动作
        else:
            # 如果随机数小于或等于epsilon，则执行此分支，并从可用动作中随机选择一个动作。
            action = random.randrange(self.n_actions)
        return action

    def update(self):
        # 这一行检查记忆库中存储的经验样本数量是否已经达到了一个批量的大小（self.batch_size）。
        # 如果没有，就退出该方法，因为没有足够的数据来进行训练。
        if len(self.memory) < self.batch_size: # 当memory中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        # print('updating')
        # 这一行从记忆库中随机采样一个批量大小的经验样本，包括当前状态、动作、奖励、下一个状态和完成标志。
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        # 这几行将采样得到的经验样本转换为 PyTorch 张量，并将其发送到指定的设备（例如 GPU）。
        # 这是为了准备将这些数据送入神经网络进行训练。
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)  
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)  
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)
        # 这一行计算当前状态动作对应的 Q 值。self.policy_net(state_batch) 使用当前状态批量计算 Q 值，
        # 然后 .gather(dim=1, index=action_batch) 根据动作批量的索引获取对应的 Q 值。
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch) # 计算当前状态(s_t,a)对应的Q(s_t, a)
        # 这一行计算下一个状态的 Q 值。self.target_net(next_state_batch) 使用下一个状态批量计算 Q 值，
        # 然后 .max(1) 返回每个样本中最大 Q 值及其索引，[0] 获取最大 Q 值，.detach() 用于将其从计算图中分离。
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach() # 计算下一时刻的状态(s_t_,a)对应的Q值
        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        # 这一行计算预期的 Q 值，根据 Q-Learning 的更新规则，将奖励、下一个状态的 Q 值以及完成标志（是否为终止状态）考虑进去。
        expected_q_values = reward_batch + self.gamma * next_q_values * (1-done_batch)
        # 这一行计算损失，使用均方误差损失函数，衡量预测的 Q 值和目标 Q 值之间的差异。
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算均方根损失
        # 优化更新模型
        # 这两行分别执行优化器的零梯度清除和反向传播操作，用于计算损失相对于模型参数的梯度。
        self.optimizer.zero_grad()  
        loss.backward()
        # 这一行对梯度进行裁剪，防止梯度爆炸。
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        # 这一行执行一步优化器更新模型参数，以减小损失。
        self.optimizer.step() 

    def save(self, path):
        torch.save(self.target_net.state_dict(), path+'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path+'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)
