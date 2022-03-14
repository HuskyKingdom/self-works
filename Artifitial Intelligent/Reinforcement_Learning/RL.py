import numpy as np
from sklearn import preprocessing
from torch import float32, optim
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# parameters
EPSILON = 0.9  # greedy level
GAMMA = 0.9  # decay value
Learning_Rate = 0.02
Memory_Capa = 20000
N_Iteration = 100
Batch_size = 32
EPISODES = 400

# initiallizing the virtual game env
# retriving data and parameter patterns from the env
env = gym.make('DemonAttack-v0')
height, width, channels = env.observation_space.shape
N_actions = env.action_space.n

class DQN_network(nn.Module):

    def __init__(self):

        super(DQN_network, self).__init__()
        
        self.CN_layer1 = nn.Conv2d(channels,32,(8,8),(4,4))
        self.CN_layer2 = nn.Conv2d(32,64,(4,4),(2,2))
        self.CN_layer3 = nn.Conv2d(64,64,(3,3))
        self.FT_layer4 = nn.Flatten()
        self.FC_layer5 = nn.Linear(22528,512)
        self.FC_layer6 = nn.Linear(512,256)
        self.FC_layer7 = nn.Linear(256,N_actions)
        

    def forward(self, x):
        x = self.CN_layer1(x)
        x = F.relu(x)
        x = self.CN_layer2(x)
        x = F.relu(x)
        x = self.CN_layer3(x)
        x = F.relu(x)
        x = self.FT_layer4(x)
        x = self.FC_layer5(x)
        x = F.relu(x)
        x = self.FC_layer6(x)
        x = F.relu(x)
        x = self.FC_layer7(x)

        return x



class DQN_agent():

    def __init__(self):
        self.eval_net, self.target_net = DQN_network(), DQN_network()
        # state, action ,reward and next state
        self.memory = [None for i in range(Memory_Capa)]
        self.memory_counter = 0
        self.rt_memory_counter = 0
        self.learn_counter = 0
        self.optimizer = optim.Adam(self.eval_net.parameters(), Learning_Rate)
        self.loss = nn.MSELoss()

    def img_preprocessing(self,img):

        img = torch.FloatTensor(img).permute((2,0,1)) # reorder the channel 
        img = img.unsqueeze(0) # add another dimension
        return img # final shape -> (N,C,H,W)

    def store_trans(self, state, action, reward, next_state):

        if self.rt_memory_counter % 500 == 0:
            print("The experience pool collects {} time experience".format(self.rt_memory_counter))
        
        
        
        index = self.memory_counter % Memory_Capa



        trans = (state, [action], [reward], next_state)
        self.memory[index] = trans
        self.memory_counter += 1
        self.rt_memory_counter += 1

    def choose_action(self, state):


        if np.random.randn() <= EPSILON:  # choose actions greedyly
            action_value = self.eval_net.forward(self.img_preprocessing(state))
            # get argmax action of q values
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]  # get the action index
        else:
            action = np.random.randint(0, N_actions)
        return action

    def learn(self):

        # learn 100 times then the target network update
        if self.learn_counter % N_Iteration == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter += 1

        sample_index = np.random.choice(Memory_Capa, Batch_size)

        batch_memory = []
        for i in range(len(sample_index)):
            batch_memory.append(self.memory[i])

     
        # state, action ,reward and next state
        

        batch_state = []
        batch_action = []
        batch_reward = []
        batch_next_state = []

        for i in range(len(batch_memory)):
            batch_state.append(batch_memory[i][0])
            batch_action.append(batch_memory[i][1])
            batch_reward.append(batch_memory[i][2])
            batch_next_state.append(batch_memory[i][3])


        batch_state = torch.FloatTensor(np.array(batch_state)).permute(0,3,1,2)
        batch_action = torch.LongTensor(batch_action)
        batch_reward = torch.FloatTensor(batch_reward)
        batch_next_state = torch.FloatTensor(np.array(batch_next_state)).permute(0,3,1,2)

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(Batch_size, 1)

        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



agent = DQN_agent()
print("The DQN is collecting experience...")
step_counter_list = []
for episode in range(EPISODES):
    state = env.reset()
    step_counter = 0
    while True:
        step_counter += 1
        env.render()
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        reward = reward * 100 if reward > 0 else reward * 5
        if reward > 0 :
            print("--------- Reward : ",reward)
        agent.store_trans(state, action, reward, next_state)
        if agent.memory_counter >= Memory_Capa:
            agent.learn()
            agent.memory_counter = 0
            if done:
                print("episode {}, the reward is {}".format(episode, round(reward, 3)))
        if done:
            step_counter_list.append(step_counter)
            break

        state = next_state




