from email import policy
from operator import mod
import gym
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Convolution2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

env = gym.make('DemonAttack-v0')
height, width, channels = env.observation_space.shape[0]
actions = env.action_space.n #number of actions we have

"""
episodes = 5
for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = random.choice([0,1,2,3,4,5])
        n_state, reward, done, info = env.step(action)
        score += reward

    print('Episode:{} Score:{}'.format(episode,score))
"""


def build_model(height,width,channels,actions):
    model = Sequential()
    model.add(Convolution2D(32,(8,8),strides=(4,4),activation='relu',input_shape=(3,height,width,channels)))
    model.add(Convolution2D(64,(4,4),strides=(2,2),activation='relu'))
    model.add(Convolution2D(64,(3,3),activation='relu'))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(actions,activation='linear'))
    return model

def build_agent(model,actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),attr='eps',value_max=1.,value_min=.1,value_test=.2,nb_steps=10000)
    memory = SequentialMemory(limit=1000,window_length=3)
    dqn = DQNAgent(model=model,memory=memory,policy=policy,enable_dueling_network=True,dueling_type='avg',nb_actions=actions,nb_steps_warmup=10000)
    return dqn

model = build_model(height,width,channels,actions)

dqn = build_agent(model,actions)
dqn.compile(Adam(learning_rate=1e-4))
dqn.fit(env,nb_steps=10000,visualize=False,verbose=2)

env.close()

#print(env.unwrapped.get_action_meanings())