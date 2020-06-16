import os
import sys
import random
import numpy as np
from collections import deque
import torch
from torch.autograd import Variable
import torch.nn as nn

GAME = 'bird'  # the name of the  being played for log files
ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 1000. # timesteps to observe before training
# In the previous OBSERVE round, the network is not trained, only the data is collected and stored in the memory
# In the OBSERVE to OBSERVE + EXPLORE rounds, the network is trained and epsilon is annealed, gradually reducing
# epsilon to FINAL_EPSILON.
# When reaching the EXPLORE round, epsilon reaches the final value FINAL_EPSILON, no longer update it
EXPLORE = 2000000.
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.0001  # Initial value of epsilon
REPLAY_MEMORY = 50000  # Memory base
BATCH_SIZE = 32  # the number of training batch
FRAME_PER_ACTION = 1  # Every FRAME_PER_ACTION round, there will be epsilon probability to explore
UPDATE_TIME = 100  # Update the target network parameters every UPDATE_TIME
width = 80
height = 80

total_reward=0

# torch.device object used throughout this script
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Neural network structure
class DeepNetWork(nn.Module):
    def __init__(self, ):
        super(DeepNetWork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 256),
            nn.ReLU()
        )
        self.out = nn.Linear(256, 2)

    def forward(self, x):
        x=x.to(device)
        x = self.conv1(x);
        x = self.conv2(x);
        x = self.conv3(x);
        x = x.view(x.size(0), -1)
        x = self.fc1(x);
        return self.out(x)


class BrainDQNMain(object):
    def save(self):
        print("save model param")
        torch.save(self.Q_net.state_dict(), 'params3.pth')

    def load(self):
        if os.path.exists("params3.pth"):
            print("load model param")
            self.Q_net.load_state_dict(torch.load('params3.pth'))
            self.Q_netT.load_state_dict(torch.load('params3.pth'))

    def __init__(self, actions):
        # At each timestep, the transfer samples (st, at, rt, st + 1) obtained by
        # the agent interacting with the environment are stored in the playback memory,
        # Randomly take out some (minibatch) data to train when training, disrupting the correlation
        self.replayMemory = deque()  # init some parameters
        self.timeStep = 0
        # There is a probability of epsilon, choose an action randomly,
        # 1-epsilon probability selects the action through the Q (max) value output by the network
        self.epsilon = INITIAL_EPSILON
        # Initialization action
        self.actions = actions
        # Current value network
        self.Q_net = DeepNetWork().to(device)
        # Target value network
        self.Q_netT = DeepNetWork().to(device)
        # Load the trained model and continue training based on the trained model
        self.load()
        # Use mean square error as a loss function
        self.loss_func = nn.MSELoss().to(device)
        LR = 1e-6
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=LR)

    # Use minibatch to train the network
    def train(self):  # Step 1: obtain random minibatch from replay memory
        # Randomly obtain BATCH_SIZE data from the memory for training
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]  # Step 2: calculate y
        # y_batch is used to store reward
        y_batch = np.zeros([BATCH_SIZE, 1])
        nextState_batch = np.array(nextState_batch)  # print("train next state shape")
        # print(nextState_batch.shape)
        nextState_batch = torch.Tensor(nextState_batch).to(device)
        action_batch = np.array(action_batch)
        # Each action contains an array of two elements, the array must be 1, 0,
        # the index of the maximum value is the index of the action
        index = action_batch.argmax(axis=1)
        print("action " + str(index))
        index = np.reshape(index, [BATCH_SIZE, 1])
        # index of predicted action
        action_batch_tensor = torch.LongTensor(index).to(device)
        # Use the target network to predict the action of nextState_batch
        QValue_batch = self.Q_netT(nextState_batch)
        QValue_batch = QValue_batch.detach().cpu().numpy()
        # Calculate the reward of each state
        for i in range(0, BATCH_SIZE):
            # the end sign
            terminal = minibatch[i][4]
            if terminal:
                y_batch[i][0] = reward_batch[i]
            else:
                # Here QValue_batch [i] is an array, the size is the size of all action sets,
                # QValue_batch [i], represents the array of Q values that do all actions,
                # y is calculated as if the game is stopped, y = reward [i], if not stopped,
                # then y = reward [i] + gamma * np.max (Qvalue [i]) represents that the current y value
                # is the current reward + the future expected maximum value * gamma (gamma: empirical coefficient)
                # The output layer of the network has a dimension of 2, and the output value Maximum value as Q value
                y_batch[i][0] = reward_batch[i] + GAMMA * np.max(QValue_batch[i])

        y_batch = np.array(y_batch)
        y_batch = np.reshape(y_batch, [BATCH_SIZE, 1])
        state_batch_tensor = Variable(torch.Tensor(state_batch)).to(device)
        y_batch_tensor = Variable(torch.Tensor(y_batch)).to(device)
        y_predict = self.Q_net(state_batch_tensor).gather(1, action_batch_tensor)
        loss = self.loss_func(y_predict, y_batch_tensor)
        print("loss is " + str(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Every UPDATE_TIME round, update the parameters of the target network with the parameters
        # of the trained network
        if self.timeStep % UPDATE_TIME == 0:
            self.Q_netT.load_state_dict(self.Q_net.state_dict())
            self.save()
        # print('y_batch===',y_batch)
        return y_batch

    # Update the memory bank and train the network if the round meets certain requirements
    def setPerception(self, nextObservation, action, reward, terminal):  # print(nextObservation.shape)
        global total_reward
        # Each state is composed of 4 frames of images
        # nextObservation is a new frame of image, denoted as 5. currentState contains 4 frames of images [1,2,3,4],
        # then newState will become [2,3,4,5]
        newState = np.append(self.currentState[1:, :, :], nextObservation,
                             axis=0)  # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        # Save current state to memory base
        self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        # If the memory base is full, replace the earliest data that entered the memory base
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        # Before training, you need to observe the data of the OBSERVE round. After collecting the data
        # of the OBSERVE round, start training the network
        if self.timeStep > OBSERVE:  # Train the network
            total_reward = self.train()

        # print info
        state = ""
        # In the previous OBSERVE round, the network is not trained, which is
        # equivalent to filling the data in the memory replayMemory
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon)
        self.currentState = newState
        self.timeStep += 1
        return total_reward

    # Get the next action
    def getAction(self):
        currentState = torch.Tensor([self.currentState])
        # QValue is the action predicted by the network
        QValue = self.Q_net(currentState)[0]
        action = np.zeros(self.actions)
        # FRAME_PER_ACTION = 1 means every step is possible to explore
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:  # There is epsilon probability to randomly choose an action
                action_index = random.randrange(self.actions)
                print("choose random action " + str(action_index))
                action[action_index] = 1
            else:  # 1-epsilon probability select next action through neural network
                # action_index = np.argmax(QValue.detach().numpy())
                action_index = np.argmax(QValue.detach().cpu().numpy())
                print("choose qnet value action " + str(action_index))
                action[action_index] = 1
        else:
            action[0] = 1  # do nothing

        # As the number of iterations increases, gradually decrease episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        return action

    # state initialization
    def setInitState(self, observation):
        # Add a dimension, the dimension of observation is 80x80,
        # after talking about stack () operation, it becomes 4x80x80
        self.currentState = np.stack((observation, observation, observation, observation), axis=0)
        print(self.currentState.shape)

