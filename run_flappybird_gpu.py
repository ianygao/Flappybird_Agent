import sys

import cv2
import numpy as np
from dqn_network_gpu import BrainDQNMain
import matplotlib.pyplot as plt
import time
sys.path.append("game/")
import wrapped_flappy_bird as game

#frequency of logging
log_freq=1000
timestep=0
#learning curve array
learning_curve=[]
#Total time steps to be finished
# TOTAL=1e6
TOTAL=100000


# Process a color image into a black and white binary image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (1, 80, 80))


def run():
    start =time.clock()
    global timestep
    actions = 2  # Number of actions
    brain = BrainDQNMain(actions)
    flappyBird = game.GameState()
    action0 = np.array([1, 0])  # random action
    # Perform an action to get the next frame of image, reward, and whether the game
    # is terminated after the action is performed
    observation0, reward0, terminal = flappyBird.frame_step(action0)
    # Convert color image to gray value image
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    # Convert grayscale image to binary image
    ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
    # Repeat a frame of pictures 4 times, each picture is a channel, becomes the input
    # with 4 channels, that is, the initial input is the same picture of 4 frames
    brain.setInitState(observation0)
    print(brain.currentState.shape)

    while 1 != 0:
        global learning_curve
        global round_reward
        # Get the next action
        action = brain.getAction()
        # Perform the action, get the next frame of the image after the action, reward, whether the game is terminated
        nextObservation, reward, terminal = flappyBird.frame_step(action)
        # Process a color image into a black and white binary image
        nextObservation = preprocess(nextObservation)
        # print(nextObservation.shape)
        round_reward = brain.setPerception(nextObservation, action, reward, terminal)
        mean_reward = np.mean(round_reward)
        # print('action=====', action, 'round_reward=====', mean_reward)
        # avgQueue.append(round_reward)
        if timestep%log_freq==0:
            learning_curve.append(mean_reward)
            print('************************',timestep, mean_reward)
        timestep+=1
        if timestep >= TOTAL:
            break

    end = time.clock()
    print('Running time: %s Seconds'%(end-start))

    #绘制学习曲线
    X=np.arange(0,len(learning_curve))
    X*=log_freq
    plt.title('Learning Curve')
    plt.xlabel('Time Step')
    plt.ylabel('Avg Reward')
    plt.plot(X,learning_curve)
    plt.show()


def main():
    run()


if __name__ == '__main__':
    main()
