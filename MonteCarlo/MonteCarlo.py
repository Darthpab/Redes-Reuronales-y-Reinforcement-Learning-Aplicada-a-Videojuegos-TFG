import gym
import numpy as np

from keras.models import Model
from keras.layers import *
from keras import backend as K
from gym_fightingice.envs import fightingice_env_data_noframeskip
from collections import deque
from keras.layers import Input, Dense
import tensorflow as tf
from keras.optimizers import Adam, Adadelta
from tensorboardX.writer import SummaryWriter
from tensorflow.python.keras.layers.core import Dropout
from tensorflow import config
tf.compat.v1.disable_eager_execution()

EPISODES = 10000
NUM_ACTIONS = 56
NUM_STATE = 143
GAMMA = 0.99
BATCH_SIZE = 128
EPOCHS = 10

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def one_hot(index):
    x = np.zeros((NUM_ACTIONS,))
    x[index] = 1
    return x

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
    return discounted_r

def pg_loss(advantage):
    def f(y_true, y_pred):
        responsible_outputs = K.sum(y_true * y_pred, axis=1)
        policy_loss = -K.sum(advantage * K.log(responsible_outputs))
        return policy_loss
    return f

def get_dropout(input_tensor, p):
    return Dropout(p)(input_tensor, training=True)


def test(self, test_episodes = 100):
    self.load()
    for e in range(100):
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done = False
        score = 0
        while not done:
            for _ in range(100):
                probs_mc_dropout += [self.Actor.predict(state)[0]]
                predictive_mean = np.mean(probs_mc_dropout, axis=0)
            
            action = np.argmax(predictive_mean)
            state, reward, done, _ = self.env.step(action)
            state = np.reshape(state, [1, self.state_size[0]])
            score += reward
            if done:
                print("episode: {}/{}, score: {}".format(e, test_episodes, score))
                break
    self.env.close()




def get_model():
    inp = Input(NUM_STATE)
    x = Dense(128, activation="relu")(inp)
    x = get_dropout(x, p=0.25)
    x = Dense(128, activation="relu")(x)
    x = get_dropout(x, p=0.25)
    out = Dense(NUM_ACTIONS, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=1e-4),metrics=['acc'])
    return model

env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path=".",port=4243)

g_model = get_model()

episode = 0
writer = SummaryWriter("MCTS/run2")

while episode < EPISODES:
    observation = env.reset()
    done = False
    state_history = []
    action_history = []
    reward_history = []
    probs_mc_dropout = []
    while not done:

        state_history.append(observation)
        action_prob = g_model.predict(observation.reshape(1, NUM_STATE))
        action = np.random.choice(NUM_ACTIONS, p=action_prob[0])
        observation, reward, done, info = env.step(action)

        reward_history.append(reward)
        action_history.append(one_hot(action))


        if done:
            reward_sum = sum(reward_history)


            adv = discount_rewards(reward_history)

            state_history = np.array(state_history)
            action_history = np.array(action_history)

            print("EPISODE: ", episode)
            loss = g_model.fit([state_history, adv], action_history, batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=True, validation_split=0.1)

            writer.add_scalar('MCTS loss', loss.history['loss'][-1], episode)
            writer.add_scalar('MCTS val_loss', loss.history['val_loss'][-1], episode)
            writer.add_scalar('MCTS accuracy', loss.history['acc'][-1], episode)
            writer.add_scalar('MCTS val_acc', loss.history['val_acc'][-1], episode)
            writer.add_scalar('Episode reward', np.array(reward_sum), episode)
            
            
            if(episode % 100 == 0):
                writer.add_scalar('Val episode reward', np.array(reward_sum), episode)
                g_model.save('model_mcts_{}.tf'.format(episode))
            episode += 1

test()


