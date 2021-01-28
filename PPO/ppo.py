
import random
import gym
import pylab
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter

tf.compat.v1.disable_eager_execution() 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import copy
from gym_fightingice.envs import fightingice_env_data_noframeskip
from threading import Thread, Lock
from multiprocessing import Process, Pipe
import time




class Environment(Process):
    def __init__(self, env_idx, child_conn, i,  state_size, action_size, visualize=False):
        super(Environment, self).__init__()
        self.env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path=".",port=i)

        self.env_idx = env_idx
        self.child_conn = child_conn
        self.state_size = state_size
        self.action_size = action_size


    def run(self):
        super(Environment, self).run()
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        self.child_conn.send(state)
        while True:
            action = self.child_conn.recv()

            state, reward, done, info = self.env.step(action)
            state = np.reshape(state, [1, self.state_size])

            if done:
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

            self.child_conn.send([state, reward, done, info])


class Actor_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space

        X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        output = Dense(self.action_space, activation="softmax")(X)

        self.Actor = Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(lr=lr))

    def ppo_loss(self, y_true, y_pred):

        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        
        total_loss = actor_loss - entropy

        return total_loss

    def predict(self, state):
        return self.Actor.predict(state)


class Critic_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        old_values = Input(shape=(1,))

        V = Dense(512, activation="relu", kernel_initializer='he_uniform')(X_input)
        V = Dense(256, activation="relu", kernel_initializer='he_uniform')(V)
        V = Dense(64, activation="relu", kernel_initializer='he_uniform')(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=[X_input, old_values], outputs = value)
        self.Critic.compile(loss=[self.critic_PPO2_loss(old_values)], optimizer=optimizer(lr=lr))

    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2
            clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            #value_loss = K.mean((y_true - y_pred) ** 2) 
            return value_loss
        return loss

    def predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])

class PPOAgent:
    def __init__(self):
        self.env = gym.make("FightingiceDataNoFrameskip-v0", java_env_path=".",port=4242)
        self.action_size = 56
        self.state_size = (143,)
        self.EPISODES = 4000 
        self.episode = 1 
        self.max_average = 0 
        self.lr = 1e-4
        self.epochs = 10 
        self.shuffle=True
        self.Training_batch = 256
        self.optimizer = Adam

        self.replay_count = 0
        self.writer = SummaryWriter(logdir="Fighting-ICE/run9",comment="_"+self.optimizer.__name__+"_"+str(self.lr))
        
        self.scores_, self.episodes_, self.average_ = [], [], [] 

        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        
        self.Actor_name = "{}_PPO_Actor.h5".format(self.episode)
        self.Critic_name = "{}_PPO_Critic.h5".format(self.episode)

    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.95, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        values = self.Critic.predict(states)
        print(values)
        next_values = self.Critic.predict(next_states)

        #discounted_r = self.discount_rewards(rewards)
        #advantages = np.vstack(discounted_r - values)
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        '''
        pylab.plot(advantages,'.')
        pylab.plot(target,'-')
        ax=pylab.gca()
        ax.grid(True)
        pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
        pylab.show()
        '''

        y_true = np.hstack([advantages, predictions, actions])
        
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=True, shuffle=self.shuffle)
        c_loss = self.Critic.Critic.fit([states, values], target, epochs=self.epochs, verbose=True, shuffle=self.shuffle)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.replay_count += 1
 
    def load(self):
        self.Actor.Actor.load_weights(self.Actor_name)
        self.Critic.Critic.load_weights(self.Critic_name)

    def save(self):
        self.Actor.Actor.save_weights(self.Actor_name)
        self.Critic.Critic.save_weights(self.Critic_name)
        
    pylab.figure(figsize=(18, 9))
    pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)

    def PlotModel(self, score, episode):
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-5:]) / len(self.scores_[-5:]))
        if str(episode)[-2:] == "00":
            pylab.plot(self.episodes_, self.scores_, 'b')
            pylab.plot(self.episodes_, self.average_, 'r')
            pylab.title("FightingICE"+" PPO training cycle", fontsize=18)
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.grid(True)
                pylab.savefig("FightingICE"+".png")
            except OSError:
                pass

        if self.average_[-1] >= self.max_average:
            self.max_average = self.average_[-1]
            self.save()
            SAVING = "SAVING"
    
            self.lr *= 0.95
            K.set_value(self.Actor.Actor.optimizer.learning_rate, self.lr)
            K.set_value(self.Critic.Critic.optimizer.learning_rate, self.lr)
        else:
            SAVING = ""

        return self.average_[-1], SAVING

    def run_multiprocesses(self, num_worker = 1):
        i = 4241
        works, parent_conns, child_conns = [], [], []
        for idx in range(num_worker):
            i = i + 1
            parent_conn, child_conn = Pipe()
            work = Environment(idx, child_conn, i, self.state_size[0], self.action_size, True)
            work.start()
            works.append(work)
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)

        states =        [[] for _ in range(num_worker)]
        next_states =   [[] for _ in range(num_worker)]
        actions =       [[] for _ in range(num_worker)]
        rewards =       [[] for _ in range(num_worker)]
        dones =         [[] for _ in range(num_worker)]
        predictions =   [[] for _ in range(num_worker)]
        score =         [0 for _ in range(num_worker)]

        state = [0 for _ in range(num_worker)]
        for worker_id, parent_conn in enumerate(parent_conns):
            state[worker_id] = parent_conn.recv()

        while self.episode < self.EPISODES:
            predictions_list = self.Actor.predict(np.reshape(state, [num_worker, self.state_size[0]]))
            print(predictions_list)
            actions_list = [np.random.choice(self.action_size, p=i) for i in predictions_list]

            for worker_id, parent_conn in enumerate(parent_conns):
                parent_conn.send(actions_list[worker_id])
                action_onehot = np.zeros([self.action_size])
                action_onehot[actions_list[worker_id]] = 1
                actions[worker_id].append(action_onehot)
                predictions[worker_id].append(predictions_list[worker_id])

            for worker_id, parent_conn in enumerate(parent_conns):
                next_state, reward, done, info = parent_conn.recv()

                states[worker_id].append(state[worker_id])
                next_states[worker_id].append(next_state)
                rewards[worker_id].append(reward)
                dones[worker_id].append(done)
                state[worker_id] = next_state
                

                if done:
                    score[worker_id] = info[0] - info[1]
                    average, SAVING = self.PlotModel(score[worker_id], self.episode)
                    print("episode: {}/{}, worker: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, worker_id, score[worker_id], average, SAVING))
                    self.writer.add_scalar('Workers:{}/score_per_episode'.format(worker_id),score[worker_id], self.episode)
                    self.writer.add_scalar('Workers:{}/learning_rate'.format(worker_id),self.lr, self.episode)
                    score[worker_id] = 0
                    if(self.episode <= self.EPISODES):
                        self.episode += 1
                        
            for worker_id in range(num_worker):
                if len(states[worker_id]) >= self.Training_batch:
                    self.replay(states[worker_id], actions[worker_id], rewards[worker_id], predictions[worker_id], dones[worker_id], next_states[worker_id])
                    
                    states[worker_id] = []
                    next_states[worker_id] = []
                    actions[worker_id] = []
                    rewards[worker_id] = []
                    dones[worker_id] = []
                    predictions[worker_id] = []


        works.append(work)
        for work in works:
            work.terminate()
            print('TERMINATED:', work)
            work.join()
            

    def test(self, test_episodes = 100):
        self.load()
        for e in range(100):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            done = False
            score = 0
            while not done:
                action = np.argmax(self.Actor.predict(state)[0])
                state, reward, done, _ = self.env.step(action)
                state = np.reshape(state, [1, self.state_size[0]])
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, test_episodes, score))
                    break
        self.env.close()

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:

            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:

            print(e)


    agent = PPOAgent()
    #agent.run() 
    #agent.run_batch() 
    agent.run_multiprocesses(num_worker = 1)  
    #agent.test()