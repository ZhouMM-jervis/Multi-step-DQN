"""
Using:
Tensorflow: 1.0
gym: 0.8.0

"""


import gym
from RL_brain import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 11

sess = tf.Session()
with tf.variable_scope('DDQN'):
    DDQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, multi_step=False, sess=sess
    )

with tf.variable_scope('Multi_step_DDQN'):
    Multi_step_DDQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, multi_step=True, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    observation = env.reset()
    while True:
        # if total_steps - MEMORY_SIZE > 8000: env.render()

        action = RL.choose_action(observation)
        # convert to [-2 ~ 2] float actions
        f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)
        observation_1, reward, done, info = env.step(np.array([f_action]))

        action_1 = RL.choose_action(observation_1)
        f_action_1 = (action_1 - (ACTION_SPACE - 1) / 2) / \
            ((ACTION_SPACE - 1) / 4)
        observation_2, reward_1, done, info = env.step(np.array([f_action_1]))

        action_2 = RL.choose_action(observation_2)
        f_action_2 = (action_2 - (ACTION_SPACE - 1) / 2) / \
            ((ACTION_SPACE - 1) / 4)
        observation_3, reward_2, done, info = env.step(np.array([f_action_2]))

        # record the newest action_value
        action_3 = RL.choose_action(observation_3)
        f_action_3 = (action_3 - (ACTION_SPACE - 1) / 2) / \
            ((ACTION_SPACE - 1) / 4)
        observation_4, reward_3, done, info = env.step(np.array([f_action_3]))

        reward /= 10
        reward_1 /= 10
        # normalize to a range of (-1, 0). r = 0 when get upright
        reward_2 /= 10
        reward_3 /= 10
        

        RL.store_transition(observation, action, reward, observation_1, action_1,
                            reward_1, observation_2, action_2, reward_2, observation_3, action_3, reward_3)

        if total_steps > MEMORY_SIZE:   # learning
            RL.learn()

        if total_steps - MEMORY_SIZE > 20000:   # stop game
            break

        observation = observation_3
        total_steps += 3
    return RL.q


q_natural = train(DDQN)
q_double = train(Multi_step_DDQN)

plt.plot(np.array(q_natural), c='r', label='DDQN')
plt.plot(np.array(q_double), c='b', label='Multi_step_DDQN')
plt.legend(loc='best')
plt.ylabel('Cost')
plt.xlabel('training steps')
plt.grid()
plt.show()
