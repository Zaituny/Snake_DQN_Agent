
from PIL import Image  # To transform the image in the Processor
import numpy as np
import gym

# Convolutional Backbone Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam

# Keras-RL
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

env = gym.make("Snake_gym:snake-v0")
nb_actions = env.action_space.n

IMG_SHAPE = (84, 84)
WINDOW_LENGTH = 4

class ImageProcessor(Processor):
    def process_observation(self, observation):
        # First convert the numpy array to a PIL Image
        img = Image.fromarray(observation)
        # Then resize the image
        img = img.resize(IMG_SHAPE)
        # And convert it to grayscale  (The L stands for luminance)
        img = img.convert("L")
        # Convert the image back to a numpy array and finally return the image
        img = np.array(img)
        return img.astype('uint8')  # saves storage in experience memory
    
    def process_state_batch(self, batch):

        # We divide the observations by 255 to compress it into the intervall [0, 1].
        # This supports the training of the network
        # We perform this operation here to save memory.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

input_shape = (WINDOW_LENGTH, IMG_SHAPE[0], IMG_SHAPE[1])

model = Sequential()
model.add(Permute((2, 3, 1), input_shape=input_shape))

model.add(Convolution2D(32, (8, 8), strides=(4, 4),kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2), kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1), kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

model.load_weights("test_dqn_snake_weights.h5f") 

memory = SequentialMemory(limit = 1000000, window_length=WINDOW_LENGTH)

processor = ImageProcessor()

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps = 1000000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
              train_interval=4, delta_clip=1)

dqn.compile(Adam(learning_rate=.00025), metrics=['mae'])

env.sleep = 0.2

dqn.test(env, nb_episodes = 5, visualize = True)

#weights_filename = 'test_dqn_snake_weights.h5f'
#checkpoint_weights_filename = 'test_dqn_' + "snake" + '_weights_{step}.h5f'
#checkpoint_callback = ModelIntervalCheckpoint(checkpoint_weights_filename, interval=100000)

#dqn.fit(env, nb_steps = 2000000, callbacks=[checkpoint_callback], log_interval=100000, visualize=False)

# After training is done, we save the final weights one more time.
#dqn.save_weights(weights_filename, overwrite=True)

"""
#import dependencies

import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = 'Snake_gym:snake-v0'

env = gym.make(env_name)
env = DummyVecEnv([lambda: env])

log_path = os.path.join("Training", "Logs")
model = DQN('CnnPolicy', env, verbose = 1, tensorboard_log = log_path)
model.learn(total_timesteps = 100000)

save_path = os.path.join("Training", "Models", "DQN_Model_Snake_100K_steps")

model.save(save_path)

del model

model = DQN.load(save_path)

evaluate_policy(model, env, n_eval_episodes = 10, render = True)
"""

               
