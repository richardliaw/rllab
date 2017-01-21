import gym
import ppaquette_gym_doom
import time
import numpy as np
import logging
# logging.

# class BatchEnvironments():
# 	def __init__(self, gym_name, size):
# 		self.envs = [gym.make(gym_name) for i in range(size)]

# 	def set_envs(self, list_envs):
# 		self.envs = list_envs

def all_random_step(envs):
	actions = [env.action_space.sample() for env in envs]
	start = time.time()
	next_states = [env.step(a) for env, a in zip(envs, actions)]
	end = time.time()
	for i in range(len(envs)):
		if next_states[i][2]:
			envs[i].reset()
	return next_states, end - start


def random_rollout():
	logging.getLogger().setLevel(logging.INFO)

	env = gym.make("ppaquette/DoomBasic-v0")
	o = env.reset()
	timing = []
	while path_length < 500:
		a = env.action_space.sample() #random
		start = time.time()
		next_o, r, d, env_info = env.step(a)
		end = time.time()
		path_length += 1
		timing.append( end - start )
		# print path_length, 
		# observations.append(env.observation_space.flatten(o))
		# rewards.append(r)
		# actions.append(env.action_space.flatten(a))
		# env_infos.append(env_info)
		if d:
			o = env.reset()
			break
