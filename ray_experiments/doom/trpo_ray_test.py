from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy

from rllab.algos.ray_sampler import RaySampler
from examples.point_env import PointEnv
from rllab.misc.ext import set_seed
from rllab import ray_setting
import traceback
import numpy as np
import ray

ray_setting.WORKERS = 1

ray.init(num_workers=ray_setting.WORKERS)

def env_init():
    return normalize(GymEnv("ppaquette/DoomBasic-v0", record_video=False))


def env_reinit(env):
    return env

ray.env.env = ray.EnvironmentVariable(env_init, env_reinit)

def policy_init():
    env = ray.env.env
    return CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=(64,64))

def policy_reinit(policy):
    # policy.reset()
    return policy

ray.env.policy = ray.EnvironmentVariable(policy_init, policy_reinit)

env = ray.env.env
baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=ray.env.env,
    policy=ray.env.policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=env.horizon,
    n_itr=50,
    discount=0.995,
    step_size=0.1,
    sampler_cls=RaySampler
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)

algo.train()
