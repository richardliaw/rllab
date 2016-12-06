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
import sys
from os import path as osp
import datetime, dateutil

ray_setting.WORKERS = int(sys.argv[1])
SETTING = int(sys.argv[2])
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')

def get_dir(setting, n_parallel):
    if setting == 0:
        exp = "NO_WAIT"
    elif setting == 1:
        exp = "WAIT"
    elif setting == 2:
        exp = "HIGHUSAGE"
    elif setting == 3:
        exp = "HUCP"
    else:
        exp = "bad"
    return "./RayResults/" + exp + "/Ray/airraid1M/{}".format(n_parallel)

ray_setting.log_dir = osp.join(get_dir(SETTING, ray_setting.WORKERS), timestamp)

ray.init(start_ray_local=True, num_workers=ray_setting.WORKERS)

def env_init():
    return normalize(GymEnv("AirRaid-ram-v0", record_video=False))


def env_reinit(env):
    # env.reset()
    return env

ray.reusables.env = ray.Reusable(env_init, env_reinit)

def policy_init():
    env = ray.reusables.env

    print "using policy env" 
    return CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=(64,64))

def policy_reinit(policy):
    # policy.reset()
    return policy

ray.reusables.policy = ray.Reusable(policy_init, policy_reinit)

env = ray.reusables.env
baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=ray.reusables.env,
    policy=ray.reusables.policy,
    baseline=baseline,
    batch_size=50000,
    max_path_length=env.horizon,
    n_itr=100,
    discount=0.995,
    step_size=0.1,
    sampler_cls=RaySampler,
    sampler_args={"setting": SETTING }
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)
ray_setting.initialize() # initializes the log and such
algo.train()
ray_setting.finish()
