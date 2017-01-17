from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy

from rllab.algos.ray_sampler import RaySampler, RayMultinodeSampler
from examples.point_env import PointEnv
from rllab.misc.ext import set_seed
from rllab import ray_setting
import traceback
import numpy as np
import ray
import sys
import psutil
from os import path as osp
import datetime, dateutil
import redis

# num_workers = int(sys.argv[1])
SETTING = 1
ADDRESS = sys.argv[1]

r = redis.StrictRedis(*ADDRESS.split(':'))
NUM_WORKERS = len(r.keys("WorkerInfo:*"))
ray_setting.WORKERS = NUM_WORKERS

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')

def get_dir(setting, n_parallel):
    # if setting == 0:
    #     exp = "NO_WAIT"
    # elif setting == 1:
    #     exp = "WAIT"
    # elif setting == 2:
    #     exp = "HIGHUSAGE"
    # elif setting == 3:
    #     exp = "HUCP"
    # else:
    #     exp = "bad"
    return "./RayResults/airraid/Ray/{}".format(n_parallel)

ray_setting.log_dir = osp.join(get_dir(SETTING, ray_setting.WORKERS), timestamp)

ray.init(redis_address=ADDRESS)

def env_init():
    return normalize(GymEnv("AirRaid-ram-v0", record_video=False))


def env_reinit(env):
    # env.reset()
    return env

ray.env.env = ray.EnvironmentVariable(env_init, env_reinit)

def policy_init():
    env = ray.env.env

    print "using policy env" 
    return CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=(64,64))

def policy_reinit(policy):
    # policy.reset()
    return policy

def id_init():
    return str(np.random.randint(10000)) + "_" + str(ray.worker.global_worker.node_ip_address)

def id_reinit(id_worker):
    return id_worker

def numworker_init():
    return NUM_WORKERS

def numworker_reinit(num_w):
    return num_w


ray.env.id = ray.EnvironmentVariable(id_init, id_reinit)
ray.env.policy = ray.EnvironmentVariable(policy_init, policy_reinit)
ray.env.num_workers = ray.EnvironmentVariable(numworker_init, numworker_reinit)


import time; time.sleep(1)

env = ray.env.env
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = TRPO(
    env=ray.env.env,
    policy=ray.env.policy,
    baseline=baseline,
    batch_size=80000,
    max_path_length=env.horizon,
    n_itr=75,
    gae=0.97,
    optimizer_args={"reg_coeff": 0.1},
    discount=0.995,
    step_size=0.01,
    sampler_cls=RayMultinodeSampler,
    sampler_args={"setting": SETTING }
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)

ray_setting.initialize(multinode=True) # initializes the log and such
print ray_setting.ids

# @ray.remote
# def getpin():
#    p = psutil.Process()
#    time.sleep(3)
#    return p.cpu_affinity()

# print ray.get([getpin.remote() for _ in range(ray_setting.WORKERS)])

algo.train()
ray_setting.finish()
