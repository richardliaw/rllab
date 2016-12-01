from __future__ import print_function
from __future__ import absolute_import

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.algos.mod_batch_sampler import ModBatchSampler

from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
import sys

N_PARALLEL = int(sys.argv[1])
WAIT_FOR_STRAGS = int(sys.argv[2])

stub(globals())

import pickle, datetime, dateutil, os

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')


env = normalize(GymEnv("AirRaid-ram-v0", record_video=False))

policy = CategoricalMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=(64,64)
    # The neural network policy should have two hidden layers, each with 32 hidden units.
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=50000,
    max_path_length=env.horizon,
    n_itr=200,
    discount=0.995,
    step_size=0.1,
    sampler_cls=ModBatchSampler,
    sampler_args={"wait_for_stragglers": WAIT_FOR_STRAGS }
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)
run_experiment_lite(
    algo.train(),
    # Number of parallel workers for sampling
    n_parallel=N_PARALLEL,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    log_tabular_only=True,
    log_dir=os.path.join("./RayResults/Timing/RLLab/airraid50k/{}".format(N_PARALLEL), timestamp)
    # plot=True,
)