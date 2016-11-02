from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

stub(globals())

import pickle, datetime, dateutil, os

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')

env = normalize(CartpoleEnv())

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=1000,
    max_path_length=500,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)

N_PARALLEL = 16
run_experiment_lite(
    algo.train(),
    # Number of parallel workers for sampling
    n_parallel=N_PARALLEL,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    # seed=20,
    log_dir=os.path.join("./Results/MTPerf/Work_{}".format(N_PARALLEL), timestamp)
    # plot=True,
)
