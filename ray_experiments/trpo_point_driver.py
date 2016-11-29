import ray
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from examples.point_env import PointEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import argparse

print "Hi"
ray.init(start_ray_local=True, num_workers=2)
# ray.register_class(PointEnv)
# env = PointEnv()
# env_id = ray.put(env)
def env_init():
	return normalize(PointEnv())

def env_reinit(env):
	env.reset()
	return env
ray.reusables.env = ray.Reusable(env_init, env_reinit)

def policy_init():
	env = ray.reusables.env
	return GaussianMLPPolicy(env_spec=env.spec)

def policy_reinit(policy):
	policy.reset()
	return policy

ray.reusables.policy = ray.Reusable(policy_init, policy_reinit)
# env = normalize(PointEnv())
# policy = GaussianMLPPolicy(
#     env_spec=env.spec,
# )
def baseline_init():
	env = ray.reusables.env
	return LinearFeatureBaseline(env_spec=env.spec)

def baseline_reinit(baseline):
	return baseline

ray.reusables.baseline = ray.Reusable(baseline_init, baseline_reinit)
# baseline = LinearFeatureBaseline(env_spec=env.spec)


# algo = TRPO(
#     env=env,
#     policy=policy,
#     baseline=baseline,
# )


def algo_init():
	env = ray.reusables.env
	policy = ray.reusables.policy 
	baseline= ray.reusables.baseline
	return TRPO(
	    env=env,
	    policy=policy,
	    baseline=baseline,
	    n_itr=10,
	)

def algo_reinit(algo):
	return algo

ray.reusables.algo = ray.Reusable(algo_init, algo_reinit)

@ray.remote
def train_policy():
	algo = ray.reusables.algo
	algo.train()

result = [train_policy.remote() for x in range(2)]
y = [ray.get(x) for x in result]
print y
# algo.train()
