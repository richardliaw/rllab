from rllab.sampler.utils import rollout
from rllab.sampler.stateful_pool import singleton_pool, SharedGlobal
from rllab.misc import ext
from rllab.misc import logger
from rllab.misc import tensor_utils
import pickle
import numpy as np
try:
    from datetime import datetime
    import ray
    from rllab import ray_setting
    from rllab import ray_timing
except Exception:
    print "No Ray Installed"


def _worker_init(G, id):
    # if singleton_pool.n_parallel > 1:
    if singleton_pool.n_parallel > 0:
        import os
        os.environ['THEANO_FLAGS'] = 'device=cpu'
    G.worker_id = id


def initialize(n_parallel):
    singleton_pool.initialize(n_parallel)
    singleton_pool.run_each(_worker_init, [(id,) for id in xrange(singleton_pool.n_parallel)])


def _get_scoped_G(G, scope):
    if scope is None:
        return G
    if not hasattr(G, "scopes"):
        G.scopes = dict()
    if scope not in G.scopes:
        G.scopes[scope] = SharedGlobal()
        G.scopes[scope].worker_id = G.worker_id
    return G.scopes[scope]


def _worker_populate_task(G, env, policy, scope=None):
    G = _get_scoped_G(G, scope)
    G.env = pickle.loads(env)
    G.policy = pickle.loads(policy)


def _worker_terminate_task(G, scope=None):
    G = _get_scoped_G(G, scope)
    if getattr(G, "env", None):
        G.env.terminate()
        G.env = None
    if getattr(G, "policy", None):
        G.policy.terminate()
        G.policy = None


def populate_task(env, policy, scope=None):
    logger.log("Populating workers...")
    # if singleton_pool.n_parallel > 1:
    if singleton_pool.n_parallel > 0:
        singleton_pool.run_each(
            _worker_populate_task,
            [(pickle.dumps(env), pickle.dumps(policy), scope)] * singleton_pool.n_parallel
        )
    else:
        # avoid unnecessary copying
        G = _get_scoped_G(singleton_pool.G, scope)
        G.env = env
        G.policy = policy
    logger.log("Populated")


def terminate_task(scope=None):
    singleton_pool.run_each(
        _worker_terminate_task,
        [(scope,)] * singleton_pool.n_parallel
    )


def _worker_set_seed(_, seed):
    ext.set_seed(seed)


def set_seed(seed):
    singleton_pool.run_each(
        _worker_set_seed,
        [(seed + i,) for i in xrange(singleton_pool.n_parallel)]
    )

def _worker_set_policy_params(G, params, scope=None):
    G = _get_scoped_G(G, scope)
    G.policy.set_param_values(params)

def _worker_collect_one_path(G, max_path_length, scope=None):
    G = _get_scoped_G(G, scope)
    path = rollout(G.env, G.policy, max_path_length)
    return path, len(path["rewards"])


def sample_paths(
        policy_params,
        max_samples,
        max_path_length=np.inf,
        scope=None):
    """
    :param policy_params: parameters for the policy. This will be updated on each worker process
    :param max_samples: desired maximum number of samples to be collected. The actual number of collected samples
    might be greater since all trajectories will be rolled out either until termination or until max_path_length is
    reached
    :param max_path_length: horizon / maximum length of a single trajectory
    :return: a list of collected paths
    """
    singleton_pool.run_each(
        _worker_set_policy_params,
        [(policy_params, scope)] * singleton_pool.n_parallel
    )
    return singleton_pool.run_collect(
        _worker_collect_one_path,
        threshold=max_samples,
        args=(max_path_length, scope),
        show_prog_bar=True
    )

def sample_paths_cont(
        policy_params,
        max_samples,
        max_path_length=np.inf,
        wait_for_stragglers=True,
        high_usage=False,
        count_prev=False,
        scope=None):
    """
    :param policy_params: parameters for the policy. This will be updated on each worker process
    :param max_samples: desired maximum number of samples to be collected. The actual number of collected samples
    might be greater since all trajectories will be rolled out either until termination or until max_path_length is
    reached
    :param max_path_length: horizon / maximum length of a single trajectory
    :return: a list of collected paths
    """
    singleton_pool.run_each(
        _worker_set_policy_params,
        [(policy_params, scope)] * singleton_pool.n_parallel
    )
    if high_usage:
        assert not wait_for_stragglers
        return singleton_pool.run_collect_highusage(
        _worker_collect_one_path,
        threshold=max_samples,
        args=(max_path_length, scope),
        )
    else:
        return singleton_pool.run_collect_continuous(
            _worker_collect_one_path,
            threshold=max_samples,
            args=(max_path_length, scope),
            show_prog_bar=True,
            wait_for_stragglers=wait_for_stragglers
        )

@ray.remote
def ray_rollout(policy_params, max_path_length):
    """returns rollout dictionary, id, (start, end)"""
    global profile
    start_time = str(datetime.now())
    env = ray.reusables.env
    policy = ray.reusables.policy
    selfid = ray.reusables.id
    policy.set_param_values(policy_params)

    import cProfile, pstats, StringIO
    pr = cProfile.Profile()
    pr.enable()
    
    traj = rollout(env, policy, max_path_length)

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    with open("./tmp/%d_%d.txt" % (ray.reusables.numworker, ray.reusables.id), "a") as f:
        f.write(s.getvalue())
        f.flush()
    return traj, selfid, (start_time, str(datetime.now()))

def wasted_work(times, num_workers, starttime=None):
    """currently not used"""
    curtime = datetime.now()
    most_recent_idx = min(times, key=lambda k: (curtime - times[k]).total_seconds())
    most_recent = times[most_recent_idx]
    wastedtime = sum((most_recent - ts).total_seconds() for ts in times.values())
    if starttime:
        total_time = (most_recent - starttime).total_seconds() * num_workers
        logger.record_tabular("PcentWastedTime", wastedtime / total_time)
    return wastedtime / num_workers


_remaining_tasks = []
def ray_sample_paths(
        policy_params,
        max_samples,
        max_path_length=np.inf,
        scope=None,
        wait_for_stragglers=True,
        high_usage=False,
        count_prev=False):
    global _remaining_tasks
    num_workers = ray_setting.WORKERS
    start = datetime.now()

    param_id = ray.put(policy_params)    
    num_samples = 0
    results = []
    remaining = []
    timing = {wid:[] for wid in ray_setting.ids}
    log_samples = {wid:[] for wid in ray_setting.ids}

    if high_usage:
        previous_stragglers = ray.get(_remaining_tasks)
        results.extend(previous_stragglers)
        prev_samples = sum(len(roll['rewards']) for roll in previous_stragglers)
        if count_prev: # NOT counting 
            num_samples += prev_samples
        logger.record_tabular('ObsFromLastItr', prev_samples)


    while num_samples < max_samples:
        for i in range(num_workers - len(remaining)): # consider doing 2x in order to obtain good throughput
            remaining.append(ray_rollout.remote(param_id, max_path_length))
        done, remaining = ray.wait(remaining)
        result, wid, timestamps = ray.get(done[0])
        trajlen = len(result['rewards'])

        #timing
        timing[wid].append(timestamps)
        log_samples[wid].append(trajlen)

        num_samples += trajlen
        results.append(result)
    batch = datetime.now()
    logger.record_tabular('BatchLimitTime', (batch - start).total_seconds())

    if wait_for_stragglers:
        straggler_results = ray.get(remaining) 
        stragglers = []
        for r, wid, timestamps in straggler_results:
            timing[wid].append(timestamps)
            log_samples[wid].append(len(r['rewards']))
            results.append(r)
        remaining = []

    _remaining_tasks = remaining

    # logger.record_tabular("WastePerWorker", wasted_work(timing, num_workers, starttime=start))

    end = datetime.now()
    logger.record_tabular('SampleTimeTaken', (end - start).total_seconds())  
    timing["total"] = (str(start), str(end))
    ray_timing.log['timing'].append(timing)
    ray_timing.log['samples'].append(log_samples)
    import ipdb; ipdb.set_trace()  # breakpoint cc784955 //

    return results

def truncate_paths(paths, max_samples):
    """
    Truncate the list of paths so that the total number of samples is exactly equal to max_samples. This is done by
    removing extra paths at the end of the list, and make the last path shorter if necessary
    :param paths: a list of paths
    :param max_samples: the absolute maximum number of samples
    :return: a list of paths, truncated so that the number of samples adds up to max-samples
    """
    # chop samples collected by extra paths
    # make a copy
    paths = list(paths)
    total_n_samples = sum(len(path["rewards"]) for path in paths)
    while len(paths) > 0 and total_n_samples - len(paths[-1]["rewards"]) >= max_samples:
        total_n_samples -= len(paths.pop(-1)["rewards"])
    if len(paths) > 0:
        last_path = paths.pop(-1)
        truncated_last_path = dict()
        truncated_len = len(last_path["rewards"]) - (total_n_samples - max_samples)
        for k, v in last_path.iteritems():
            if k in ["observations", "actions", "rewards"]:
                truncated_last_path[k] = tensor_utils.truncate_tensor_list(v, truncated_len)
            elif k in ["env_infos", "agent_infos"]:
                truncated_last_path[k] = tensor_utils.truncate_tensor_dict(v, truncated_len)
            else:
                raise NotImplementedError
        paths.append(truncated_last_path)
    return paths
