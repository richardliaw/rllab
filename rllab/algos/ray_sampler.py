import numpy as np
from rllab.algos.base import RLAlgorithm
from rllab.algos.batch_polopt import BatchSampler
from rllab.sampler import parallel_sampler
from rllab.sampler.base import Sampler
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
from rllab.sampler.utils import rollout
import rllab.misc.logger as logger
import rllab.plotter as plotter
import time
from os import path as osp
try:
    from datetime import datetime
    import ray
    from rllab import ray_setting
    from rllab import ray_timing
except Exception:
    print "No Ray Installed"
import pickle
from rllab.policies.base import Policy

NO_WAIT = 0
WAIT_FOR_STRAGS = 1
HIGH_USAGE = 2
HU_APPEND = 3

class RaySampler(BatchSampler):
    def __init__(self, algo, setting=WAIT_FOR_STRAGS):
        self.high_usage = False
        self.wait_for_stragglers = True
        self.count_prev = False
        if setting == NO_WAIT:
            self.wait_for_stragglers = False
        elif setting == HIGH_USAGE:
            self.high_usage = True
            self.wait_for_stragglers = False
            self.count_prev = True
        elif setting == HU_APPEND:
            self.high_usage = True
            self.wait_for_stragglers = False
            self.count_prev = False
        self.algo = algo

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, itr):
        logger.log("USING RAY")
        
        cur_params = self.algo.policy.get_param_values()
        
        paths = parallel_sampler.ray_sample_paths(
            policy_params=cur_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
            wait_for_stragglers=self.wait_for_stragglers,
            high_usage=self.high_usage,
            count_prev=self.count_prev
        )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated

class RayMultinodeSampler(RaySampler):
    def __init__(self, algo, **kwargs):
        self.num_batch_tasks = 100
        self.task_info = []
        self.timing = []
        self.cur_itr = 0
        super(RayMultinodeSampler, self).__init__(algo, **kwargs)


    def obtain_samples(self, itr):
        logger.log("USING RAY")
        
        cur_params = self.algo.policy.get_param_values()
        
        paths = self.sample_paths(
            policy_params=cur_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
            wait_for_stragglers=self.wait_for_stragglers,
            high_usage=self.high_usage,
            count_prev=self.count_prev
        )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated

    def sample_paths(self, policy_params,
            max_samples,
            max_path_length=np.inf,
            scope=None,
            wait_for_stragglers=True,
            high_usage=False,
            count_prev=False):

        self.cur_itr += 1
        start = time.time()
        logger.record_tabular('SampleStart', start)

        param_id = ray.put(policy_params)    
        num_samples = 0
        results = []
        remaining = []
        log_samples = {wid:[] for wid in ray_setting.ids}

        remaining = [ray_rollout_debug.remote(param_id, max_path_length, {"itr": self.cur_itr, "job_id": "%d_%d" % (self.cur_itr, i)}) 
                        for i in range(int(self.num_batch_tasks * 1.5))]

        while num_samples < max_samples and len(remaining):
            done, remaining = ray.wait(remaining, num_returns=min(len(remaining), 5))
            for d in done:
                result, info = ray.get(d)
                trajlen = len(result['rewards'])
    
                log_samples[info['worker_id']].append(trajlen)
    
                num_samples += trajlen

                info["collected_itr"] = self.cur_itr
                self.task_info.append(info)

                results.append(result)

        end = time.time()
        logger.record_tabular('SampleEnd', end)

        # ray_timing.log['samples'].append(log_samples)
        avg_traj_len = num_samples / sum(len(jobs) for jobs in log_samples.values())
        self.num_batch_tasks = self.algo.batch_size / avg_traj_len
        return results

    def shutdown_worker(self):
        import csv
        keys = self.task_info[0].keys()
        with open(osp.join(ray_setting.log_dir, "tasks.csv"), "w") as f:
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            writer.writerows(self.task_info)
        print "Done saving task info."
        


@ray.remote
def ray_rollout_debug(policy_params, max_path_length, info={}):
    """returns rollout dictionary, id, (start, end)"""
    # global profile
    env = ray.env.env
    policy = ray.env.policy
    selfid = ray.env.id
    policy.set_param_values(policy_params)

    # TODO: log job number
    ray.log_event("rollout:id", contents={"id": info["job_id"]})

    # import cProfile, pstats, StringIO
    # pr = cProfile.Profile()
    # pr.enable()
    info["worker_id"] = selfid
    info["start"] = time.time()
    traj = rollout(env, policy, max_path_length)
    info["end"] = time.time()
    info["trajlen"] = len(traj['rewards'])

    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'name'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # with open("./tmp/%d_%d.txt" % (ray.env.numworker, ray.env.id), "a") as f:
    #     f.write(s.getvalue())
    #     f.flush()
    return traj, info
