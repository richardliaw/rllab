import numpy as np
from rllab.algos.base import RLAlgorithm
from rllab.algos.batch_polopt import BatchSampler
from rllab.sampler import parallel_sampler
from rllab.sampler.base import Sampler
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger
import rllab.plotter as plotter
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

        self._remaining_tasks = []
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

        num_workers = ray.reusables.num_workers
        start = datetime.now()

        param_id = ray.put(policy_params)    
        num_samples = 0
        results = []
        remaining = []
        timing = {wid:[] for wid in ray_setting.ids}
        print timing
        log_samples = {wid:[] for wid in ray_setting.ids}

        # if high_usage:
        #     previous_stragglers = ray.get(_remaining_tasks)
        #     results.extend(previous_stragglers)
        #     prev_samples = sum(len(roll['rewards']) for roll in previous_stragglers)
        #     if count_prev: # NOT counting 
        #         num_samples += prev_samples
        #     logger.record_tabular('ObsFromLastItr', prev_samples)


        # while num_samples < max_samples:
        #     for i in range(num_workers - len(remaining)): # consider doing 2x in order to obtain good throughput
        #         remaining.append(parallel_sampler.ray_rollout.remote(param_id, max_path_length))
        #     done, remaining = ray.wait(remaining)
        #     result, wid, timestamps = ray.get(done[0])
        #     trajlen = len(result['rewards'])

        #     #timing
        #     timing[wid].append(timestamps)
        #     log_samples[wid].append(trajlen)

        #     num_samples += trajlen
        #     results.append(result)
        remaining = [parallel_sampler.ray_rollout.remote(param_id, max_path_length) for _ in range(50)]
        result_info = ray.get(remaining)

        for result, wid, timestamp in result_info:
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

        # _remaining_tasks = remaining

        # logger.record_tabular("WastePerWorker", wasted_work(timing, num_workers, starttime=start))

        end = datetime.now()
        logger.record_tabular('SampleTimeTaken', (end - start).total_seconds())  
        timing["total"] = (str(start), str(end))
        ray_timing.log['timing'].append(timing)
        ray_timing.log['samples'].append(log_samples)
        return results