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
import ray
from rllab.policies.base import Policy

class RaySampler(BatchSampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
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
        )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated