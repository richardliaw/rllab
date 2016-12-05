from __future__ import print_function
from __future__ import absolute_import
from joblib.pool import MemmapingPool
import multiprocessing as mp
from rllab.misc import logger
import pyprind
import time
import traceback
import sys
from datetime import datetime


class ProgBarCounter(object):
    def __init__(self, total_count):
        self.total_count = total_count
        self.max_progress = 1000000
        self.cur_progress = 0
        self.cur_count = 0
        if not logger.get_log_tabular_only():
            self.pbar = pyprind.ProgBar(self.max_progress)
        else:
            self.pbar = None

    def inc(self, increment):
        if not logger.get_log_tabular_only():
            self.cur_count += increment
            new_progress = self.cur_count * self.max_progress / self.total_count
            if new_progress < self.max_progress:
                self.pbar.update(new_progress - self.cur_progress)
            self.cur_progress = new_progress

    def stop(self):
        if not logger.get_log_tabular_only():
            self.pbar.stop()


class SharedGlobal(object):
    pass


class StatefulPool(object):
    def __init__(self):
        self.n_parallel = 1
        self.pool = None
        self.queue = None
        self.worker_queue = None
        self.G = SharedGlobal()

    def initialize(self, n_parallel):
        self.n_parallel = n_parallel
        if self.pool is not None:
            print("Warning: terminating existing pool")
            self.pool.terminate()
            self.queue.close()
            self.worker_queue.close()
            self.G = SharedGlobal()
        # if n_parallel > 1:
        if n_parallel > 0:
            self.queue = mp.Queue()
            self.worker_queue = mp.Queue()
            self.pool = MemmapingPool(
                self.n_parallel,
                temp_folder="/tmp",
            )

    def run_each(self, runner, args_list=None):
        """
        Run the method on each worker process, and collect the result of execution.
        The runner method will receive 'G' as its first argument, followed by the arguments
        in the args_list, if any
        :return:
        """
        if args_list is None:
            args_list = [tuple()] * self.n_parallel
        assert len(args_list) == self.n_parallel
        # if self.n_parallel > 1:
        if self.n_parallel > 0:
            results = self.pool.map_async(
                _worker_run_each, [(runner, args) for args in args_list]
            )
            for i in range(self.n_parallel):
                self.worker_queue.get()
            for i in range(self.n_parallel):
                self.queue.put(None)
            return results.get()
        return [runner(self.G, *args_list[0])]

    def run_map(self, runner, args_list):
        # if self.n_parallel > 1:
        if self.n_parallel > 0:
            return self.pool.map((_worker_run_map, runner, args) for args in args_list)
        else:
            ret = []
            for args in args_list:
                ret.append(runner(self.G, *args))
            return ret

    def run_collect_highusage(self, collect_once, threshold, args=None, 
                                    show_prog_bar=True):
        start = datetime.now() # we assume that last straggler doesn't affect 
        if args is None:
            args = tuple()
        assert self.pool, "MP Pool not available!"
        manager = mp.Manager()
        counter = manager.Value('i', 0)
        lock = manager.RLock()
        if not hasattr(self, "_collected"):
            self._collected = manager.list()
            logger.record_tabular('ObsFromLastItr', 0)
        else:
            with lock:
                debug_startgetremain = datetime.now()
                counter.value += sum(len(x['rewards']) for x in self._collected)
                print( "Getting stragglers took %0.3f seconds..." % (datetime.now() - debug_startgetremain).total_seconds())
                logger.record_tabular('ObsFromLastItr', counter.value)

        overflow = manager.list()
        results_handle = self.pool.map_async(
            _worker_run_collect_highusage, # TODO
            [(collect_once, counter, lock, threshold, self._collected, overflow, args)] * self.n_parallel
        )
        # last_value = 0
        while True:
            time.sleep(0.1)
            with lock:
                if counter.value >= threshold:
                    batch = datetime.now()
                    logger.record_tabular('BatchLimitTime', (batch - start).total_seconds())
                    res = self._collected._getvalue() # this may take a little time
                    self._collected = overflow
                    break
            # last_value = counter.value


        end = datetime.now()
        logger.record_tabular('SampleTimeTaken', (end - start).total_seconds())

        return res

    def run_collect_continuous(self, collect_once, threshold, args=None, 
                                    show_prog_bar=True, 
                                    wait_for_stragglers=True):
        start = datetime.now() # we assume that last straggler doesn't affect 
        if args is None:
            args = tuple()
        assert self.pool, "MP Pool not available!"

        manager = mp.Manager()
        counter = manager.Value('i', 0)
        lock = manager.RLock()
        collected = manager.list()
        results_handle = self.pool.map_async(
            _worker_run_collect_continuous,
            [(collect_once, counter, lock, threshold, collected, args)] * self.n_parallel
        )
        # last_value = 0
        while True:
            time.sleep(0.1)
            with lock:
                if counter.value >= threshold:
                    batch = datetime.now()
                    logger.record_tabular('BatchLimitTime', (batch - start).total_seconds())
                    break
                # last_value = counter.value
        if wait_for_stragglers:
            results_handle.get() # wait for stragglers
            res = collected._getvalue() 
        else:
            res = collected._getvalue() # this may take a little time

        end = datetime.now()
        logger.record_tabular('SampleTimeTaken', (end - start).total_seconds())

        return res

    def run_collect(self, collect_once, threshold, args=None, show_prog_bar=True):
        """
        Run the collector method using the worker pool. The collect_once method will receive 'G' as
        its first argument, followed by the provided args, if any. The method should return a pair of values.
        The first should be the object to be collected, and the second is the increment to be added.
        This will continue until the total increment reaches or exceeds the given threshold.

        Sample script:

        def collect_once(G):
            return 'a', 1

        stateful_pool.run_collect(collect_once, threshold=3) # => ['a', 'a', 'a']

        :param collector:
        :param threshold:
        :return:
        """
        start = datetime.now()
        if args is None:
            args = tuple()
        if self.pool:
            manager = mp.Manager()
            counter = manager.Value('i', 0)
            lock = manager.RLock()
            results = self.pool.map_async(
                _worker_run_collect,
                [(collect_once, counter, lock, threshold, args)] * self.n_parallel
            )
            if show_prog_bar:
                pbar = ProgBarCounter(threshold)
            last_value = 0
            while True:
                time.sleep(0.1)
                with lock:
                    if counter.value >= threshold:
                        
                        batch = datetime.now()
                        logger.record_tabular('BatchLimitTime', (batch - start).total_seconds())

                        if show_prog_bar:
                            pbar.stop()
                        break
                    if show_prog_bar:
                        pbar.inc(counter.value - last_value)
                    last_value = counter.value
            res = sum(results.get(), [])
            end = datetime.now()
            logger.record_tabular('SampleTimeTaken', (end - start).total_seconds())
            return res
        else:
            count = 0
            results = []
            if show_prog_bar:
                pbar = ProgBarCounter(threshold)
            while count < threshold:
                result, inc = collect_once(self.G, *args)
                results.append(result)
                count += inc
                if show_prog_bar:
                    pbar.inc(inc)
            if show_prog_bar:
                pbar.stop()
            return results


singleton_pool = StatefulPool()


def _worker_run_each(all_args):
    try:
        runner, args = all_args
        # signals to the master that this task is up and running
        singleton_pool.worker_queue.put(None)
        # wait for the master to signal continuation
        singleton_pool.queue.get()
        return runner(singleton_pool.G, *args)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def _worker_run_collect(all_args):
    try:
        collect_once, counter, lock, threshold, args = all_args
        collected = []
        while True:
            with lock:
                if counter.value >= threshold:
                    return collected
            result, inc = collect_once(singleton_pool.G, *args)
            collected.append(result)
            with lock:
                counter.value += inc
                if counter.value >= threshold:
                    return collected
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))

def _worker_run_collect_continuous(all_args):
    try:
        collect_once, counter, lock, threshold, collected, args = all_args
        while True:
            with lock:
                if counter.value >= threshold:
                    return collected
            result, inc = collect_once(singleton_pool.G, *args)
            with lock:
                collected.append(result) # this is changed to shared variable
                counter.value += inc
                if counter.value >= threshold:
                    return collected
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))

def _worker_run_collect_highusage(all_args):
    try:
        collect_once, counter, lock, threshold, collected, overflow, args = all_args
        while True:
            with lock:
                if counter.value >= threshold:
                    return collected
            result, inc = collect_once(singleton_pool.G, *args)
            with lock:
                if counter.value >= threshold:
                    overflow.append(result)
                    return collected
                counter.value += inc
                collected.append(result)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def _worker_run_map(all_args):
    try:
        runner, args = all_args
        return runner(singleton_pool.G, *args)
    except Exception:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))
