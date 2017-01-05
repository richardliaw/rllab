import rllab.misc.logger as logger
import argparse
import os.path as osp
import dateutil.tz
import datetime
import ray
import ast
import time
import psutil


WORKERS = 4
tabular_log_file = './tmp/progress.csv'
text_log_file = './tmp/debug.log'
log_dir = './Results/'
ids = range(WORKERS)

@ray.remote
def get_id():
    time.sleep(0.2)
    wid = ray.reusables.id
    return wid

def refresh_ids():
    global ids
    ids = ray.get([get_id.remote() for _ in range(WORKERS)])
    assert len(set(ids)) == WORKERS
    return ids

@ray.remote
def pin(n):
   # Pin whatever worker runs this remote function to core n
   p = psutil.Process()
   time.sleep(5)
   p.cpu_affinity([n])


def initialize(argv=[]):    
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')

    default_exp_name = 'experiment_%s' % (timestamp)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_parallel', type=int, default=1,
                        help='Number of parallel workers to perform rollouts.')
    parser.add_argument(
        '--exp_name', type=str, default=default_exp_name, help='Name of the experiment.')
    parser.add_argument('--snapshot_mode', type=str, default='none',
                        help='Mode to save the snapshot. Can be either "all" '
                             '(all iterations will be saved), "last" (only '
                             'the last iteration will be saved), or "none" '
                             '(do not save snapshots)')
    parser.add_argument('--tabular_log_file', type=str, default='progress.csv',
                        help='Name of the tabular log file (in csv).')
    parser.add_argument('--text_log_file', type=str, default='debug.log',
                        help='Name of the text log file (in pure text).')
    parser.add_argument('--params_log_file', type=str, default='params.json',
                        help='Name of the parameter log file (in json).')
    parser.add_argument('--variant_log_file', type=str, default='variant.json',
                        help='Name of the variant log file (in json).')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Name of the pickle file to resume experiment from.')
    parser.add_argument('--log_tabular_only', type=ast.literal_eval, default=True,
                        help='Whether to only print the tabular log information (in a horizontal format)')
    parser.add_argument('--seed', type=int,
                        help='Random seed for numpy')
    parser.add_argument('--args_data', type=str,
                        help='Pickled data for stub objects')

    args = parser.parse_args(argv[1:])
    refresh_ids()
    print "WORKERS", WORKERS
    print "NOTE: Workers are being pinned linearly for use on EC2 m4 machines"
    [pin.remote(n) for n in range(WORKERS)] # for use on EC2

    global tabular_log_file, text_log_file

    tabular_log_file = osp.join(log_dir, args.tabular_log_file)
    text_log_file = osp.join(log_dir, args.text_log_file)
    params_log_file = osp.join(log_dir, args.params_log_file)

    logger.log_parameters_lite(params_log_file, args)
    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)
    prev_snapshot_dir = logger.get_snapshot_dir()
    prev_mode = logger.get_snapshot_mode()
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(args.snapshot_mode)
    logger.set_log_tabular_only(args.log_tabular_only)
    logger.push_prefix("[%s] " % args.exp_name)

def finish():
    logger.remove_tabular_output(tabular_log_file)
    logger.remove_text_output(text_log_file)
    logger.pop_prefix()
