import rllab.misc.logger as logger
import argparse
import os.path as osp
import dateutil.tz
import datetime
import pickle
try:
	import ray
	from rllab import ray_setting
except Exception:
	pass
import ast
import time
import json

log = {"timing": [], "optimization": [], "samples":[]}
_count = 0

def trydump():
	pass
	# global _count, log
	# if len(log['timing']) >= 5:
	# 	with open(osp.join(ray_setting.log_dir, "times_%d.json" % _count), "w") as f:
	# 		json.dump(log, f)
	# 	_count += 1
	# 	log = {"timing": [], "optimization": [], "samples":[]}

def multinode_save_statistics():
	client = ray.worker.global_worker.redis_client
	with open(osp.join(ray_setting.log_dir, "eventlog.pkl"), "w") as f:
		logs = [(k, client.lrange(k, 0, -1)[0]) for k in client.keys("event*")]
		pickle.dump(logs, f)

