import rllab.misc.logger as logger
import argparse
import os.path as osp
import dateutil.tz
import datetime
try:
	import ray
	from rllab import ray_setting
except Exception:
	pass
import ast
import time
import json

log = {"sampling": [], "optimization": []}
_count = 0

def trydump():
	global _count, log
	if len(log['sampling']) >= 10:
		with open(osp.join(ray_setting.log_dir, "times_%d.json" % _count), "w") as f:
			json.dump(log, f)
		_count += 1
		log = {"sampling": [], "optimization": []}