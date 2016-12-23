#!/bin/bash

ipython ray_experiments/airraid/trpo_ray.py 2 0 # no wait
ipython ray_experiments/airraid/trpo_ray.py 4 0 # no wait
ipython ray_experiments/airraid/trpo_ray.py 8 0 # no wait

ipython ray_experiments/airraid/trpo_ray.py 2 1 # wait
ipython ray_experiments/airraid/trpo_ray.py 4 1 # wait
ipython ray_experiments/airraid/trpo_ray.py 8 1 # wait
