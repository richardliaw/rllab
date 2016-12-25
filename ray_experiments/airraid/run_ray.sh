#!/bin/bash

for x in {1..3}
do
	ipython ray_experiments/airraid/trpo_ray.py 4 1 # wait
	ipython ray_experiments/airraid/trpo_ray.py 8 1 # wait
	ipython ray_experiments/airraid/trpo_ray.py 12 1 # wait
	ipython ray_experiments/airraid/trpo_ray.py 16 1 # wait
done
