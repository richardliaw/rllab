#!/bin/bash

for x in {1..5}
do
	ipython ray_experiments/airraid/trpo_ray.py $1
done
