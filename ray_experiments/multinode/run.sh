#!/bin/bash

for x in {1..3}
do
	ipython ray_experiments/multinode/trpo_ray.py $1
done
