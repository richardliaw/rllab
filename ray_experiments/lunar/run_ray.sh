#!/bin/bash

for x in {1..4}
do
  for i in {15..1}
  do
     out=$(( $i % 4 ))
     if [ $out -eq 0 ] 
     then
      ipython ray_experiments/lunar/trpo_ray.py $i 0 # no wait
      ipython ray_experiments/lunar/trpo_ray.py $i 1 # wait
      # ipython ray_experiments/lunar/trpo_ray.py $i 2 # high_usage # should fail
      # ipython ray_experiments/lunar/trpo_ray.py $i 3 # cp 
     else
  	echo "$i is ODD number"	
     fi	
  done
done