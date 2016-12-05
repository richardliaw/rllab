#!/bin/bash

for x in {1..3}
do
	for i in {15..1}
	do
	   out=$(( $i % 4 ))
	   if [ $out -eq 0 ] 
	   then
	    # ipython ray_experiments/airraid/trpo_ray_cores_waiting.py $i 0 #no wait
	    # ipython ray_experiments/airraid/trpo_ray_cores_waiting.py $i 1 #no wait
	    ipython ray_experiments/airraid/trpo_ray_cores_waiting.py $i 2 #no wait
	   else
		echo "$i is ODD number"	
	   fi	
	done
done