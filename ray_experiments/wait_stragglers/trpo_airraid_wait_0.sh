#!/bin/bash
for x in {1..3}
do
	for i in {15..1}
	do
	   out=$(( $i % 2 ))
	   if [ $out -eq 0 ] 
	   then
	    ipython ray_experiments/airraid/trpo_rl0_cores_waiting.py $i
	   else
		echo "$i is ODD number"	
	   fi	
	done
done