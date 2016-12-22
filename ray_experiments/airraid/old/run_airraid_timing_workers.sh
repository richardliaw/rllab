#!/bin/bash
for i in {1..15}
do
   out=$(( $i % 2 ))
   if [ $out -eq 0 ] 
   then
    ipython ray_experiments/airraid/trpo_airraid_workers.py $i
   else
	echo "$i is ODD number"	
   fi	
done
