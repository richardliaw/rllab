#!/bin/bash
for n in {1..15}
do
   out=$(( $n % 2 ))
   if [ $out -eq 0 ] 
   then
    ipython experiments/airraid/trpo_airraid_bigbatch.py $i
   else
	echo "$n is ODD number"	
   fi	
done