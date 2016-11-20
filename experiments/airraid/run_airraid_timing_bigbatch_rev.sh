#!/bin/bash
for n in {15..1}
do
   out=$(( $n % 2 ))
   if [ $out -eq 0 ] 
   then
    ipython experiments/airraid/trpo_airraid_bigbatch.py $i
   else
	echo "$n is ODD number"	
   fi	
done