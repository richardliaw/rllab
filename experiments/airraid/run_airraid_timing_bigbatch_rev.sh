#!/bin/bash
for i in {15..1}
do
   out=$(( $i % 2 ))
   if [ $out -eq 0 ] 
   then
    ipython experiments/airraid/trpo_airraid_bigbatch.py $i
   else
	echo "$i is ODD number"	
   fi	
done