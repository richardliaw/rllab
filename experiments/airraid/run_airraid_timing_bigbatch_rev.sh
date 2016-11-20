#!/bin/bash


for i in {15..1}
do
    ipython experiments/airraid/trpo_airraid_bigbatch.py $i
done

