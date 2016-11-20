#!/bin/bash


for i in {1..15}
do
    ipython experiments/airraid/trpo_airraid_bigbatch.py $i
done

