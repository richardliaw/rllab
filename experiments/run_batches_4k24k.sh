#!/bin/bash


for i in {1..3}
do
    ipython experiments/airraid/trpo_airraid_batch.py 4000
    ipython experiments/airraid/trpo_airraid_batch.py 24000
done

