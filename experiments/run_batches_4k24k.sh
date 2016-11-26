#!/bin/bash


for i in {1..3}
do
    ipython experiments/airraid/trpo_airraid_batch.py 50000
    ipython experiments/airraid/trpo_airraid_batch.py 100000
    ipython experiments/airraid/trpo_airraid_batch.py 200000
done

