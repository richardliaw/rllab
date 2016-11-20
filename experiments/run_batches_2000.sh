#!/bin/bash


for i in {1..5}
do
    ipython experiments/airraid/trpo_airraid_batch.py 2000
    ipython experiments/airraid/trpo_airraid_batch.py 4000
    ipython experiments/airraid/trpo_airraid_batch.py 6000
    ipython experiments/airraid/trpo_airraid_batch.py 8000
    ipython experiments/airraid/trpo_airraid_batch.py 10000
done

