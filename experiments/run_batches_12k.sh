#!/bin/bash

for i in {1..5}
do
    ipython experiments/airraid/trpo_airraid_batch.py 12000
    ipython experiments/airraid/trpo_airraid_batch.py 14000
    ipython experiments/airraid/trpo_airraid_batch.py 16000
    ipython experiments/airraid/trpo_airraid_batch.py 18000
    ipython experiments/airraid/trpo_airraid_batch.py 20000
done

