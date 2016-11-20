#!/bin/bash

for i in {1..5}
do
    ipython experiments/airraid/trpo_airraid_batch.py 8000
    ipython experiments/airraid/trpo_airraid_batch.py 12000
    ipython experiments/airraid/trpo_airraid_batch.py 16000
done

