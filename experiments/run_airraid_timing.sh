#!/bin/bash


for i in {1..15}
do
    ipython experiments/airraid/trpo_airraid.py $i
done

