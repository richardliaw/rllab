#!/bin/bash


for i in {15..1}
do
    ipython experiments/airraid/trpo_airraid.py $i
done

