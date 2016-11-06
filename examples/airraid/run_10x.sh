#!/bin/bash
for i in {1..10}
do
    ipython examples/airraid/trpo_airraid.py $1
done

