#!/bin/sh
for i in 200 400 800 1000
do
	python3 run_LQR_estimation.py --basis_function_dim $i
done