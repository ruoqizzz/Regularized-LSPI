#!/bin/sh
for i in 1000
do
	for j in 0.01 0.1 1.0
	do 
		python3 run_LQR_estimation.py --basis_function_dim $i --reg_param $j --reg_opt 'l1'
	done
done