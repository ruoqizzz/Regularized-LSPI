#!/bin/sh
for i in 20 40 100 200 500 1000
do
	for j in 0.2 0.3 0.4 0.5 0.6 0.7
	do 
		python3 run_LQR_estimation.py --basis_function_dim $i --reg_param $j --reg_opt 'l1'
		# python3 run_LinearRegression.py --basis_function_dim $i --reg_param $j --reg_opt 'l2'
	done
done