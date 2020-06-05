# for z in {1..2}; do
# 	for b in 100 200 400; do
# 		python3 run_cartpole_video_RBF-1.py --basis_function $b --reg_opt wl1 --samples_episodes 200
# 		python3 run_cartpole_video_RBF-1.py --basis_function $b --reg_opt none --samples_episodes 200
# 		for r in 0.001 0.01 0.1; do
# 			for o in 'l1' 'l2'; do
# 				python3 run_cartpole_video_RBF-1.py --basis_function $b --reg_param $r --reg_opt $o --samples_episodes 200
# 			done
# 		done
# 	done

# 	for b in 10 20; do
# 		python3 run_cartpole_video_Laplace-1.py --basis_function $b --reg_opt wl1 --samples_episodes 200
# 		python3 run_cartpole_video_Laplace-1.py --basis_function $b --reg_opt none --samples_episodes 200
# 		for r in 0.001 0.01 0.1; do
# 			for o in 'l1' 'l2'; do
# 				python3 run_cartpole_video_Laplace-1.py --basis_function $b --reg_param $r --reg_opt $o --samples_episodes 200
# 			done
# 		done
# 	done
# done


	# for b in 20; do
	# 	# python3 run_cartpole_video_Laplace-1.py --basis_function $b --reg_opt wl1 --samples_episodes 200
	# 	python3 run_cartpole_video_Laplace-1.py --basis_function $b --reg_opt none --samples_episodes 200
	# 	for r in 0.001 0.01 0.1; do
	# 		for o in 'l1' 'l2'; do
	# 			python3 run_cartpole_video_Laplace-1.py --basis_function $b --reg_param $r --reg_opt $o --samples_episodes 200
	# 		done
	# 	done
	# done