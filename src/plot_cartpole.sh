for z in {1..10}; do
	python3 run_cartpole_video_RBF.py --basis_function 200 --reg_param 0.1 --reg_opt l2 --samples_episodes 200
	python3 run_cartpole_video_RBF.py --basis_function 400 --reg_param 0.1 --reg_opt l2 --samples_episodes 200
	python3 run_cartpole_video_RBF.py --basis_function 800 --reg_param 0.1 --reg_opt l2 --samples_episodes 200
	
	python3 run_cartpole_video_Laplace.py --basis_function 10 --reg_opt wl1 --samples_episodes 200
	python3 run_cartpole_video_Laplace.py --basis_function 20 --reg_opt wl1 --samples_episodes 200

	python3 run_cartpole_video_Laplace.py --basis_function 10 --reg_param 0.1 --reg_opt l1 --samples_episodes 200
	python3 run_cartpole_video_Laplace.py --basis_function 10 --reg_param 0.01 --reg_opt l1 --samples_episodes 200
	python3 run_cartpole_video_Laplace.py --basis_function 10 --reg_param 0.1 --reg_opt l1 --samples_episodes 200

	python3 run_cartpole_video_Laplace.py --basis_function 20 --reg_param 0.01 --reg_opt l1 --samples_episodes 200
	python3 run_cartpole_video_Laplace.py --basis_function 20 --reg_param 0.1 --reg_opt l1 --samples_episodes 200

	python3 run_cartpole_video_Laplace.py --basis_function 10 --reg_param 0.01 --reg_opt l2 --samples_episodes 200
done



# for z in {1..2}; do
# 	for b in 100 200 400; do
# 		python3 run_cartpole_video_RBF.py --basis_function $b --reg_opt wl1 --samples_episodes 200
# 		for r in 0.001 0.01 0.1; do
# 			for o in 'l1' 'l2'; do
# 				python3 run_cartpole_video_RBF.py --basis_function $b --reg_param $r --reg_opt $o --samples_episodes 200
# 			done
# 		done
# 	done
# done


# for z in {1..2}; do
# 	for b in 10 20; do
# 		python3 run_cartpole_video_Laplace.py --basis_function $b --reg_opt wl1 --samples_episodes 200
# 		for r in 0.001 0.01 0.1; do
# 			for o in 'l1' 'l2'; do
# 				python3 run_cartpole_video_Laplace.py --basis_function $b --reg_param $r --reg_opt $o --samples_episodes 200
# 			done
# 		done
# 	done
# done

# for z in {1..2}; do
# 	for b in 100 200 400; do
# 		python3 run_cartpole_video_RBF.py --basis_function $b --reg_opt wl1 --samples_episodes 200
# 	done
# done

# for z in {1..2}; do
# 	for b in 10 20; do
# 		python3 run_cartpole_video_Laplace.py --basis_function $b --reg_opt wl1 --samples_episodes 1000
# 	done
# done