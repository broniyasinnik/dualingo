.PHONY: debug train evaluate

experiment_name=baseline2
gpu=0
checkpoint=best
results_dir=Results

train:
	CUDA_VISIBLE_DEVICES=${gpu} python train.py --data_dir data/en_hu --model_dir experiments/${experiment_name}\
                    --tensorboard_dir experiments/${experiment_name}/runs \
                    --restore_file best

evaluate:
	CUDA_VISIBLE_DEVICES=${gpu} python evaluation.py --data_dir data/en_hu \
												   --model_dir experiments/${experiment_name} \
												   --checkpoint ${checkpoint} \
												   --results_dir experiments/${experiment_name}/${results_dir}