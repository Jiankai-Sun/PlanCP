# train
python scripts/train.py --config config.maze2d --dataset maze2d-large-v1
python scripts/train.py --config config.maze2d --dataset maze2d-medium-v1
python scripts/train.py --config config.maze2d --dataset maze2d-umaze-v1
# eval single task
#python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1 # --diffusion_loadpath diffusion/H384_T256_20230315185353
#python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-medium-v1
#python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1