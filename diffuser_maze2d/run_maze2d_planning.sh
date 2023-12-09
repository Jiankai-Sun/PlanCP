for i in {0..99};
do
  python scripts/plan_maze2d.py --dataset maze2d-large-v1 --conditional False --logbase logs --diffusion_loadpath diffusion/H384_T256_20230403232714 --suffix $i
done
mv logs/maze2d-large-v1/plans/  logs/maze2d-large-v1/plans_conditional_False/
for i in {0..99};
do
  python scripts/plan_maze2d.py --dataset maze2d-medium-v1 --conditional False --logbase logs --diffusion_loadpath diffusion/H256_T256_20230405115716 --suffix $i
done
mv logs/maze2d-medium-v1/plans/  logs/maze2d-medium-v1/plans_conditional_False/
for i in {0..99};
do
  python scripts/plan_maze2d.py --dataset maze2d-umaze-v1 --conditional False --logbase logs --diffusion_loadpath diffusion/H128_T64_20230406155851 --suffix $i
done
mv logs/maze2d-umaze-v1/plans/  logs/maze2d-umaze-v1/plans_conditional_False/