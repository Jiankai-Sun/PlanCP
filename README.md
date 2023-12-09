# PlanCP

Our code is based on [diffuser](https://github.com/jannerm/diffuser)

## Setup

#### Create environment

Install [Conda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.

```bash
# For Maze2D
cd diffuser_maze2d
conda env create -f environment_maze2d.yml
```

## How to Run
#### diffuser_maze2d
```
cd diffuser_maze2d
bash train.sh
# train
#python scripts/train.py --config config.maze2d --dataset maze2d-large-v1
#python scripts/train.py --config config.maze2d --dataset maze2d-medium-v1
python scripts/train.py --config config.maze2d --dataset maze2d-umaze-v1
# eval single task
#python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-large-v1
#python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-medium-v1
python scripts/plan_maze2d.py --config config.maze2d --dataset maze2d-umaze-v1
```
Evaluate for 100 epochs for mean and std
```
cd diffuser_maze2d
bash run_maze2d_planning.sh
cd scripts
python read_results_maze2d.py
```

# Reference
**[Conformal Prediction for Uncertainty-Aware Planning with Diffusion Dynamics Model](https://openreview.net/pdf?id=VeO03T59Sh)**
<br />
[Jiankai Sun](https://scholar.google.com/citations?user=726MCb8AAAAJ&hl=en),
[Yiqi Jiang](), 
[Jianing Qiu](), 
[Parth Talpur Nobel](),
[Mykel Kochenderfer](),
and
[Mac Schwager](http://web.stanford.edu/~schwager/)
<br />
**In Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS 2023)**
<br />
[[Paper]](https://openreview.net/pdf?id=VeO03T59Sh)[[Code]](https://github.com/Jiankai-Sun/PlanCP)

```
@ARTICLE{sun2023conformal,
     author={J. {Sun} and Y. {Jiang} and J. {Qiu} and P. {Nobel} and M. {Kochenderfer} and M. {Schwager}},
     journal={Thirty-seventh Conference on Neural Information Processing Systems},
     title={Conformal Prediction for Uncertainty-Aware Planning with Diffusion Dynamics Model},
     year={2023},
}
```

## Acknowledgement
- [diffuser](https://github.com/jannerm/diffuser)