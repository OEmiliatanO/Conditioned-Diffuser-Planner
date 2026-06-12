# Conditioned Diffuser Planner
This is the official implementation of [Conditioned Diffuser Planner for Consistent State Transition](https://openreview.net/forum?id=STmcRa3EzE&invitationId=ntu.edu.tw/National_Taiwan_University/Fall_2026/ML-MiniConf/Submission4/-/Camera_Ready_Revision&referrer=%5BTasks%5D(%2Ftasks))

## Environment
Install the enviroment through miniconda:
```
conda install -f env.yml
```

## Data generation

The training requires generating the maze data first:
```
python generate_dataset.py --maze_size small --total_trajectories 1000 --save_path small_maze_data.pth
```

## Training and Inference (baseline)

```
python train_diffusion.py --data_path small_maze_data.pth --save_dir small_maze_baseline_ckpts --total_steps 10000
python zeroshot_origin_diffusion.py --maze_size small --checkpoint_path small_maze_baseline_ckpts/diffusion_final.pth --save_dir small_maze_baseline --num_samples 1000
```

## Training and Inference (CDP)

```
python train_conditional_diffusion.py --data_path small_maze_data.pth --save_dir small_maze_cdp_ckpts --total_steps 10000
python zeroshot_diffusion.py --maze_size small --checkpoint_path small_maze_cdp_ckpts/diffusion_final.pth --save_dir small_maze_cdp --num_samples 1000 --projection_enabled --lambda_r 0.005 --lambda_c 250
```
