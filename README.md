# Controllability-Aware Unsupervised Skill Discovery


## Overview
This is the official implementation of [**Controllability-aware Skill Discovery** (**CSD**)](https://arxiv.org/abs/2302.05103) on locomotion environments (MuJoCo Ant, HalfCheetah, and Humanoid).
The codebase is based on the implementation of [LSD](https://github.com/seohongpark/LSD).
We refer to http://github.com/seohongpark/CSD-manipulation for the implementation of CSD on manipulation environments.

Please visit [our project page](https://seohong.me/projects/csd/) for videos.

## Installation

```
conda create --name csd-locomotion python=3.8
conda activate csd-locomotion
pip install -r requirements.txt
pip install -e .
pip install -e garaged --no-deps
```

## Examples

CSD Ant (16 discrete skills)
```
python tests/main.py --run_group EXP --env ant --max_path_length 200 --dim_option 16 --num_random_trajectories 200 --seed 0 --normalizer_type ant_preset --use_gpu 1 --traj_batch_size 10 --n_parallel 1 --n_epochs_per_eval 1000 --n_thread 1 --record_metric_difference 0 --n_epochs_per_tb 100 --n_epochs_per_save 10000 --n_epochs_per_pt_save 5000 --n_epochs_per_pkl_update 1000 --eval_record_video 1 --n_epochs 2000001 --n_epochs_per_log 100 --discrete 1 --sac_discount 0.99 --sac_update_target_per_gradient 1 --max_optimization_epochs 1 --trans_minibatch_size 1024 --trans_optimization_epochs 64 --te_max_optimization_epochs 1 --te_trans_optimization_epochs 32 --sac_scale_reward 10 --spectral_normalization 0 --alpha 0.03 --sac_lr_a -1 --common_lr 0.0001 --dual_reg 1 --dual_dist s2_from_s --dual_lam 3000 --dual_slack 1e-06 --eval_plot_axis -50 50 -50 50 --model_master_dim 512
```
CSD HalfCheetah (16 discrete skills)
```
python tests/main.py --run_group EXP --env half_cheetah --max_path_length 200 --dim_option 16 --num_random_trajectories 200 --seed 0 --normalizer_type half_cheetah_preset --use_gpu 1 --traj_batch_size 10 --n_parallel 1 --n_epochs_per_eval 1000 --n_thread 1 --record_metric_difference 0 --n_epochs_per_tb 100 --n_epochs_per_save 10000 --n_epochs_per_pt_save 5000 --n_epochs_per_pkl_update 1000 --eval_record_video 1 --n_epochs 2000001 --n_epochs_per_log 100 --discrete 1 --sac_discount 0.99 --sac_update_target_per_gradient 1 --max_optimization_epochs 1 --trans_minibatch_size 1024 --trans_optimization_epochs 64 --te_max_optimization_epochs 1 --te_trans_optimization_epochs 32 --sac_scale_reward 10 --spectral_normalization 0 --alpha 0.1 --sac_lr_a -1 --common_lr 0.0001 --dual_reg 1 --dual_dist s2_from_s --dual_lam 3000 --dual_slack 1e-06 --model_master_dim 512
```
CSD Humanoid (16 discrete skills)
```
python tests/main.py --run_group EXP --env humanoid --max_path_length 1000 --dim_option 16 --num_random_trajectories 200 --seed 0 --normalizer_type humanoid_preset --use_gpu 1 --traj_batch_size 5 --n_parallel 1 --n_epochs_per_eval 1000 --n_thread 1 --record_metric_difference 0 --n_epochs_per_tb 100 --n_epochs_per_save 10000 --n_epochs_per_pt_save 5000 --n_epochs_per_pkl_update 1000 --eval_record_video 1 --n_epochs 2000001 --n_epochs_per_log 100 --discrete 1 --sac_discount 0.99 --sac_update_target_per_gradient 1 --max_optimization_epochs 1 --trans_minibatch_size 1024 --trans_optimization_epochs 64 --te_max_optimization_epochs 1 --te_trans_optimization_epochs 32 --sac_scale_reward 10 --spectral_normalization 0 --alpha 0.3 --sac_lr_a -1 --common_lr 0.0003 --dual_reg 1 --dual_dist s2_from_s --dual_lam 3000 --dual_slack 1e-06 --video_skip_frames 3 --model_master_dim 1024 --sac_replay_buffer 1
```

LSD Ant (16 discrete skills)
```
python tests/main.py --run_group EXP --env ant --max_path_length 200 --dim_option 16 --num_random_trajectories 200 --seed 0 --normalizer_type ant_preset --use_gpu 1 --traj_batch_size 10 --n_parallel 1 --n_epochs_per_eval 1000 --n_thread 1 --record_metric_difference 0 --n_epochs_per_tb 100 --n_epochs_per_save 10000 --n_epochs_per_pt_save 5000 --n_epochs_per_pkl_update 1000 --eval_record_video 1 --n_epochs 2000001 --n_epochs_per_log 100 --discrete 1 --sac_discount 0.99 --sac_update_target_per_gradient 1 --max_optimization_epochs 1 --trans_minibatch_size 1024 --trans_optimization_epochs 64 --te_max_optimization_epochs 1 --te_trans_optimization_epochs 32 --sac_scale_reward 1 --spectral_normalization 1 --alpha 0.003 --sac_lr_a -1 --common_lr 0.0001 --eval_plot_axis -50 50 -50 50 --model_master_dim 512
```
LSD HalfCheetah (16 discrete skills)
```
python tests/main.py --run_group EXP --env half_cheetah --max_path_length 200 --dim_option 16 --num_random_trajectories 200 --seed 0 --normalizer_type half_cheetah_preset --use_gpu 1 --traj_batch_size 10 --n_parallel 1 --n_epochs_per_eval 1000 --n_thread 1 --record_metric_difference 0 --n_epochs_per_tb 100 --n_epochs_per_save 10000 --n_epochs_per_pt_save 5000 --n_epochs_per_pkl_update 1000 --eval_record_video 1 --n_epochs 2000001 --n_epochs_per_log 100 --discrete 1 --sac_discount 0.99 --sac_update_target_per_gradient 1 --max_optimization_epochs 1 --trans_minibatch_size 1024 --trans_optimization_epochs 64 --te_max_optimization_epochs 1 --te_trans_optimization_epochs 32 --sac_scale_reward 1 --spectral_normalization 1 --alpha 0.003 --sac_lr_a -1 --common_lr 0.0001 --model_master_dim 512
```
LSD Humanoid (16 discrete skills)
```
python tests/main.py --run_group EXP --env humanoid --max_path_length 1000 --dim_option 16 --num_random_trajectories 200 --seed 0 --normalizer_type humanoid_preset --use_gpu 1 --traj_batch_size 5 --n_parallel 1 --n_epochs_per_eval 1000 --n_thread 1 --record_metric_difference 0 --n_epochs_per_tb 100 --n_epochs_per_save 10000 --n_epochs_per_pt_save 5000 --n_epochs_per_pkl_update 1000 --eval_record_video 1 --n_epochs 2000001 --n_epochs_per_log 100 --discrete 1 --sac_discount 0.99 --sac_update_target_per_gradient 1 --max_optimization_epochs 1 --trans_minibatch_size 1024 --trans_optimization_epochs 64 --te_max_optimization_epochs 1 --te_trans_optimization_epochs 32 --sac_scale_reward 1 --spectral_normalization 1 --alpha 0.03 --sac_lr_a -1 --common_lr 0.0001 --video_skip_frames 3 --model_master_dim 1024 --sac_replay_buffer 1
```

## Comments on the Implementations of CSD and LSD

The CSD and LSD implementations in this repository, which we use to produce the results in the CSD paper,
are based on a sample-efficient version of LSD.
The main difference between [the original LSD](https://github.com/seohongpark/LSD) and this sample-efficient version is that
the latter updates the target network every gradient step, not every epoch.
This (in combination with additional hyperpameter adjustments) greatly improves the sample efficiency of LSD by ~10 times
(e.g., the original LSD uses 400M steps for Ant while this version uses 40M steps),
but it may also slightly degrade the performance.
For reproducing the results in the LSD paper,
we recommend using [the original LSD implementation](https://github.com/seohongpark/LSD).

## Licence

MIT
