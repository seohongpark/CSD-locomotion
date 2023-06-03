#!/usr/bin/env python3
import dowel_wrapper

assert dowel_wrapper is not None
import dowel

import argparse
import datetime
import functools
import os
import torch.multiprocessing as mp

import better_exceptions
import numpy as np


better_exceptions.hook()

import torch

from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.torch.distributions import TanhNormal
from garage.torch.q_functions import ContinuousMLPQFunction

from garage.replay_buffer import PathBuffer
from garagei.experiment.option_local_runner import OptionLocalRunner
from garagei.envs.consistent_normalized_env import consistent_normalize
from garagei.envs.normalized_env_ex import normalize_ex
from garagei.sampler.option_multiprocessing_sampler import OptionMultiprocessingSampler
from garagei.torch.modules.gaussian_mlp_module_ex import GaussianMLPTwoHeadedModuleEx, GaussianMLPIndependentStdModuleEx, GaussianMLPModuleEx, GaussianMixtureMLPModule
from garagei.torch.modules.normalizer import Normalizer
from garagei.torch.modules.parameter_module import ParameterModule
from garagei.torch.policies.policy_ex import PolicyEx
from garagei.torch.optimizers.optimizer_group_wrapper import OptimizerGroupWrapper
from garagei.torch.utils import xavier_normal_ex
from iod.lsd import LSD
from iod.utils import make_env_spec_for_option_policy, get_normalizer_preset


EXP_DIR = 'exp'
if os.environ.get('START_METHOD') is not None:
    START_METHOD = os.environ['START_METHOD']
else:
    START_METHOD = 'spawn'
print('START_METHOD', START_METHOD)


def get_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--run_group', type=str, required=True)
    parser.add_argument('--policy_type', type=str, default='tanhgaussian', choices=['tanhgaussian'])
    parser.add_argument('--normalizer_type', type=str, default='off',
                        choices=['off', 'garage_ex', 'consistent', 'manual', 'half_cheetah_preset', 'ant_preset', 'humanoid_preset'])
    parser.add_argument('--normalizer_obs_alpha', type=float, default=0.001)
    parser.add_argument('--normalized_env_eval_update', type=int, default=0)
    parser.add_argument('--normalizer_mean', type=float, default=0)
    parser.add_argument('--normalizer_std', type=float, default=1)

    parser.add_argument('--env', type=str, default='ant', choices=['half_cheetah', 'ant', 'humanoid',])

    parser.add_argument('--mujoco_render_hw', type=int, default=100)

    parser.add_argument('--max_path_length', type=int, default=200)

    parser.add_argument('--use_gpu', type=int, default=0, choices=[0, 1])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_parallel', type=int, default=10)
    parser.add_argument('--n_thread', type=int, default=1)

    parser.add_argument('--n_epochs', type=int, default=1000000)
    parser.add_argument('--max_optimization_epochs', type=int, default=[1], nargs='+')
    parser.add_argument('--traj_batch_size', type=int, default=20)
    parser.add_argument('--minibatch_size', type=int, default=None)
    parser.add_argument('--trans_minibatch_size', type=int, default=None)
    parser.add_argument('--trans_optimization_epochs', type=int, default=None)
    parser.add_argument('--record_metric_difference', type=int, default=0, choices=[0, 1])

    parser.add_argument('--n_epochs_per_eval', type=int, default=int(500))
    parser.add_argument('--n_epochs_per_first_n_eval', type=int, default=None)
    parser.add_argument('--custom_eval_steps', type=int, default=None, nargs='*')
    parser.add_argument('--n_epochs_per_log', type=int, default=None)
    parser.add_argument('--n_epochs_per_tb', type=int, default=None)
    parser.add_argument('--n_epochs_per_save', type=int, default=int(1000))
    parser.add_argument('--n_epochs_per_pt_save', type=int, default=None)
    parser.add_argument('--n_epochs_per_pkl_update', type=int, default=None)
    parser.add_argument('--num_eval_options', type=int, default=int(49))
    parser.add_argument('--num_eval_trajectories_per_option', type=int, default=int(4))
    parser.add_argument('--num_random_trajectories', type=int, default=int(200))
    parser.add_argument('--eval_record_video', type=int, default=int(0))
    parser.add_argument('--eval_deterministic_traj', type=int, default=1)
    parser.add_argument('--eval_deterministic_video', type=int, default=1)
    parser.add_argument('--eval_plot_axis', type=float, default=None, nargs='*')
    parser.add_argument('--video_skip_frames', type=int, default=1)

    parser.add_argument('--dim_option', type=int, default=2)

    parser.add_argument('--common_lr', type=float, default=1e-4)
    parser.add_argument('--lr_sp', type=float, default=None)
    parser.add_argument('--lr_op', type=float, default=None)
    parser.add_argument('--lr_te', type=float, default=None)

    parser.add_argument('--alpha', type=float, default=0.01)

    parser.add_argument('--sac_tau', type=float, default=5e-3)
    parser.add_argument('--sac_lr_q', type=float, default=None)
    parser.add_argument('--sac_lr_a', type=float, default=None)
    parser.add_argument('--sac_discount', type=float, default=0.99)
    parser.add_argument('--sac_scale_reward', type=float, default=1.)
    parser.add_argument('--sac_target_coef', type=float, default=1.)
    parser.add_argument('--sac_update_target_per_gradient', type=int, default=0, choices=[0, 1])
    parser.add_argument('--sac_update_with_loss_alpha_prior_opt', type=int, default=0, choices=[0, 1])
    parser.add_argument('--sac_replay_buffer', type=int, default=0, choices=[0, 1])
    parser.add_argument('--sac_max_buffer_size', type=int, default=1000000)
    parser.add_argument('--sac_min_buffer_size', type=int, default=10000)

    parser.add_argument('--spectral_normalization', type=int, default=0, choices=[0, 1])
    parser.add_argument('--spectral_coef', type=float, default=1.)

    parser.add_argument('--model_master_dim', type=int, default=None)
    parser.add_argument('--model_common_dim', type=int, default=None)
    parser.add_argument('--model_master_num_layers', type=int, default=2)
    parser.add_argument('--model_master_nonlinearity', type=str, default=None, choices=['relu', 'tanh'])
    parser.add_argument('--op_hidden_dims', type=int, default=None, nargs='*')
    parser.add_argument('--te_hidden_dims', type=int, default=None, nargs='*')
    parser.add_argument('--te_max_optimization_epochs', type=int, default=2)
    parser.add_argument('--te_trans_optimization_epochs', type=int, default=None)

    parser.add_argument('--discrete', type=int, default=0, choices=[0, 1])
    parser.add_argument('--inner', type=int, default=1, choices=[0, 1])

    parser.add_argument('--dual_reg', type=int, default=0, choices=[0, 1])
    parser.add_argument('--dual_lam', type=float, default=1.)
    parser.add_argument('--dual_slack', type=float, default=0.)
    parser.add_argument('--dual_dist', type=str, default='l2', choices=['l2', 's2_from_s'])

    return parser


args = get_argparser().parse_args()
g_start_time = int(datetime.datetime.now().timestamp())


def get_exp_name(hack_slurm_job_id_override=None):
    parser = get_argparser()

    exp_name = ''
    exp_name += f'sd{args.seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ or hack_slurm_job_id_override is not None:
        exp_name += f's_{hack_slurm_job_id_override or os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name_prefix = exp_name
    if 'SLURM_RESTART_COUNT' in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
    exp_name += f'{g_start_time}'

    exp_name_abbrs = set()
    exp_name_arguments = set()

    def list_to_str(arg_list):
        return str(arg_list).replace(",", "|").replace(" ", "").replace("'", "")

    def add_name(abbr, argument, value_dict=None, max_length=None, log_only_if_changed=True):
        nonlocal exp_name

        if abbr is not None:
            assert abbr not in exp_name_abbrs
            exp_name_abbrs.add(abbr)
        else:
            abbr = ''
        exp_name_arguments.add(argument)

        value = getattr(args, argument)
        if log_only_if_changed and parser.get_default(argument) == value:
            return
        if isinstance(value, list):
            if value_dict is not None:
                value = [value_dict.get(v) for v in value]
            value = list_to_str(value)
        elif value_dict is not None:
            value = value_dict.get(value)

        if value is None:
            value = 'X'

        if max_length is not None:
            value = str(value)[:max_length]

        if isinstance(value, str):
            value = value.replace('/', '-')

        exp_name += f'_{abbr}{value}'

    add_name(None, 'env', {
        'half_cheetah': 'CH',
        'ant': 'ANT',
        'humanoid': 'HUM',
    }, log_only_if_changed=False)

    add_name('clr', 'common_lr', log_only_if_changed=False)
    add_name('slra', 'sac_lr_a', log_only_if_changed=False)
    add_name('a', 'alpha', log_only_if_changed=False)
    add_name('sg', 'sac_update_target_per_gradient', log_only_if_changed=False)
    add_name('do', 'dim_option', log_only_if_changed=False)
    add_name('sr', 'sac_replay_buffer')
    add_name('md', 'model_master_dim')
    add_name('sdc', 'sac_discount', log_only_if_changed=False)
    add_name('ss', 'sac_scale_reward')
    add_name('ds', 'discrete')
    add_name('in', 'inner')
    add_name('dr', 'dual_reg')
    if args.dual_reg:
        add_name('dl', 'dual_lam')
        add_name('dk', 'dual_slack')
        add_name('dd', 'dual_dist', max_length=1)

    # Check lr arguments
    for key in vars(args):
        if key.startswith('lr_') or key.endswith('_lr') or '_lr_' in key:
            val = getattr(args, key)
            assert val is None or bool(val), 'To specify a lr of 0, use a negative value'

    return exp_name, exp_name_prefix


def get_log_dir():
    exp_name, exp_name_prefix = get_exp_name()
    assert len(exp_name) <= os.pathconf('/', 'PC_NAME_MAX')
    # Resolve symlinks to prevent runs from crashing in case of home nfs crashing.
    log_dir = os.path.realpath(os.path.join(EXP_DIR, args.run_group, exp_name))
    assert not os.path.exists(log_dir), f'The following path already exists: {log_dir}'

    return log_dir


def get_gaussian_module_construction(args,
                                     *,
                                     hidden_sizes,
                                     hidden_nonlinearity=torch.relu,
                                     w_init=torch.nn.init.xavier_uniform_,
                                     init_std=1.0,
                                     min_std=1e-6,
                                     max_std=None,
                                     **kwargs):
    module_kwargs = dict()
    module_cls = GaussianMLPIndependentStdModuleEx
    module_kwargs.update(dict(
        std_hidden_sizes=hidden_sizes,
        std_hidden_nonlinearity=hidden_nonlinearity,
        std_hidden_w_init=w_init,
        std_output_w_init=w_init,
        init_std=init_std,
        min_std=min_std,
        max_std=max_std,
    ))

    module_kwargs.update(dict(
        hidden_sizes=hidden_sizes,
        hidden_nonlinearity=hidden_nonlinearity,
        hidden_w_init=w_init,
        output_w_init=w_init,
        std_parameterization='exp',
        bias=True,
        spectral_normalization=args.spectral_normalization,
        spectral_coef=args.spectral_coef,
        **kwargs,
    ))
    return module_cls, module_kwargs


def create_policy(*, name, env_spec, hidden_sizes, hidden_nonlinearity=None, dim_option=None, policy_type=None):
    option_info = {
        'dim_option': dim_option,
    }

    policy_kwargs = dict(
        env_spec=env_spec,
        name=name,
        option_info=option_info,
    )
    module_kwargs = dict(
        hidden_sizes=hidden_sizes,
        layer_normalization=False,
    )
    if hidden_nonlinearity is not None:
        module_kwargs.update(hidden_nonlinearity=hidden_nonlinearity)

    if policy_type == 'tanhgaussian':
        module_cls = GaussianMLPTwoHeadedModuleEx
        module_kwargs.update(dict(
            max_std=np.exp(2.),
            normal_distribution_cls=TanhNormal,
            output_w_init=functools.partial(xavier_normal_ex, gain=1.),
            init_std=1.,
        ))
    else:
        raise NotImplementedError


    policy_cls = PolicyEx
    policy_kwargs.update(dict(
        module_cls=module_cls,
        module_kwargs=module_kwargs,
    ))

    policy = policy_cls(**policy_kwargs)

    return policy


@wrap_experiment(log_dir=get_log_dir(), name=get_exp_name()[0])
def main(ctxt=None):
    dowel.logger.log('ARGS: ' + str(args))
    if args.n_thread is not None:
        torch.set_num_threads(args.n_thread)

    set_seed(args.seed)
    runner = OptionLocalRunner(ctxt)
    max_path_length = args.max_path_length
    if args.env == 'half_cheetah':
        from envs.mujoco.half_cheetah_env import HalfCheetahEnv
        env = HalfCheetahEnv(
            render_hw=args.mujoco_render_hw,
        )
    elif args.env == 'ant':
        from envs.mujoco.ant_env import AntEnv
        env = AntEnv(
            done_allowing_step_unit=None,
            render_hw=args.mujoco_render_hw,
        )
    elif args.env == 'humanoid':
        from envs.mujoco.humanoid_env import HumanoidEnv
        env = HumanoidEnv(
            done_allowing_step_unit=None,
            render_hw=args.mujoco_render_hw,
        )
    else:
        raise NotImplementedError

    normalizer_type = args.normalizer_type
    normalizer_mean = args.normalizer_mean
    normalizer_std = args.normalizer_std
    normalizer_kwargs = {}

    if normalizer_type == 'off':
        env = consistent_normalize(env, normalize_obs=False, **normalizer_kwargs)
    elif normalizer_type == 'garage_ex':
        env = normalize_ex(env, normalize_obs=True, obs_alpha=args.normalizer_obs_alpha, **normalizer_kwargs)
    elif normalizer_type == 'consistent':
        env = consistent_normalize(env, normalize_obs=True, **normalizer_kwargs)
    elif normalizer_type == 'manual':
        env = consistent_normalize(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std, **normalizer_kwargs)
    elif normalizer_type.endswith('preset'):
        normalizer_mean, normalizer_std = get_normalizer_preset(normalizer_type)
        normalizer_type = 'manual'
        env = consistent_normalize(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std, **normalizer_kwargs)

    device = torch.device('cuda' if args.use_gpu else 'cpu')

    if normalizer_type == 'consistent':
        normalizer = Normalizer(
            shape=env.observation_space.shape,
            alpha=args.normalizer_obs_alpha,
            do_normalize=True,
        )
    else:
        normalizer = None

    if args.model_master_dim is not None:
        master_dims = [args.model_master_dim] * args.model_master_num_layers
    else:
        master_dims = None

    if args.model_common_dim is not None:
        common_dims = [args.model_common_dim] * args.model_master_num_layers
    else:
        common_dims = None

    if args.model_master_nonlinearity == 'relu':
        nonlinearity = torch.relu
    elif args.model_master_nonlinearity == 'tanh':
        nonlinearity = torch.tanh
    else:
        nonlinearity = None

    op_env_spec = make_env_spec_for_option_policy(env.spec, args.dim_option, use_option=True)
    option_policy = create_policy(
        name='option_policy',
        env_spec=op_env_spec,
        hidden_sizes=master_dims or args.op_hidden_dims or common_dims or [32, 32],
        hidden_nonlinearity=nonlinearity,
        dim_option=args.dim_option,
        policy_type=args.policy_type,
    )

    module_cls, module_kwargs = get_gaussian_module_construction(
        args,
        hidden_sizes=master_dims or args.te_hidden_dims or common_dims or [32, 32],
        hidden_nonlinearity=nonlinearity or torch.relu,
        w_init=torch.nn.init.xavier_uniform_,
        input_dim=env.spec.observation_space.flat_dim,
        output_dim=args.dim_option,
    )
    traj_encoder = module_cls(**module_kwargs)

    module_cls, module_kwargs = get_gaussian_module_construction(
        args,
        hidden_sizes=master_dims or args.te_hidden_dims or common_dims or [32, 32],
        hidden_nonlinearity=nonlinearity or torch.relu,
        w_init=torch.nn.init.xavier_uniform_,
        input_dim=env.spec.observation_space.flat_dim,
        output_dim=env.spec.observation_space.flat_dim,
        min_std=1e-6,
        max_std=1e6,
    )
    dist_predictor = module_cls(**module_kwargs)

    dual_lam = ParameterModule(torch.Tensor([np.log(args.dual_lam)]))

    def _finalize_lr(lr):
        if lr is None:
            lr = args.common_lr
        else:
            assert bool(lr), 'To specify a lr of 0, use a negative value'
        if lr < 0.0:
            dowel.logger.log(f'Setting lr to ZERO given {lr}')
            lr = 0.0
        return lr

    optimizers = {
        'option_policy': torch.optim.Adam([
            {'params': option_policy.parameters(), 'lr': _finalize_lr(args.lr_op)},
        ]),
        'traj_encoder': torch.optim.Adam([
            {'params': traj_encoder.parameters(), 'lr': _finalize_lr(args.lr_te)},
        ]),
        'dist_predictor': torch.optim.Adam([
            {'params': dist_predictor.parameters(), 'lr': _finalize_lr(args.lr_op)},
        ]),
        'dual_lam': torch.optim.Adam([
            {'params': dual_lam.parameters(), 'lr': _finalize_lr(args.lr_op)},
        ]),
    }

    if args.sac_replay_buffer:
        replay_buffer = PathBuffer(capacity_in_transitions=int(args.sac_max_buffer_size))
    else:
        replay_buffer = None

    qf1 = ContinuousMLPQFunction(
        env_spec=op_env_spec,
        hidden_sizes=master_dims or common_dims or [32, 32],
        hidden_nonlinearity=nonlinearity or torch.relu,
        layer_normalization=False,
    )
    qf2 = ContinuousMLPQFunction(
        env_spec=op_env_spec,
        hidden_sizes=master_dims or common_dims or [32, 32],
        hidden_nonlinearity=nonlinearity or torch.relu,
        layer_normalization=False,
    )
    log_alpha = ParameterModule(torch.Tensor([np.log(args.alpha)]))
    optimizers.update({
        'qf1': torch.optim.Adam([
            {'params': qf1.parameters(), 'lr': _finalize_lr(args.sac_lr_q)},
        ]),
        'qf2': torch.optim.Adam([
            {'params': qf2.parameters(), 'lr': _finalize_lr(args.sac_lr_q)},
        ]),
        'log_alpha': torch.optim.Adam([
            {'params': log_alpha.parameters(), 'lr': _finalize_lr(args.sac_lr_a)},
        ])
    })

    optimizer = OptimizerGroupWrapper(
        optimizers=optimizers,
        max_optimization_epochs=None,
        minibatch_size=args.minibatch_size,
    )

    algo_kwargs = dict(
        env_spec=env.spec,
        normalizer=normalizer,
        normalizer_type=normalizer_type,
        normalizer_mean=normalizer_mean,
        normalizer_std=normalizer_std,
        normalized_env_eval_update=args.normalized_env_eval_update,
        option_policy=option_policy,
        traj_encoder=traj_encoder,
        dist_predictor=dist_predictor,
        dual_lam=dual_lam,
        optimizer=optimizer,
        alpha=args.alpha,
        max_path_length=args.max_path_length,
        max_optimization_epochs=args.max_optimization_epochs,
        n_epochs_per_eval=args.n_epochs_per_eval,
        n_epochs_per_first_n_eval=args.n_epochs_per_first_n_eval,
        custom_eval_steps=args.custom_eval_steps,
        n_epochs_per_log=args.n_epochs_per_log or 1,
        n_epochs_per_tb=args.n_epochs_per_tb or args.n_epochs_per_eval,
        n_epochs_per_save=args.n_epochs_per_save,
        n_epochs_per_pt_save=args.n_epochs_per_eval if args.n_epochs_per_pt_save is None else args.n_epochs_per_pt_save,
        n_epochs_per_pkl_update=args.n_epochs_per_eval if args.n_epochs_per_pkl_update is None else args.n_epochs_per_pkl_update,
        dim_option=args.dim_option,
        num_eval_options=args.num_eval_options,
        num_eval_trajectories_per_option=args.num_eval_trajectories_per_option,
        num_random_trajectories=args.num_random_trajectories,
        eval_record_video=args.eval_record_video,
        video_skip_frames=args.video_skip_frames,
        eval_deterministic_traj=args.eval_deterministic_traj,
        eval_deterministic_video=args.eval_deterministic_video,
        eval_plot_axis=args.eval_plot_axis,
        name='LSD',
        device=device,
        num_train_per_epoch=1,
        record_metric_difference=args.record_metric_difference,
        te_max_optimization_epochs=args.te_max_optimization_epochs,
        te_trans_optimization_epochs=args.te_trans_optimization_epochs,
        trans_minibatch_size=args.trans_minibatch_size,
        trans_optimization_epochs=args.trans_optimization_epochs,
        discount=args.sac_discount,
        discrete=args.discrete,
    )

    algo = LSD(
        **algo_kwargs,
        qf1=qf1,
        qf2=qf2,
        log_alpha=log_alpha,
        tau=args.sac_tau,
        scale_reward=args.sac_scale_reward,
        target_coef=args.sac_target_coef,
        update_target_per_gradient=args.sac_update_target_per_gradient,

        replay_buffer=replay_buffer,
        min_buffer_size=args.sac_min_buffer_size,
        inner=args.inner,

        dual_reg=args.dual_reg,
        dual_slack=args.dual_slack,
        dual_dist=args.dual_dist,
    )

    algo.option_policy.cpu()
    runner.setup(
        algo=algo,
        env=env,
        sampler_cls=OptionMultiprocessingSampler,
        sampler_args=dict(n_thread=args.n_thread),
        n_workers=args.n_parallel,
    )
    algo.option_policy.to(device)
    runner.train(n_epochs=args.n_epochs, batch_size=args.traj_batch_size)


if __name__ == '__main__':
    mp.set_start_method(START_METHOD)
    main()
