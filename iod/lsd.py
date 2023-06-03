import numpy as np
import torch

import global_context
from garage import TrajectoryBatch
from garagei import log_performance_ex
from iod import sac_utils
from iod.iod import IOD
import copy

from iod.utils import get_torch_concat_obs, to_np_object_arr, FigManager, get_option_colors, record_video, \
    draw_2d_gaussians


class LSD(IOD):
    def __init__(
            self,
            *,
            qf1,
            qf2,
            log_alpha,
            tau,
            scale_reward,
            target_coef,
            update_target_per_gradient,

            replay_buffer,
            min_buffer_size,
            inner,

            dual_reg,
            dual_slack,
            dual_dist,

            **kwargs,
    ):
        super().__init__(**kwargs)

        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.log_alpha = log_alpha.to(self.device)

        self.tau = tau
        self.update_target_per_gradient = update_target_per_gradient

        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size
        self.inner = inner

        self.dual_reg = dual_reg
        self.dual_slack = dual_slack
        self.dual_dist = dual_dist

        self.param_modules.update(
            qf1=self.qf1,
            qf2=self.qf2,
            log_alpha=self.log_alpha
        )

        self._reward_scale_factor = scale_reward
        self._target_entropy = -np.prod(self._env_spec.action_space.shape).item() / 2. * target_coef

    @property
    def policy(self):
        return {
            'option_policy': self.option_policy,
        }

    def _get_concat_obs(self, obs, option):
        return get_torch_concat_obs(obs, option)

    def _get_train_trajectories_kwargs(self, runner):
        if self.discrete:
            extras = self._generate_option_extras(np.eye(self.dim_option)[np.random.randint(0, self.dim_option, runner._train_args.batch_size)])
        else:
            random_options = np.random.randn(runner._train_args.batch_size, self.dim_option)
            extras = self._generate_option_extras(random_options)

        return dict(
            extras=extras,
            sampler_key='option_policy',
        )

    def _update_inputs(self, data, tensors, v):
        super()._update_inputs(data, tensors, v)

        options = list(data['option'])
        traj_options = torch.stack([x[0] for x in options], dim=0)
        assert traj_options.size() == (v['num_trajs'], self.dim_option)
        options_flat = torch.cat(options, dim=0)

        cat_obs_flat = self._get_concat_obs(v['obs_flat'], options_flat)

        next_options = list(data['next_option'])
        next_options_flat = torch.cat(next_options, dim=0)
        next_cat_obs_flat = self._get_concat_obs(v['next_obs_flat'], next_options_flat)

        v.update({
            'traj_options': traj_options,
            'options_flat': options_flat,
            'cat_obs_flat': cat_obs_flat,
            'next_cat_obs_flat': next_cat_obs_flat,
        })

    def _update_replay_buffer(self, data):
        if self.replay_buffer is not None:
            # Add paths to the replay buffer
            for i in range(len(data['actions'])):
                path = {}
                for key in data.keys():
                    cur_list = data[key][i]
                    if isinstance(cur_list, torch.Tensor):
                        cur_list = cur_list.detach().cpu().numpy()
                    if cur_list.ndim == 1:
                        cur_list = cur_list[..., np.newaxis]
                    elif cur_list.ndim == 0:  # valids
                        continue
                    path[key] = cur_list
                self.replay_buffer.add_path(path)

    def _sample_replay_buffer(self):
        samples = self.replay_buffer.sample_transitions(self._trans_minibatch_size)
        data = {}
        for key, value in samples.items():
            if value.shape[1] == 1:
                value = np.squeeze(value, axis=1)
            data[key] = to_np_object_arr([torch.from_numpy(value).float().to(self.device)])
        data['valids'] = [self._trans_minibatch_size]
        self._compute_reward(data)

        assert len(data['obs']) == 1
        assert self.normalizer_type not in ['consistent', 'garage_ex']

        tensors = {}
        internal_vars = {
            'maybe_no_grad': {},
        }

        self._update_inputs(data, tensors, internal_vars)

        return data, tensors, internal_vars

    def _train_once_inner(self, data):
        self._update_replay_buffer(data)

        self._compute_reward(data)

        for minibatch in self._optimizer.get_minibatch(data, max_optimization_epochs=self.max_optimization_epochs[0]):
            self._train_op_with_minibatch(minibatch)

        for minibatch in self._optimizer.get_minibatch(data, max_optimization_epochs=self.te_max_optimization_epochs):
            self._train_te_with_minibatch(minibatch)

        if not self.update_target_per_gradient:
            sac_utils.update_targets(self)

    def _train_te_with_minibatch(self, data):
        tensors, internal_vars = self._compute_common_tensors(data)

        if self.te_trans_optimization_epochs is None:
            assert self.replay_buffer is None
            self._optimize_te(tensors, internal_vars)
        else:
            if self.replay_buffer is None:
                num_transitions = internal_vars['num_transitions']
                for _ in range(self.te_trans_optimization_epochs):
                    mini_tensors, mini_internal_vars = self._get_mini_tensors(
                        tensors, internal_vars, num_transitions, self._trans_minibatch_size
                    )
                    self._optimize_te(mini_tensors, mini_internal_vars)
            else:
                if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:
                    for i in range(self.te_trans_optimization_epochs):
                        data, tensors, internal_vars = self._sample_replay_buffer()
                        self._optimize_te(tensors, internal_vars)

    def _train_op_with_minibatch(self, data):
        tensors, internal_vars = self._compute_common_tensors(data)

        if self._trans_optimization_epochs is None:
            assert self.replay_buffer is None
            self._optimize_op(tensors, internal_vars)
        else:
            if self.replay_buffer is None:
                num_transitions = internal_vars['num_transitions']
                for _ in range(self._trans_optimization_epochs):
                    mini_tensors, mini_internal_vars = self._get_mini_tensors(
                        tensors, internal_vars, num_transitions, self._trans_minibatch_size
                    )
                    self._optimize_op(mini_tensors, mini_internal_vars)
            else:
                if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:
                    for _ in range(self._trans_optimization_epochs):
                        data, tensors, internal_vars = self._sample_replay_buffer()
                        self._optimize_op(tensors, internal_vars)

    def _optimize_te(self, tensors, internal_vars):
        self._update_loss_te(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossTe'],
            optimizer_keys=['traj_encoder'],
        )

        if self.dual_reg:
            self._update_loss_dual_lam(tensors, internal_vars)
            self._gradient_descent(
                tensors['LossDualLam'],
                optimizer_keys=['dual_lam'],
            )
            if self.dual_dist != 'l2':
                self._gradient_descent(
                    tensors['LossDp'],
                    optimizer_keys=['dist_predictor'],
                )

    def _optimize_op(self, tensors, internal_vars):
        self._update_loss_qf(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossQf1'],
            optimizer_keys=['qf1'],
        )
        self._gradient_descent(
            tensors['LossQf2'],
            optimizer_keys=['qf2'],
        )

        self._update_loss_op(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossSacp'],
            optimizer_keys=['option_policy'],
        )

        self._update_loss_alpha(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossAlpha'],
            optimizer_keys=['log_alpha'],
        )

        if self.update_target_per_gradient:
            sac_utils.update_targets(self)

    def _compute_common_tensors(self, data, *, compute_extra_metrics=False, op_compute_chunk_size=None):
        tensors = {}
        internal_vars = {}

        self._update_inputs(data, tensors, internal_vars)

        if compute_extra_metrics:
            self._update_loss_te(tensors, internal_vars)
            if self.dual_reg:
                self._update_loss_dual_lam(tensors, internal_vars)
            self._compute_reward(data, metric_tensors=tensors)
            self._update_loss_qf(tensors, internal_vars)
            self._update_loss_op(tensors, internal_vars)
            self._update_loss_alpha(tensors, internal_vars)

        return tensors, internal_vars

    def _get_rewards(self, tensors, v):
        obs_flat = v['obs_flat']
        next_obs_flat = v['next_obs_flat']

        if self.inner:
            # Only use the mean of the distribution
            cur_z = self.traj_encoder(obs_flat).mean
            next_z = self.traj_encoder(next_obs_flat).mean
            target_z = next_z - cur_z

            if self.discrete:
                masks = (v['options_flat'] - v['options_flat'].mean(dim=1, keepdim=True)) * (self.dim_option) / (self.dim_option - 1 if self.dim_option != 1 else 1)
                rewards = (target_z * masks).sum(dim=1)
            else:
                inner = (target_z * v['options_flat']).sum(dim=1)
                rewards = inner

            # For dual LSD
            v.update({
                'cur_z': cur_z,
                'next_z': next_z,
            })
        else:
            target_dists = self.traj_encoder(next_obs_flat)

            if self.discrete:
                logits = target_dists.mean
                rewards = -torch.nn.functional.cross_entropy(logits, v['options_flat'].argmax(dim=1), reduction='none')
            else:
                rewards = target_dists.log_prob(v['options_flat'])

        tensors.update({
            'RewardMean': rewards.mean(),
            'RewardStd': rewards.std(),
        })

        return rewards

    def _update_loss_te(self, tensors, v):
        rewards = self._get_rewards(tensors, v)

        obs_flat = v['obs_flat']
        next_obs_flat = v['next_obs_flat']

        if self.dual_dist != 'l2':
            s2_dist = self.dist_predictor(obs_flat)
            loss_dp = -s2_dist.log_prob(next_obs_flat - obs_flat).mean()
            tensors.update({
                'LossDp': loss_dp,
            })

        if self.dual_reg:
            dual_lam = self.dual_lam.param.exp()
            x = obs_flat
            y = next_obs_flat
            phi_x = v['cur_z']
            phi_y = v['next_z']

            if self.dual_dist == 'l2':
                cst_dist = torch.square(y - x).mean(dim=1)
            else:
                s2_dist = self.dist_predictor(obs_flat)
                s2_dist_mean = s2_dist.mean
                s2_dist_std = s2_dist.stddev
                scaling_factor = 1. / s2_dist_std
                geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
                normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
                cst_dist = torch.mean(torch.square((y - x) - s2_dist_mean) * normalized_scaling_factor, dim=1)

                tensors.update({
                    'ScalingFactor': scaling_factor.mean(dim=0),
                    'NormalizedScalingFactor': normalized_scaling_factor.mean(dim=0),
                })

            cst_penalty = cst_dist - torch.square(phi_y - phi_x).mean(dim=1)
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)
            rewards = rewards + dual_lam.detach() * cst_penalty

            v.update({
                'cst_penalty': cst_penalty
            })
            tensors.update({
                'DualCstPenalty': cst_penalty.mean(),
            })

        reward_mean = rewards.mean()

        loss_te = -reward_mean

        v.update({
            'rewards': rewards,
            'reward_mean': reward_mean,
        })

        tensors.update({
            'LossTe': loss_te,
        })

    def _update_loss_dual_lam(self, tensors, v):
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (v['cst_penalty'].detach()).mean()

        tensors.update({
            'DualLam': dual_lam,
            'LossDualLam': loss_dual_lam,
        })

    def _update_loss_qf(self, tensors, v):
        processed_cat_obs_flat = self.option_policy.process_observations(v['cat_obs_flat'])
        next_processed_cat_obs_flat = self.option_policy.process_observations(v['next_cat_obs_flat'])

        sac_utils.update_loss_qf(
            self, tensors, v,
            obs_flat=processed_cat_obs_flat,
            actions_flat=v['actions_flat'],
            next_obs_flat=next_processed_cat_obs_flat,
            dones_flat=v['dones_flat'],
            rewards_flat=v['rewards_flat'] * self._reward_scale_factor,
            policy=self.option_policy,
        )

        v.update({
            'processed_cat_obs_flat': processed_cat_obs_flat,
            'next_processed_cat_obs_flat': next_processed_cat_obs_flat,
        })

    def _update_loss_op(self, tensors, v):
        sac_utils.update_loss_sacp(
            self, tensors, v,
            obs_flat=v['processed_cat_obs_flat'],
            policy=self.option_policy,
        )

    def _update_loss_alpha(self, tensors, v):
        sac_utils.update_loss_alpha(
            self, tensors, v,
        )

    def _compute_reward(self, data, metric_tensors=None):
        tensors = {}
        v = {}

        self._update_inputs(data, tensors, v)

        with torch.no_grad():
            rewards = self._get_rewards(tensors, v)

            if metric_tensors is not None:
                metric_tensors.update({
                    'LsdTotalRewards': rewards.mean(),
                })
            rewards = rewards.split(v['valids'], dim=0)

            data['rewards'] = to_np_object_arr(rewards)

    def _prepare_for_evaluate_policy(self, runner):
        return {}

    def _evaluate_policy(self, runner, **kwargs):
        if self.discrete:
            random_options = np.eye(self.dim_option)
            random_options = random_options.repeat(self.num_eval_trajectories_per_option, axis=0)
            colors = np.arange(0, self.dim_option)
            colors = colors.repeat(self.num_eval_trajectories_per_option, axis=0)
            num_evals = len(random_options)
            from matplotlib import cm
            cmap = 'tab10' if self.dim_option <= 10 else 'tab20'
            random_option_colors = []
            for i in range(num_evals):
                random_option_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
            random_option_colors = np.array(random_option_colors)
        else:
            random_options = np.random.randn(self.num_random_trajectories, self.dim_option)
            random_option_colors = get_option_colors(random_options * 4)
        random_op_trajectories = self._get_trajectories(
            runner,
            sampler_key='option_policy',
            extras=self._generate_option_extras(random_options),
            max_path_length_override=self._cur_max_path_length,
            worker_update=dict(
                _deterministic_initial_state=False,
                _deterministic_policy=self.eval_deterministic_traj,
            ),
            env_update=dict(_action_noise_std=None),
        )

        with FigManager(runner, 'TrajPlot_RandomZ') as fm:
            runner._env.render_trajectories(
                random_op_trajectories, random_option_colors, self.eval_plot_axis, fm.ax
            )

        sp_trajectories = random_op_trajectories
        data = self.process_samples(sp_trajectories)
        use_zero_options = False
        sp_option_means, sp_option_stddevs, sp_option_samples = self._get_sp_options_at_timesteps(
            data,
            use_zero_options=use_zero_options,
        )

        sp_option_colors = random_option_colors
        sp_option_sample_colors = random_option_colors

        if self.dim_option == 2:
            with FigManager(runner, f'PhiPlot') as fm:
                draw_2d_gaussians(sp_option_means, sp_option_stddevs, sp_option_colors, fm.ax)
                draw_2d_gaussians(
                    sp_option_samples,
                    [[0.03, 0.03]] * len(sp_option_samples),
                    sp_option_sample_colors,
                    fm.ax,
                    fill=True,
                    use_adaptive_axis=True,
                )
        else:
            with FigManager(runner, f'PhiPlot') as fm:
                draw_2d_gaussians(sp_option_means[:, :2], sp_option_stddevs[:, :2], sp_option_colors, fm.ax)
                draw_2d_gaussians(
                    sp_option_samples[:, :2],
                    [[0.03, 0.03]] * len(sp_option_samples),
                    sp_option_sample_colors,
                    fm.ax,
                    fill=True,
                )

        if self.eval_record_video:
            if self.discrete:
                random_options = np.eye(self.dim_option)
                random_options = random_options.repeat(2, axis=0)
            else:
                random_options = np.random.randn(9, self.dim_option)
                random_options = random_options.repeat(2, axis=0)
            video_op_trajectories = self._get_trajectories(
                runner,
                sampler_key='local_option_policy',
                extras=self._generate_option_extras(random_options),
                worker_update=dict(
                    _render=True,
                    _deterministic_initial_state=False,
                    _deterministic_policy=self.eval_deterministic_video,
                ),
            )
            record_video(runner, 'Video_RandomZ', video_op_trajectories, skip_frames=self.video_skip_frames)

        with global_context.GlobalContext({'phase': 'eval', 'policy': 'option'}):
            log_performance_ex(
                runner.step_itr,
                TrajectoryBatch.from_trajectory_list(self._env_spec, random_op_trajectories),
                discount=self.discount,
                additional_records=dict(),
                additional_prefix=type(runner._env.unwrapped).__name__,
            )
        self._log_eval_metrics(runner)
