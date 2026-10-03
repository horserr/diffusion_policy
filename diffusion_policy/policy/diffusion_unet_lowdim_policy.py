import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

# ============================================================================
# DiffusionUnetLowdimPolicy —— CNN 骨干的条件扩散策略（低维观察版）
#
# 这是论文的 "CNN-based Diffusion Policy" 核心实现。
# 论文公式(方法部分)：A^{k-1}_t = α( A^k_t − γ·ε_θ(O_t, A^k_t, k) + N(0,σ²I) )
#   训练：从数据里取动作序列 A^0，加噪后让网络 ε_θ 预测噪声（MSE loss）。
#   推理：从高斯噪声 A^K 出发，逐步去噪 K 步，得到干净动作序列 A^0。
#
# 观察 O_t 的三种注入方式（论文 Fig.1b 的条件机制）：
#   - obs_as_local_cond : 观察作为「局部条件」，与动作序列拼在同一时间轴上，用 FiLM 注入（论文主推的 FiLM）。
#   - obs_as_global_cond: 观察编码成单个全局向量，与扩散步 embedding 拼接后注入每个残差块。
#   - 二者皆 False      : "inpainting" 方式 —— 把观察拼在动作后面一起扩散，训练时 mask 掉未知动作部分。
#
# 关键概念：receding horizon（闭环动作序列预测）——
#   一次预测 horizon=T 步动作，但只执行 n_action_steps=Ta 步，之后重新规划。
# ============================================================================


class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler: DDPMScheduler,
        horizon,
        obs_dim,
        action_dim,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_local_cond=False,
        obs_as_global_cond=False,
        pred_action_steps_only=False,
        oa_step_convention=False,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()
        # local 与 global 两种条件方式互斥
        assert not (obs_as_local_cond and obs_as_global_cond)
        # 只预测动作（不预测观察）时，观察必须作为 global 条件单独传入
        if pred_action_steps_only:
            assert obs_as_global_cond
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        # mask_generator 用于训练时生成 "inpainting mask"：
        # 标记哪些位置是已知条件（观察部分）不需要加噪/算 loss。
        # 当观察作为 local/global 条件时 obs_dim=0，即不参与 inpainting。
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        # 观察/动作的线性归一化器（见 model/common/normalizer.py）
        self.normalizer = LinearNormalizer()
        self.horizon = horizon  # T  = 动作预测长度 T_p
        self.obs_dim = obs_dim  # Do = 观察维度
        self.action_dim = action_dim  # Da = 动作维度
        self.n_action_steps = n_action_steps  # Ta = 实际执行的步数
        self.n_obs_steps = n_obs_steps  # To = 观察长度
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        # 推理步数默认与训练步数一致；若指定更小值则等价于 DDIM 跳步加速推理。
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        """
        核心去噪采样循环（论文 eq: A^{k-1} = α(A^k − γ·ε_θ(O,A^k,k) + N(0,σ²I))）。
        condition_data/condition_mask 是已知条件（例如观察），在每步被强制写回。
        """
        model = self.model
        scheduler = self.noise_scheduler

        # 1. 从标准高斯噪声初始化 A^K
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # 2. 设定去噪时间步（num_inference_steps 可 < num_train_timesteps，即 DDIM）
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 2.1 强制施加已知条件（inpainting / 观察）
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2.2 噪声预测网络 ε_θ(O_t, A^k_t, k)
            model_output = model(
                trajectory, t, local_cond=local_cond, global_cond=global_cond
            )

            # 2.3 一步去噪：A^k -> A^{k-1}
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        # 3. 最后再强制一次条件，确保已知部分被精确保留
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        推理入口：给定观察，输出要执行的动作序列。
        obs_dict: 必须包含 "obs" 键，形状 (B, To, Do)
        result:   包含 "action" 键，形状 (B, Ta, Da)
        """

        assert 'obs' in obs_dict
        assert "past_action" not in obs_dict  # 历史动作条件暂未实现
        # 归一化观察（用训练时统计的 scale/bias）
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # 根据不同的观察注入方式构造条件数据
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # 局部条件：观察占据序列前 To 个时间步，其余位置填 0
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # 全局条件：观察展平成单个向量 (B, To*Do)
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # inpainting：观察拼在动作后面，前 To 步的观察位置作为已知条件
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # 执行扩散采样
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # 反归一化：把网络输出的归一化动作还原到真实动作空间
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # 取动作：只执行预测序列中的 Ta 步（receding horizon）
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        # inpainting 方式额外输出观察预测（调试/可视化用）
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        训练损失（论文 eq: L = MSE(ε^k, ε_θ(O_t, A^0+ε^k, k))）。
        流程：归一化 → 构造轨迹 → 随机加噪 → 预测噪声 → MSE。
        """
        # 归一化输入
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # 处理不同观察注入方式，得到完整「轨迹」（含动作与可能的观察）
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # 观察作为局部条件：n_obs_steps 之后的观察位置置 0
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            # inpainting：动作与观察沿特征维拼接
            trajectory = torch.cat([action, obs], dim=-1)

        # 生成 inpainting mask：标记哪些位置是「已知条件」，不参与加噪与 loss
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # 采样噪声 ε^k 与随机扩散步 k
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        # 前向扩散：A^0 + ε^k（按噪声调度加噪）
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # loss 只算非条件（即待预测的动作）部分
        loss_mask = ~condition_mask

        # 已知条件位置写回干净值（保持确定）
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        # 网络预测噪声 ε_θ
        pred = self.model(
            noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond
        )

        # 根据调度器的预测类型决定回归目标：预测噪声(epsilon) 或 预测干净样本(sample)
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # 逐元素 MSE，只在 loss_mask 位置求平均
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
