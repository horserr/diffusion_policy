import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# ============================================================================
# DiffusionUnetImagePolicy —— CNN 骨干的条件扩散策略（图像观察版）
#
# 与 DiffusionUnetLowdimPolicy 逻辑几乎一致，唯一区别：
#   观察是图像字典（RGB 相机 + 低维 proprioception），先经 MultiImageObsEncoder
#   编码成特征向量 obs_feature_dim，再作为「全局条件」(obs_as_global_cond=True)
#   注入 1D 时序 U-Net。这正是论文 Fig.1b 的 CNN-based Diffusion Policy。
#
# 数据流（训练）：
#   obs(图像序列) → obs_encoder → global_cond(B, obs_feature_dim*To)
#   action 加噪 → UNet(noisy_action, k, global_cond) → 预测噪声 → MSE
# 数据流（推理）：
#   obs → obs_encoder → global_cond → 高斯噪声去噪 K 步 → 反归一化 → 取 Ta 步
# ============================================================================


class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # 解析动作维度（来自 shape_meta['action']['shape']）
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # 观察特征维度：由 obs_encoder 输出维度决定
        obs_feature_dim = obs_encoder.output_shape()[0]

        # 构造扩散模型
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            # 观察作为全局条件：UNet 输入只含动作，观察维度并入全局条件
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        # inpainting mask 生成器（obs_as_global_cond 时 obs_dim=0，即观察不进 inpainting）
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon  # T  = 动作预测长度 T_p
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps  # Ta = 执行步数
        self.n_obs_steps = n_obs_steps  # To = 观察步数
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        # 推理步数默认与训练一致；更小值即 DDIM 加速
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
        核心去噪采样循环（与低维版相同，见 DiffusionUnetLowdimPolicy.conditional_sample）。
        """
        model = self.model
        scheduler = self.noise_scheduler

        # 从高斯噪声初始化 A^K
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # 设定去噪时间步
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 强制施加已知条件
            trajectory[condition_mask] = condition_data[condition_mask]

            # 噪声预测 ε_θ(O_t, A^k_t, k)
            model_output = model(
                trajectory, t, local_cond=local_cond, global_cond=global_cond
            )

            # 一步去噪 A^k -> A^{k-1}
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        # 最后再强制条件
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        推理入口：obs_dict 含图像与低维观察（B,To,...），输出 action (B,Ta,Da)。
        """
        assert 'past_action' not in obs_dict  # not implemented yet
        # 归一化观察
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # 处理观察注入
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # 全局条件：取前 To 帧，逐帧编码后展平为单个向量
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)
            # 动作部分为空（待生成）
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # inpainting：观察特征拼在动作后面
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # 执行扩散采样
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # 反归一化动作
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # 取前 Ta 步执行（receding horizon）
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        训练损失：L = MSE(ε^k, ε_θ(O_t, A^0+ε^k, k))。观察经编码器得到全局条件。
        """
        # 归一化
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # 处理观察注入
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # 取前 n_obs_steps 帧逐帧编码，展平为全局条件
            this_nobs = dict_apply(
                nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # inpainting：观察特征拼在动作后一起扩散
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # 生成 inpainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # 采样噪声与随机扩散步
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        # 前向加噪
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # loss 只算待预测（非条件）部分
        loss_mask = ~condition_mask

        # 条件位置写回干净值
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # 网络预测噪声
        pred = self.model(
            noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond
        )

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
