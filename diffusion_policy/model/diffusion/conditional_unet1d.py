import logging

import einops
import torch
from einops.layers.torch import Rearrange
from torch import nn

from diffusion_policy.model.diffusion.conv1d_components import (
    Conv1dBlock,
    Downsample1d,
    Upsample1d,
)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

# ============================================================================
# ConditionalUnet1D —— 1D 时序 U-Net（论文 Fig.1b 的 CNN 骨干）
#
# 输入是形状 (B, T, input_dim) 的动作序列（T=horizon）。
# 结构与图像 U-Net 相同，只是把 2D 卷积换成 1D 卷积：
#   down_modules（逐层下采样）→ mid_modules（瓶颈）→ up_modules（逐层上采样+skip）→ final_conv
#
# 条件注入（FiLM，Feature-wise Linear Modulation，论文 Fig.1b）：
#   每个 ConditionalResidualBlock1D 会把「条件向量 cond」经 MLP 编码成
#   per-channel 的 scale 与 bias，再对特征做 scale*out+bias 调制。
#   cond 由两部分拼接：扩散步 k 的正弦 embedding + 全局观察特征（global_cond）。
# ============================================================================


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()

        # 两个 1D 卷积残差块（GroupNorm + Mish + Conv1d）
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM 调制 https://arxiv.org/abs/1709.07871
        # 把条件向量 cond 编码成 per-channel 的 scale（与可选 bias）
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        # cond_encoder: (B, cond_dim) -> (B, cond_channels, 1)，广播到时间维
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )

        # 残差连接：输入输出通道不一致时用 1x1 卷积对齐
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            # 预测 scale 与 bias，做 FiLM 调制：out = scale*out + bias
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            # 只加一个偏置
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()
        # all_dims: 各层通道数（输入维度 + 下采样通道序列）
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        # 扩散步 k 的正弦位置编码 → MLP → 特征向量
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        # 若提供全局条件（观察特征），拼接到 cond 后面
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        # 相邻层的 (in_channels, out_channels) 对
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # 局部条件编码器（FiLM 的另一种用法：观察序列作为局部条件）
        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList(
                [
                    # down encoder
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                    # up encoder
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                ]
            )

        # 瓶颈层（最低分辨率处的两个残差块）
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
            ]
        )

        # 下采样路径：每个阶段 [残差块, 残差块, 下采样]
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # 上采样路径：每个阶段 [残差块(2x通道 skip concat), 残差块, 上采样]
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        # 最终输出卷积：回到 input_dim 维度（预测噪声或样本）
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float,
        local_cond=None,
        global_cond=None,
        **kwargs,
    ):
        """
        x: (B,T,input_dim)         —— 待去噪的动作轨迹（noisy action sequence）
        timestep: (B,) or int, diffusion step   —— 扩散步 k
        local_cond: (B,T,local_cond_dim)        —— 局部条件（可选，FiLM）
        global_cond: (B,global_cond_dim)        —— 全局条件（观察特征，可选）
        output: (B,T,input_dim)   —— 预测的噪声 ε_θ
        """
        sample = einops.rearrange(sample, "b h t -> b t h")

        # 1. 处理扩散步 k：转成 batch 维一致的 tensor，再编码为正弦 embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # 标量 → tensor（CPU/GPU 同步，尽量直接传 tensor 避免）
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # 广播到 batch 维（兼容 ONNX/Core ML 的方式）
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        # 拼接全局条件（观察特征）
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        # 编码局部条件（FiLM 的观察序列版本）
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, "b h t -> b t h")
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)

        x = sample
        h = []
        # 下采样路径（记录每层输出用于 skip connection）
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # 瓶颈
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # 上采样路径（skip connection 拼接）
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            # 正确条件应为 idx == (len(self.up_modules)-1) 且 len(h_local)>0，
            # 但修改会破坏与已发布 checkpoint 的兼容性，故保留注释。
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x
