import torch
import torch.nn

from diffusion_policy.model.common.normalizer import LinearNormalizer

# ============================================================================
# 数据集接口定义（任务侧与策略侧的「统一契约」，见 README「Codebase Tutorial」）
#
# 设计目标：实现 N 个任务 + M 个方法只需 O(N+M) 代码。因此所有任务的数据集
# 都继承这两个基类，输出统一格式的样本；所有策略也遵循同一接口消费数据。
#
# 低维接口：
#   BaseLowdimDataset.__getitem__ 返回:
#     'obs'   : Tensor (T, Do)    —— 归一化前的观察序列
#     'action': Tensor (T, Da)    —— 动作序列
#   get_normalizer() 返回 LinearNormalizer（含 'obs','action' 两个 key）。
#
# 图像接口：
#   BaseImageDataset.__getitem__ 返回:
#     'obs'   : Dict{key: Tensor (T, *)}   —— 图像 (T,H,W,3) 与低维 (T,D)
#     'action': Tensor (T, Da)
#   get_normalizer() 返回 LinearNormalizer（每个 obs key 一个，外加 'action'）。
#
# 注意：normalizer 是训练/推理保持一致的关键，也是最常见的 bug 来源。
# ============================================================================


class BaseLowdimDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseLowdimDataset':
        # 默认返回空数据集（各子类会重写，切出验证集）
        return BaseLowdimDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        output:
            obs: T, Do
            action: T, Da
        """
        raise NotImplementedError()


class BaseImageDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseLowdimDataset':
        # 默认返回空数据集（各子类会重写）
        return BaseImageDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        output:
            obs:
                key: T, *
            action: T, Da
        """
        raise NotImplementedError()
