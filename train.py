"""
训练入口。用法示例：
    python train.py --config-name=train_diffusion_unet_image_workspace task=pusht_image

工作流程：
    1. 用 hydra 读取 diffusion_policy/config/ 下的 YAML 配置（含命令行覆盖项）；
    2. 解析 cfg._target_ 得到对应的 Workspace 类（如 TrainDiffusionUnetImageWorkspace）；
    3. 实例化 Workspace 并调用 run() 开始训练（内部完成数据集、策略、训练循环与评测）。
"""

import sys
# 对 stdout/stderr 开启行缓冲，保证日志实时输出
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# 允许在配置里用 ${eval:''} resolver 执行任意 python 表达式
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # 立即解析配置，使所有 ${now:} 等 resolver 使用同一时间戳
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
