import copy
import pathlib
import threading

import dill
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

# ============================================================================
# BaseWorkspace —— 训练/评测生命周期管理的基类（README「Key Components」）
#
# 一个 Workspace 封装一次实验的全部状态与流程：
#   - 继承 BaseWorkspace；由一个 hydra 生成的 OmegaConf 配置驱动（对应 config/*.yaml）。
#   - run() 方法包含完整训练/评测流水线。
#   - checkpoint 在 Workspace 层面进行：所有「对象属性」会在 save_checkpoint 时自动保存。
#   - 实验的其它临时状态应作为 run() 的局部变量（不参与序列化）。
#
# 保存策略：
#   - save_checkpoint: 长期存储，保存 cfg + 各模块 state_dict + 指定 pickle 对象，
#                      用 dill 序列化，异步线程写盘（避免阻塞训练）。
#   - save_snapshot:   科研快速保存/加载，整对象 torch.save，但要求代码不变。
# ============================================================================


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: str | None = None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            # 由 hydra 决定输出目录（config 里 hydra.run.dir）
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def run(self):
        """
        创建任何不应被序列化的资源都应作为局部变量放在这里。
        """

    def save_checkpoint(
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        use_thread=True,
    ):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {"cfg": self.cfg, "state_dicts": dict(), "pickles": dict()}

        # 遍历所有对象属性：有 state_dict 的（模型/优化器/sampler）存 state_dict；
        # 在 include_keys 里的对象用 dill 直接序列化。
        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            # 异步线程写盘，避免阻塞训练循环
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        return str(path.absolute())

    def get_checkpoint_path(self, tag="latest"):
        return pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload["pickles"].keys()

        # 恢复 state_dict（模型/优化器等）
        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        # 恢复 pickled 对象（如 normalizer、global_step、epoch）
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    def load_checkpoint(
        self, path=None, tag="latest", exclude_keys=None, include_keys=None, **kwargs
    ):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(
        cls, path, exclude_keys=None, include_keys=None, **kwargs
    ):
        # 从 checkpoint 直接重建一个 Workspace 实例（eval.py 用）
        payload = torch.load(open(path, "rb"), pickle_module=dill)
        instance = cls(payload["cfg"])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs,
        )
        return instance

    def save_snapshot(self, tag="latest"):
        """
        科研用快速保存/加载：保存整个 Workspace 对象。
        加载 snapshot 要求代码与保存时完全一致；长期存储请用 save_checkpoint。
        """
        path = pathlib.Path(self.output_dir).joinpath("snapshots", f"{tag}.pkl")
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


def _copy_to_cpu(x):
    # 递归把 state_dict 里的 tensor 搬到 CPU（异步保存线程里使用，避免与训练争 GPU）
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)
