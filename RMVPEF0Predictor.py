from typing import Union
import sys, torch, numpy as np, traceback, pdb

import torch
import torch.nn.functional as F

from rmvpe import RMVPE

class RMVPEF0Predictor:
    def __init__(self, hop_length=320, f0_min=50, f0_max=1100, dtype=torch.float32, device=None,
                 sampling_rate=16000,
                 threshold=0.05):
        self.rmvpe = RMVPE(model_path="pretrain/rmvpe.pt", dtype=dtype, device=device)
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self.name = "rmvpe"

    def repeat_expand(
            self, content: Union[torch.Tensor, np.ndarray], target_len: int, mode: str = "nearest"
    ):
        ndim = content.ndim

        if content.ndim == 1:
            content = content[None, None]
        elif content.ndim == 2:
            content = content[None]

        assert content.ndim == 3

        is_np = isinstance(content, np.ndarray)
        if is_np:
            content = torch.from_numpy(content)

        results = torch.nn.functional.interpolate(content, size=target_len, mode=mode)

        if is_np:
            results = results.numpy()

        if ndim == 1:
            return results[0, 0]
        elif ndim == 2:
            return results[0]
    #   默认
    def post_process0(self, x, sampling_rate, f0, pad_to):
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float().to(x.device)

        if pad_to is None:
            return f0

        f0 = self.repeat_expand(f0, pad_to)

        vuv_vector = torch.zeros_like(f0)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0

        # 去掉0频率, 并线性插值
        nzindex = torch.nonzero(f0).squeeze()
        f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()
        time_org = self.hop_length / sampling_rate * nzindex.cpu().numpy()
        time_frame = np.arange(pad_to) * self.hop_length / sampling_rate

        vuv_vector = F.interpolate(vuv_vector[None, None, :], size=pad_to)[0][0]

        if f0.shape[0] <= 0:
            return torch.zeros(pad_to, dtype=torch.float, device=x.device).cpu().numpy(), vuv_vector.cpu().numpy()
        if f0.shape[0] == 1:
            return (torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[
                0]).cpu().numpy(), vuv_vector.cpu().numpy()

        # 大概可以用 torch 重写?
        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
        #vuv_vector = np.ceil(scipy.ndimage.zoom(vuv_vector,pad_to/len(vuv_vector),order = 0))

        return f0, vuv_vector.cpu().numpy()

    #   分段线性插值
    def post_process1(self, x, sampling_rate, f0, pad_to):
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float().to(x.device)

        if pad_to is None:
            return f0

        f0 = self.repeat_expand(f0, pad_to)

        vuv_vector = torch.zeros_like(f0)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0

        nzindex = torch.nonzero(f0).squeeze()
        segments = []
        for i in range(len(nzindex) - 1):
            if nzindex[i + 1] - nzindex[i] > 1:
                segments.append((nzindex[i], nzindex[i + 1]))

        for start, end in segments:
            slope = (f0[end] - f0[start]) / (end - start)
            intercept = f0[start] - slope * start
            f0[start + 1: end] = slope * torch.arange(start + 1, end, device=f0.device) + intercept

        return f0.cpu().numpy(), vuv_vector.cpu().numpy()

    # 使用三次样条插值
    def post_process2(self, x, sampling_rate, f0, pad_to):
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float().to(x.device)

        if pad_to is None:
            return f0

        f0 = self.repeat_expand(f0, pad_to)

        vuv_vector = torch.zeros_like(f0)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0

        nzindex = torch.nonzero(f0).squeeze()
        f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()
        time_org = self.hop_length / sampling_rate * nzindex.cpu().numpy()
        time_frame = np.arange(pad_to) * self.hop_length / sampling_rate

        vuv_vector = F.interpolate(vuv_vector[None, None, :], size=pad_to)[0][0]

        if f0.shape[0] <= 0:
            return torch.zeros(pad_to, dtype=torch.float, device=x.device).cpu().numpy(), vuv_vector.cpu().numpy()
        if f0.shape[0] == 1:
            return (torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[
                0]).cpu().numpy(), vuv_vector.cpu().numpy()

        from scipy.interpolate import CubicSpline
        cs = CubicSpline(time_org, f0)
        f0 = cs(time_frame)

        return f0, vuv_vector.cpu().numpy()

    def compute_f0(self, wav, p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        if p_len is None:
            p_len = x.shape[0] // self.hop_length
        else:
            assert abs(p_len - x.shape[0] // self.hop_length) < 4, "pad length error"
        f0 = self.rmvpe.infer_from_audio(x, self.sampling_rate, self.threshold)
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn, rtn
        return self.post_process0(x, self.sampling_rate, f0, p_len)[0]

    def compute_f0_uv(self, wav, p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        if p_len is None:
            p_len = x.shape[0] // self.hop_length
        else:
            assert abs(p_len - x.shape[0] // self.hop_length) < 4, "pad length error"
        f0 = self.rmvpe.infer_from_audio(x, self.sampling_rate, self.threshold)
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn, rtn
        return self.post_process0(x, self.sampling_rate, f0, p_len)
