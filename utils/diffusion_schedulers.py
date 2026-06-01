"""Schedulers for Denoising Diffusion Probabilistic Models"""

import math

import numpy as np
import torch
    
class CategoricalFM(object):
    def __init__(self):
        pass
    
    def sample(self, x1, t):
        uniform = torch.ones_like(x1)*0.5
        xt = t*x1 + (1-t)*uniform
        return torch.bernoulli(xt.clamp(0,1))

class CategoricalDiffusion(object):
    """Gaussian Diffusion process with linear beta scheduling"""
    
    def __init__(self, T, schedule):
        # Diffusion steps
        self.T = T
        
        # Noise schedule
        if schedule == 'linear':
            b0 = 1e-4
            bT = 2e-2
            self.beta = np.linspace(b0, bT, T)
        elif schedule == 'cosine':
            self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(0)  # Generate an extra alpha for bT
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)
        
        beta = self.beta.reshape((-1, 1, 1))
        eye = np.eye(2).reshape((1, 2, 2))
        ones = np.ones((2, 2)).reshape((1, 2, 2))
        
        self.Qs = (1 - beta) * eye + (beta / 2) * ones
        
        Q_bar = [np.eye(2)]
        for Q in self.Qs:
            Q_bar.append(Q_bar[-1] @ Q)
            self.Q_bar = np.stack(Q_bar, axis=0)
        
    def __cos_noise(self, t):
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2
    
    def sample(self, x0_onehot, t):
        # Select noise scales
        Q_bar = torch.from_numpy(self.Q_bar[t]).float().to(x0_onehot.device)
        xt = torch.matmul(x0_onehot, Q_bar)
        return torch.bernoulli(xt[..., 1].clamp(0, 1))

class FMInferenceSchedule(object):
    def __init__(self, inference_schedule="linear", inference_T=1000):
        self.inference_schedule = inference_schedule
        self.inference_T = inference_T

    def __call__(self, i):
        if self.inference_schedule == "linear":
            t_start = 0 + float(i) / self.inference_T
            t_end = 0 + float(i + 1) / self.inference_T
            t_end = np.clip(t_end, 0, 1)
            return t_start, t_end
        
        elif self.inference_schedule == "cosine":
            t_start = 0 + np.sin(float(i) / self.inference_T * np.pi / 2)
            t_end = 0 + np.sin(float(i + 1) / self.inference_T * np.pi / 2)
            t_end = np.clip(t_end, 0, 1)
            return t_start, t_end

        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler}")

class DiffusionInferenceSchedule(object):
    def __init__(self, inference_schedule="linear", T=1000, inference_T=1000):
        self.inference_schedule = inference_schedule
        self.T = T
        self.inference_T = inference_T
    
    def __call__(self, i):
        assert 0 <= i < self.inference_T
    
        if self.inference_schedule == "linear":
            t1 = self.T - int((float(i) / self.inference_T) * self.T)
            t1 = np.clip(t1, 1, self.T)
            
            t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2
        elif self.inference_schedule == "cosine":
            t1 = self.T - int(
                np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T)
            t1 = np.clip(t1, 1, self.T)
            
            t2 = self.T - int(
                np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2
        else:
            raise ValueError("Unknown inference schedule: {}".format(self.inference_schedule))
