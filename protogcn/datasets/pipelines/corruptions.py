import random
import numpy as np

from ..builder import PIPELINES


def _severity_value(severity, values):
    severity = int(severity)
    severity = max(1, min(5, severity))
    return values[severity - 1]


@PIPELINES.register_module()
class JointGaussianNoise:
    def __init__(self, severity=1):
        self.severity = severity
        self.sigmas = [0.005, 0.01, 0.02, 0.04, 0.08]

    def __call__(self, results):
        kp = results['keypoint'].astype(np.float32)
        sigma = _severity_value(self.severity, self.sigmas)
        noise = np.random.normal(0, sigma, size=kp.shape).astype(np.float32)
        valid = (np.abs(kp).sum(axis=-1, keepdims=True) > 0).astype(np.float32)
        results['keypoint'] = kp + noise * valid
        return results


@PIPELINES.register_module()
class JointMasking:
    def __init__(self, severity=1):
        self.severity = severity
        self.ratios = [0.05, 0.1, 0.2, 0.3, 0.4]

    def __call__(self, results):
        kp = results['keypoint'].astype(np.float32)
        M, T, V, C = kp.shape
        ratio = _severity_value(self.severity, self.ratios)
        mask = (np.random.rand(M, T, V, 1) > ratio).astype(np.float32)
        results['keypoint'] = kp * mask
        return results


@PIPELINES.register_module()
class FrameDrop:
    def __init__(self, severity=1):
        self.severity = severity
        self.ratios = [0.05, 0.1, 0.2, 0.3, 0.4]

    def __call__(self, results):
        kp = results['keypoint'].astype(np.float32)
        M, T, V, C = kp.shape
        ratio = _severity_value(self.severity, self.ratios)
        keep = np.random.rand(T) > ratio
        keep[0] = True
        new_kp = kp.copy()
        last = 0
        for t in range(T):
            if keep[t]:
                last = t
            else:
                new_kp[:, t] = kp[:, last]
        results['keypoint'] = new_kp
        return results


@PIPELINES.register_module()
class TemporalJitter:
    def __init__(self, severity=1):
        self.severity = severity
        self.window = [1, 2, 3, 4, 5]

    def __call__(self, results):
        kp = results['keypoint'].astype(np.float32)
        M, T, V, C = kp.shape
        w = _severity_value(self.severity, self.window)
        idx = np.arange(T)
        jitter = np.random.randint(-w, w + 1, size=T)
        idx = np.clip(idx + jitter, 0, T - 1)
        results['keypoint'] = kp[:, idx]
        return results


@PIPELINES.register_module()
class ViewPerturbation:
    def __init__(self, severity=1):
        self.severity = severity
        self.thetas = [0.05, 0.1, 0.2, 0.35, 0.5]
        self.scales = [0.02, 0.04, 0.06, 0.08, 0.1]

    def _rot3d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        rx = np.array([[1, 0, 0], [0, cos[0], sin[0]], [0, -sin[0], cos[0]]], dtype=np.float32)
        ry = np.array([[cos[1], 0, -sin[1]], [0, 1, 0], [sin[1], 0, cos[1]]], dtype=np.float32)
        rz = np.array([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]], dtype=np.float32)
        return rz @ ry @ rx

    def __call__(self, results):
        kp = results['keypoint'].astype(np.float32)
        C = kp.shape[-1]
        theta = _severity_value(self.severity, self.thetas)
        scale_jitter = _severity_value(self.severity, self.scales)
        if C == 3:
            angles = np.random.uniform(-theta, theta, size=3).astype(np.float32)
            rot = self._rot3d(angles)
            kp = np.einsum('ab,mtvb->mtva', rot, kp)
        scale = 1.0 + np.random.uniform(-scale_jitter, scale_jitter)
        results['keypoint'] = kp * scale
        return results
