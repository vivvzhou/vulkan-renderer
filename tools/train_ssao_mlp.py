#!/usr/bin/env python3
"""Offline training for the neural ambient-occlusion MLP (Phase 9).

The renderer's ssao.comp gathers, for each pixel, a fixed set of screen-space samples of the
G-buffer and builds a scale-invariant feature vector: per sample, the horizon cue
max(dot(dir, N), 0) and the relative sample distance. This script trains a small MLP to map
those sparse features to the ambient-occlusion value a dense (many-sample) estimator would
produce, so the shader gets a denser-looking result from few samples.

Training data is synthetic: random smooth height fields (sums of Gaussians + a tilt) sampled at
the same Fibonacci-disk offsets the shader uses. Everything is scale-invariant, so the network
transfers to the real scene. Weights are written as a flat float32 blob that the shader reads as
an SSBO: W1,b1,W2,b2,W3,b3 with the exact layout below.
"""

import math
import struct
import numpy as np

SEED = 1234
np.random.seed(SEED)

K_SAMPLES = 16          # sparse samples the shader takes
DENSE = 64              # samples used for the training target
IN_DIM = 2 * K_SAMPLES  # [horizon cue, relative distance] per sample
H1, H2 = 32, 16
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def fibonacci_disk(n):
    """The same low-discrepancy disk the shader generates."""
    k = np.arange(n)
    r = np.sqrt((k + 0.5) / n)
    a = k * GOLDEN_ANGLE
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)  # (n, 2)


SPARSE_OFF = fibonacci_disk(K_SAMPLES)  # (K, 2)
DENSE_OFF = fibonacci_disk(DENSE)       # (DENSE, 2)


def make_batch(batch):
    """Generate a batch of (features, ao_target) from random smooth height fields."""
    g = 4  # gaussians per field
    centers = np.random.uniform(-1.2, 1.2, size=(batch, g, 2))
    amps = np.random.uniform(-1.0, 1.0, size=(batch, g)) * np.random.uniform(0.2, 2.5, size=(batch, 1))
    widths = np.random.uniform(0.3, 1.0, size=(batch, g))
    tilt = np.random.uniform(-0.6, 0.6, size=(batch, 2))

    def field_eval(offsets):
        diff = offsets[None, None, :, :] - centers[:, :, None, :]
        d2 = np.sum(diff * diff, axis=3)
        gauss = amps[:, :, None] * np.exp(-d2 / (widths[:, :, None] ** 2))
        h = np.sum(gauss, axis=1)                       # (B, M)
        lin = offsets @ tilt.T                          # (M, B)
        return h + lin.T                                # (B, M)

    h0 = field_eval(np.zeros((1, 2)))[:, 0]             # (B,) center height

    # Sparse features.
    hs = field_eval(SPARSE_OFF)                         # (B, K)
    dz = hs - h0[:, None]                               # (B, K)
    horiz = np.linalg.norm(SPARSE_OFF, axis=1)[None, :] # (1, K)
    dist = np.sqrt(horiz ** 2 + dz ** 2)                # (B, K)
    cue = np.maximum(dz / np.maximum(dist, 1e-6), 0.0)  # (B, K) = max(dir.z, 0), N = +Z
    mean_dist = np.mean(dist, axis=1, keepdims=True)
    rel = dist / np.maximum(mean_dist, 1e-6)            # (B, K)
    feats = np.concatenate([cue, rel], axis=1)          # (B, 2K)

    # Dense reference AO.
    hd = field_eval(DENSE_OFF)                          # (B, DENSE)
    dzd = hd - h0[:, None]
    horizd = np.linalg.norm(DENSE_OFF, axis=1)[None, :]
    distd = np.sqrt(horizd ** 2 + dzd ** 2)
    cued = np.maximum(dzd / np.maximum(distd, 1e-6), 0.0)
    falloff = np.maximum(1.0 - horizd, 0.0)             # nearer samples occlude more
    occ = np.mean(cued * falloff, axis=1)               # (B,)
    ao = np.clip(1.0 - 2.0 * occ, 0.0, 1.0)             # strength = 2.0
    return feats.astype(np.float32), ao.astype(np.float32)[:, None]


# --- Tiny MLP with manual backprop + Adam ---------------------------------------------------
def he(shape):
    return (np.random.randn(*shape) * math.sqrt(2.0 / shape[1])).astype(np.float32)


params = {
    "W1": he((H1, IN_DIM)), "b1": np.zeros((H1,), np.float32),
    "W2": he((H2, H1)),     "b2": np.zeros((H2,), np.float32),
    "W3": he((1, H2)),      "b3": np.zeros((1,), np.float32),
}
adam_m = {k: np.zeros_like(v) for k, v in params.items()}
adam_v = {k: np.zeros_like(v) for k, v in params.items()}


def forward(x):
    z1 = x @ params["W1"].T + params["b1"]; a1 = np.maximum(z1, 0.0)
    z2 = a1 @ params["W2"].T + params["b2"]; a2 = np.maximum(z2, 0.0)
    z3 = a2 @ params["W3"].T + params["b3"]; y = 1.0 / (1.0 + np.exp(-z3))
    return y, (x, z1, a1, z2, a2, z3, y)


def backward(cache, target):
    x, z1, a1, z2, a2, z3, y = cache
    n = x.shape[0]
    dz3 = (y - target) * y * (1.0 - y) / n              # MSE through sigmoid
    grads = {"W3": dz3.T @ a2, "b3": dz3.sum(0)}
    da2 = dz3 @ params["W3"]; dz2 = da2 * (z2 > 0)
    grads["W2"] = dz2.T @ a1; grads["b2"] = dz2.sum(0)
    da1 = dz2 @ params["W2"]; dz1 = da1 * (z1 > 0)
    grads["W1"] = dz1.T @ x; grads["b1"] = dz1.sum(0)
    return grads


def adam_step(grads, t, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
    for k in params:
        adam_m[k] = b1 * adam_m[k] + (1 - b1) * grads[k]
        adam_v[k] = b2 * adam_v[k] + (1 - b2) * grads[k] ** 2
        mhat = adam_m[k] / (1 - b1 ** t)
        vhat = adam_v[k] / (1 - b2 ** t)
        params[k] -= lr * mhat / (np.sqrt(vhat) + eps)


def main():
    steps = 4000
    batch = 512
    for t in range(1, steps + 1):
        x, target = make_batch(batch)
        y, cache = forward(x)
        adam_step(backward(cache, target), t)
        if t % 500 == 0 or t == 1:
            loss = float(np.mean((y - target) ** 2))
            print(f"step {t:5d}  mse {loss:.5f}")

    # Validate against a fresh batch.
    xv, tv = make_batch(4096)
    yv, _ = forward(xv)
    print(f"val mse {float(np.mean((yv - tv) ** 2)):.5f}  "
          f"val mae {float(np.mean(np.abs(yv - tv))):.5f}")

    # Serialize as a flat float32 blob: W1,b1,W2,b2,W3,b3 (row-major, out-major weights).
    order = ["W1", "b1", "W2", "b2", "W3", "b3"]
    flat = np.concatenate([params[k].ravel() for k in order]).astype(np.float32)
    out = "assets/ssao_mlp.bin"
    with open(out, "wb") as f:
        f.write(struct.pack("<I", flat.size))  # float count header
        f.write(flat.tobytes())
    print(f"wrote {out}: {flat.size} floats ({flat.size * 4 + 4} bytes)")


if __name__ == "__main__":
    main()
