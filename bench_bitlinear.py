"""
Minimal BitLinear demo / benchmark.

Run:
    python3 bench_bitlinear.py
"""

import time

import torch

from novamind.training.bitlinear import BitLinear158


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(8, 128, 512, device=device)
    layer = BitLinear158(512, 1024, bias=False).to(device)

    with torch.no_grad():
        t0 = time.time()
        y_train = layer(x)
        t1 = time.time()

        layer.eval()
        layer.compile_circuit(torch.device(device))
        y_add = layer(x, additive_inference=True)
        t2 = time.time()

    delta = (y_train - y_add).abs().mean().item()

    print(f"device={device}")
    print(f"train/STE path: {(t1 - t0):.4f}s")
    print(f"additive path : {(t2 - t1):.4f}s")
    print(f"mean |delta|  : {delta:.6f}")
    print("note: additive path is a reference implementation; real speedups need custom kernels.")


if __name__ == "__main__":
    main()
