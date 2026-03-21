from setuptools import setup, find_packages

setup(
    name="novamind",
    version="0.1.0",
    description="NovaMind — Next-generation lightweight cognitive architecture",
    author="Felix / 飞悠-Glitch",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "einops>=0.7.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
    ],
    extras_require={
        "fast": [
            "mamba-ssm>=2.0.0",   # CUDA 优化的 SSM 内核
            "causal-conv1d>=1.2.0",
        ],
        "train": [
            "bitsandbytes>=0.43.0",  # 8-bit 优化器
            "accelerate>=0.28.0",
            "wandb",
        ],
    },
)
