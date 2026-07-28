"""Reproduce the FP32 weight sizes reported for the STM32 deployment table.

FP32 weights [KiB] = n_params * 4 / 1024, with n_params taken from manifest.json.

    python verify_fp32_weights.py
"""
import json

for v in json.load(open("manifest.json", encoding="utf-8"))["variants"]:
    print(f"{v['name']:24} {v['n_params']:>10,} params  ->  {v['n_params'] * 4 / 1024:8.1f} KiB")
