from __future__ import annotations

from typing import Dict, List, Optional


CURVES = {
    "taic": {
        "detection": {
            "metric": "mAP-bbox(%)",
            "bpp": [0.089, 0.109, 0.167, 0.271],
            "score": [45.31, 46.12, 46.83, 47.34],
        },
        "instance": {
            "metric": "mAP-mask(%)",
            "bpp": [0.091, 0.152, 0.227, 0.341],
            "score": [41.2, 42.6, 43.7, 44.3],
        },
        "semantic": {
            "metric": "mIoU(%)",
            "bpp": [0.089, 0.133, 0.196, 0.327],
            "score": [71.1, 71.8, 72.3, 72.8],
        },
        "panoptic": {
            "metric": "Panoptic Quality(%)",
            "bpp": [0.133, 0.196, 0.27, 0.39],
            "score": [55.7, 56.3, 56.7, 57.1],
        },
        "pose": {
            "metric": "mAP-OKS(%)",
            "bpp": [0.13, 0.21, 0.30, 0.43],
            "score": [58.2, 62.7, 65.4, 67.7],
        },
    },
    "ctaic": {
        "instance": {
            "metric": "mAP-mask(%)",
            "bpp": [0.089, 0.141, 0.217, 0.341],
            "score": [42.4, 43.4, 44.2, 44.6],
        },
        "panoptic": {
            "metric": "Panoptic Quality(%)",
            "bpp": [0.14, 0.20, 0.28, 0.41],
            "score": [56.4, 56.8, 57.1, 57.4],
        },
        "pose": {
            "metric": "mAP-OKS(%)",
            "bpp": [0.12, 0.20, 0.28, 0.41],
            "score": [60.7, 64.8, 66.6, 68.2],
        },
    },
}


def _interp1d(xs: List[float], ys: List[float], x: float) -> float:
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            ratio = (x - x0) / (x1 - x0)
            return float(y0 + ratio * (y1 - y0))
    return float(ys[-1])


def simulate_task_metric(task: str, bpp: float, family: str = "taic") -> Optional[Dict[str, float]]:
    task = str(task).lower()
    family = str(family).lower()
    spec = CURVES.get(family, {}).get(task)
    if spec is None:
        return None
    score = _interp1d(spec["bpp"], spec["score"], float(bpp))
    return {
        "metric": spec["metric"],
        "score": score,
        "bpp_input": float(bpp),
        "simulated": True,
    }
