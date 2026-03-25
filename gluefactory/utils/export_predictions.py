"""
Export the predictions of a model for a given dataloader (e.g. ImageFolder).
Use a standalone script with `python3 -m dsfm.scipts.export_predictions dir`
or call from another script.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from .tensor import batch_to_device

@torch.no_grad()
def export_predictions_with_time(
    loader,
    model,
    output_file,
    as_half=False,
    keys="*",
    callback_fn=None,
    optional_keys=[],
):
    import time
    
    assert keys == "*" or isinstance(keys, (tuple, list))
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    hfile = h5py.File(str(output_file), "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    
    # 初始化时间记录变量
    total_inference_time = 0.0
    inference_count = 0
    use_cuda_timing = device == "cuda"
    
    for data_ in tqdm(loader):
        data = batch_to_device(data_, device, non_blocking=True)
        
        # 开始计时
        if use_cuda_timing:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.time()
        
        # 模型推理
        pred = model(data)
        
        # 结束计时
        if use_cuda_timing:
            end_event.record()
            torch.cuda.synchronize()
            inference_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
        else:
            inference_time = time.time() - start_time
        
        total_inference_time += inference_time
        inference_count += 1
        
        if callback_fn is not None:
            pred = {**callback_fn(pred, data), **pred}
        if keys != "*":
            if len(set(keys) - set(pred.keys())) > 0:
                raise ValueError(f"Missing key {set(keys) - set(pred.keys())}")
            #只保留['keypoints0', 'keypoints1', 'keypoint_scores0', 'keypoint_scores1', 'matches0', 'matches1', 'matching_scores0', 'matching_scores1']
            pred = {k: v for k, v in pred.items() if k in keys + optional_keys}
        assert len(pred) > 0

        # renormalization
        for k in pred.keys():
            if k.startswith("keypoints"):
                idx = k.replace("keypoints", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]
            if k.startswith("lines"):
                idx = k.replace("lines", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]
            if k.startswith("orig_lines"):
                idx = k.replace("orig_lines", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]

        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)
        try:
            name = data["name"][0]
            grp = hfile.create_group(name)
            for k, v in pred.items():
                grp.create_dataset(k, data=v)
        except:
            print(f"Error: {data['name']}")
            continue

        del pred
    
    hfile.close()
    
    # 计算并打印平均推理时间统计
    if inference_count > 0:
        avg_inference_time = total_inference_time / inference_count
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        print(f"\n=== 推理时间统计 ===")
        print(f"总推理次数: {inference_count}")
        print(f"总推理时间: {total_inference_time:.4f} 秒")
        print(f"平均推理时间: {avg_inference_time:.4f} 秒")
        print(f"推理速度 (FPS): {fps:.2f}")
        print(f"使用设备: {device}")
        print("=" * 20)
    
    return output_file

@torch.no_grad()
def export_predictions(
    loader,
    model,
    output_file,
    as_half=False,
    keys="*",
    callback_fn=None,
    optional_keys=[],
):
    assert keys == "*" or isinstance(keys, (tuple, list))
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    hfile = h5py.File(str(output_file), "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    for data_ in tqdm(loader):
        data = batch_to_device(data_, device, non_blocking=True)
        pred = model(data)
        if callback_fn is not None:
            pred = {**callback_fn(pred, data), **pred}
        if keys != "*":
            if len(set(keys) - set(pred.keys())) > 0:
                raise ValueError(f"Missing key {set(keys) - set(pred.keys())}")
            #只保留['keypoints0', 'keypoints1', 'keypoint_scores0', 'keypoint_scores1', 'matches0', 'matches1', 'matching_scores0', 'matching_scores1']
            pred = {k: v for k, v in pred.items() if k in keys + optional_keys}
        assert len(pred) > 0

        # renormalization
        for k in pred.keys():
            if k.startswith("keypoints"):
                idx = k.replace("keypoints", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]
            if k.startswith("lines"):
                idx = k.replace("lines", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]
            if k.startswith("orig_lines"):
                idx = k.replace("orig_lines", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]

        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)
        try:
            name = data["name"][0]
            grp = hfile.create_group(name)
            for k, v in pred.items():
                grp.create_dataset(k, data=v)
        except:
            print(f"Error: {data['name']}")
            continue
        # except RuntimeError:
        #     continue

        del pred
    hfile.close()
    return output_file

@torch.no_grad()
def export_predictions_dense(
    loader,
    model,
    output_file,
    as_half=False,
    keys="*",
    callback_fn=None,
    optional_keys=[],
):
    assert keys == "*" or isinstance(keys, (tuple, list))
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    hfile = h5py.File(str(output_file), "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    for data_ in tqdm(loader):
        data = batch_to_device(data_, device, non_blocking=True)
        pred = model(data)
        if callback_fn is not None:
            pred = {**callback_fn(pred, data), **pred}
        if keys != "*":
            if len(set(keys) - set(pred.keys())) > 0:
                raise ValueError(f"Missing key {set(keys) - set(pred.keys())}")
            pred = {k: v for k, v in pred.items() if k in keys + optional_keys}
        assert len(pred) > 0

        # renormalization
        for k in pred.keys():
            if k.startswith("keypoints"):
                idx = k.replace("keypoints", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]
            if k.startswith("lines"):
                idx = k.replace("lines", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]
            if k.startswith("orig_lines"):
                idx = k.replace("orig_lines", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]

        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)
        try:
            name = data["name"][0]
            grp = hfile.create_group(name)
            for k, v in pred.items():
                grp.create_dataset(k, data=v)
        except RuntimeError:
            continue

        del pred
    hfile.close()
    return output_file
