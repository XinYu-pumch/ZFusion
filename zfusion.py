#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-in-One Workflow: LoRA Fix -> Merge -> Quantize to GGUF

整合工作流：
1. (可选) 修复 ai-toolkit 生成的 LoRA (Zimage/Lumina2 格式转换)。
2. 将 LoRA 与基础模型融合。
3. 将融合后的模型转换为 GGUF 格式并进行量化。

作者: 根据用户提供的三个脚本整合而成
日期: 2024-05-22
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import traceback
import logging
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
import tempfile
from tqdm import tqdm

# --- 依赖检查 ---
try:
    import torch
    from safetensors.torch import load_file, save_file
    from safetensors import safe_open
    import gguf
    from gguf import GGUFWriter, GGMLQuantizationType, LlamaFileType
except ImportError as e:
    # 在GUI启动前进行检查，如果失败则无法启动
    print(f"错误：缺少必要的库。请先安装依赖：\n{e}")
    print("\n请在终端中运行以下命令安装所有必需的库:")
    print("pip install torch safetensors gguf numpy tqdm")
    sys.exit(1)

# --- 全局日志记录器 ---
class GuiLogger:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.text_widget.config(state='disabled')

    def log(self, msg, level="INFO"):
        self.text_widget.config(state='normal')
        self.text_widget.insert(tk.END, f"[{level}] {msg}\n")
        self.text_widget.see(tk.END)
        self.text_widget.config(state='disabled')
        self.text_widget.update_idletasks()

# ==============================================================================
# SCRIPT 1: Zimage从ai-toolkit转换补全层级.py (核心逻辑)
# ==============================================================================
def convert_lora_for_comfyui(input_path, output_path, logger=None, progress_callback=None):
    """
    将 ai-toolkit 的 LoRA 转换为 ComfyUI 兼容格式。
    核心功能是从 Zimage...py 脚本中提取的。
    """
    if logger: logger.log("开始 LoRA 格式转换...")
    if progress_callback: progress_callback(0, "正在加载 LoRA 文件...")

    try:
        lora_dict = load_file(input_path)
    except Exception as e:
        raise ValueError(f"加载 LoRA 文件失败: {e}")

    total_keys = len(lora_dict)
    if logger: logger.log(f"LoRA 加载完成，共 {total_keys} 个键。")
    if progress_callback: progress_callback(5, f"分析键值...")

    # 检查是否需要转换
    needs_conversion = any('.attention.to_q.' in k for k in lora_dict.keys())
    if not needs_conversion:
        if logger: logger.log("LoRA 似乎已经是标准格式，无需转换。将直接复制文件。")
        if progress_callback: progress_callback(100, "无需转换，跳过此步骤。")
        save_file(lora_dict, output_path) # 简单复制
        return 0 # 返回0表示没有层被转换

    layer_groups = defaultdict(lambda: defaultdict(dict))
    output_dict = {}
    
    processed = 0
    for key, value in lora_dict.items():
        processed += 1
        if progress_callback and processed % 50 == 0:
            progress_callback(5 + 45 * processed // total_keys, f"解析键... ({processed}/{total_keys})")

        if '.attention.to_out.0.' in key:
            new_key = key.replace('.to_out.0.', '.out.')
            output_dict[new_key] = value
            if 'lora_A' in key:
                base = key.rsplit('.lora_A', 1)[0]
                alpha_key = f"{base}.alpha"
                if alpha_key in lora_dict:
                    new_alpha = alpha_key.replace('.to_out.0.', '.out.')
                    output_dict[new_alpha] = lora_dict[alpha_key]
            continue

        if '.attention.to_' in key and '.alpha' in key:
            continue

        if '.attention.to_' in key and any(x in key for x in ('.to_q.', '.to_k.', '.to_v.')):
            parts = key.split('.')
            layer_idx, attn_type, lora_type = None, None, None
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts): layer_idx = parts[i + 1]
                elif p in ('to_q', 'to_k', 'to_v'): attn_type = p[3:]
                elif p in ('lora_A', 'lora_B'): lora_type = p
            
            if layer_idx and attn_type and lora_type:
                base_parts = [p for p in parts if p not in ('to_q', 'to_k', 'to_v')]
                base_key = '.'.join(base_parts[:-2])
                layer_groups[base_key][attn_type][lora_type] = value
                continue

        output_dict[key] = value

    if logger: logger.log(f"找到 {len(layer_groups)} 个需要合并的注意力层。")
    if progress_callback: progress_callback(50, "开始合并 qkv 层...")

    converted_count = 0
    step = 40.0 / max(len(layer_groups), 1)
    current = 0
    for base_key, qkv_dict in layer_groups.items():
        current += 1
        if progress_callback:
            progress_callback(50 + step * current, f"合并层 {current}/{len(layer_groups)}")

        if not all(x in qkv_dict for x in ('q', 'k', 'v')): continue
        qB, kB, vB = qkv_dict['q'].get('lora_B'), qkv_dict['k'].get('lora_B'), qkv_dict['v'].get('lora_B')
        qA, kA, vA = qkv_dict['q'].get('lora_A'), qkv_dict['k'].get('lora_A'), qkv_dict['v'].get('lora_A')

        if None in (qB, kB, vB, qA, kA, vA): continue

        try:
            hidden_dim, rank = qB.shape
            qkv_B = torch.zeros(3 * hidden_dim, 3 * rank, dtype=qB.dtype)
            qkv_B[:hidden_dim, :rank] = qB
            qkv_B[hidden_dim:2*hidden_dim, rank:2*rank] = kB
            qkv_B[2*hidden_dim:, 2*rank:] = vB
            qkv_A = torch.cat([qA, kA, vA], dim=0)

            output_dict[f"{base_key}.qkv.lora_B.weight"] = qkv_B
            output_dict[f"{base_key}.qkv.lora_A.weight"] = qkv_A
            converted_count += 1

            alpha_key_q = f"{base_key}.to_q.alpha"
            orig_alpha = lora_dict.get(alpha_key_q) or lora_dict.get(f"{base_key}.to_q.lora_A.alpha")
            if orig_alpha is not None:
                output_dict[f"{base_key}.qkv.alpha"] = orig_alpha * 3.0
        except Exception as e:
            if logger: logger.log(f"合并层 {base_key} 失败: {e}", "ERROR")

    if progress_callback: progress_callback(95, "正在保存修复后的 LoRA...")
    
    metadata = {}
    try:
        with safe_open(input_path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
    except Exception: pass
    metadata['converted_by'] = 'All-in-One Workflow GUI'
    
    save_file(output_dict, output_path, metadata=metadata)
    if logger: logger.log(f"LoRA 修复完成，共转换 {converted_count} 个注意力层。")
    if progress_callback: progress_callback(100, "LoRA 修复完成。")
    return converted_count

# ==============================================================================
# SCRIPT 2: nextdit_lora_merger_AB.py (核心逻辑)
# ==============================================================================
def merge_lora_A_B_with_base_model(base_model_path, lora_path, output_path, lora_strength=1.0, logger=None, progress_callback=None):
    """
    融合 lora_A/B 格式的 LoRA 到基础模型。
    核心功能是从 nextdit_lora_merger_AB.py 脚本中提取的。
    """
    if logger:
        logger.log("=== 开始模型与 LoRA 融合 ===")
        logger.log(f"基础模型: {Path(base_model_path).name}")
        logger.log(f"LoRA: {Path(lora_path).name}")
        logger.log(f"权重: {lora_strength}")

    # 加载基础模型
    if progress_callback: progress_callback(0, "正在加载基础模型...")
    try:
        base_model = load_file(base_model_path)
        if logger: logger.log(f"基础模型加载成功: {len(base_model.keys())} 个张量。")
    except Exception as e:
        raise RuntimeError(f"加载基础模型失败: {e}")

    # 加载LoRA
    if progress_callback: progress_callback(20, "正在加载 LoRA...")
    try:
        lora_model = load_file(lora_path)
        if logger: logger.log(f"LoRA 加载成功: {len(lora_model.keys())} 个张量。")
    except Exception as e:
        raise RuntimeError(f"加载 LoRA 失败: {e}")

    # 提取 lora_A/B 对
    if progress_callback: progress_callback(40, "正在提取 LoRA A/B 对...")
    lora_pairs = {}
    prefix_to_remove = 'diffusion_model.' if any(k.startswith('diffusion_model.') for k in lora_model.keys()) else None
    
    for key in lora_model.keys():
        if '.lora_A.weight' in key:
            clean_key = key
            if prefix_to_remove and clean_key.startswith(prefix_to_remove):
                clean_key = clean_key[len(prefix_to_remove):]
            
            base_key = clean_key.replace('.lora_A.weight', '.weight')
            lora_b_key = key.replace('.lora_A.weight', '.lora_B.weight')
            
            if lora_b_key in lora_model:
                lora_pairs[base_key] = {'A': lora_model[key], 'B': lora_model[lora_b_key]}
    
    if logger: logger.log(f"找到 {len(lora_pairs)} 个 LoRA A/B 对。")
    if not lora_pairs:
        raise ValueError("在 LoRA 文件中未找到 'lora_A/lora_B' 格式的权重。请检查 LoRA 文件。")

    # 应用 LoRA
    if progress_callback: progress_callback(60, "正在应用 LoRA 权重...")
    merged_model = base_model.copy()
    applied_count, skipped_count = 0, 0
    
    total_pairs = len(lora_pairs)
    processed_pairs = 0
    for base_key, pair_info in lora_pairs.items():
        processed_pairs += 1
        if progress_callback:
            progress_callback(60 + 35 * processed_pairs // total_pairs, f"应用层 {processed_pairs}/{total_pairs}")

        if base_key in merged_model:
            base_weight = merged_model[base_key]
            lora_A, lora_B = pair_info['A'], pair_info['B']
            
            try:
                if len(lora_A.shape) == 2 and len(lora_B.shape) == 2:
                    lora_delta = torch.mm(lora_B, lora_A) * lora_strength
                elif len(lora_A.shape) == 4 and len(lora_B.shape) == 4:
                    lora_delta = torch.nn.functional.conv2d(lora_A.permute(1, 0, 2, 3), lora_B).permute(1, 0, 2, 3) * lora_strength
                else:
                    if logger: logger.log(f"跳过不支持的 LoRA 形状: A: {lora_A.shape}, B: {lora_B.shape}", "WARN")
                    continue
                
                if base_weight.shape != lora_delta.shape:
                    if logger: logger.log(f"跳过形状不匹配的层: base: {base_weight.shape}, delta: {lora_delta.shape}", "WARN")
                    continue

                merged_model[base_key] = base_weight + lora_delta.to(base_weight.dtype)
                applied_count += 1
            except Exception as e:
                if logger: logger.log(f"应用 LoRA 到 {base_key} 时出错: {e}", "ERROR")
                skipped_count += 1
        else:
            skipped_count += 1

    if logger:
        logger.log(f"成功应用 LoRA 到 {applied_count} 个层。")
        if skipped_count > 0:
            logger.log(f"跳过 {skipped_count} 个层 (在基础模型中未找到或出错)。", "WARN")
    
    if applied_count == 0:
        raise RuntimeError("错误: 没有任何 LoRA 层被应用！这很可能是因为基础模型和 LoRA 之间的键名不匹配。")

    # 保存融合后的模型
    if progress_callback: progress_callback(95, "正在保存融合后的模型...")
    try:
        save_file(merged_model, output_path)
        if logger: logger.log(f"融合后的模型已保存。")
    except Exception as e:
        raise RuntimeError(f"保存融合模型失败: {e}")
    
    if progress_callback: progress_callback(100, "模型融合完成。")
    return merged_model

# ==============================================================================
# SCRIPT 3: convert_quantize.py (核心逻辑)
# ==============================================================================
# --- 从 convert_quantize.py 脚本中提取的辅助类和函数 ---
QUANTIZATION_THRESHOLD = 1024
REARRANGE_THRESHOLD = 512
MAX_TENSOR_NAME_LENGTH = 127
MAX_TENSOR_DIMS = 4

class ModelTemplate:
    arch, shape_fix = "invalid", False
    keys_detect, keys_banned, keys_hiprec, keys_ignore = [], [], [], []
    def handle_nd_tensor(self, key, data): raise NotImplementedError(f"Tensor >4D: {key} @ {data.shape}")

class ModelFlux(ModelTemplate):
    arch = "flux"; keys_detect = [("transformer_blocks.0.attn.norm_added_k.weight",), ("double_blocks.0.img_attn.proj.weight",)]; keys_banned = ["transformer_blocks.0.attn.norm_added_k.weight",]
class ModelSD3(ModelTemplate):
    arch = "sd3"; keys_detect = [("transformer_blocks.0.attn.add_q_proj.weight",), ("joint_blocks.0.x_block.attn.qkv.weight",)]; keys_banned = ["transformer_blocks.0.attn.add_q_proj.weight",]
class ModelAura(ModelTemplate):
    arch = "aura"; keys_detect = [("double_layers.3.modX.1.weight",), ("joint_transformer_blocks.3.ff_context.out_projection.weight",)]; keys_banned = ["joint_transformer_blocks.3.ff_context.out_projection.weight",]
class ModelHiDream(ModelTemplate):
    arch = "hidream"; keys_detect = [("caption_projection.0.linear.weight", "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight")]; keys_hiprec = [".ff_i.gate.weight", "img_emb.emb_pos"]
class CosmosPredict2(ModelTemplate):
    arch = "cosmos"; keys_detect = [("blocks.0.mlp.layer1.weight", "blocks.0.adaln_modulation_cross_attn.1.weight")]; keys_hiprec = ["pos_embedder"]; keys_ignore = ["_extra_state", "accum_"]
class ModelHyVid(ModelTemplate):
    arch = "hyvid"; keys_detect = [("double_blocks.0.img_attn_proj.weight", "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight")]
    def handle_nd_tensor(self, key, data): raise RuntimeError(f"5D tensor fix file already exists!")
class ModelWan(ModelHyVid):
    arch = "wan"; keys_detect = [("blocks.0.self_attn.norm_q.weight", "text_embedding.2.weight", "head.modulation")]; keys_hiprec = [".modulation"]
class ModelLTXV(ModelTemplate):
    arch = "ltxv"; keys_detect = [("adaln_single.emb.timestep_embedder.linear_2.weight", "transformer_blocks.27.scale_shift_table", "caption_projection.linear_2.weight")]; keys_hiprec = ["scale_shift_table"]
class ModelSDXL(ModelTemplate):
    arch = "sdxl"; shape_fix = True; keys_detect = [("down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight",), ("input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "output_blocks.2.2.conv.weight", "output_blocks.5.2.conv.weight"), ("label_emb.0.0.weight",)]
class ModelSD1(ModelTemplate):
    arch = "sd1"; shape_fix = True; keys_detect = [("down_blocks.0.downsamplers.0.conv.weight",), ("input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "input_blocks.9.0.op.weight", "output_blocks.2.1.conv.weight", "output_blocks.5.2.conv.weight", "output_blocks.8.2.conv.weight")]
class ModelLumina2(ModelTemplate):
    arch = "lumina2"; keys_detect = [("cap_embedder.1.weight", "context_refiner.0.attention.qkv.weight")]; keys_hiprec = ["pad_token"]

arch_list = [ModelFlux, ModelSD3, ModelAura, ModelHiDream, CosmosPredict2, ModelLTXV, ModelHyVid, ModelWan, ModelSDXL, ModelSD1, ModelLumina2]

def detect_arch(state_dict):
    for arch_cls in arch_list:
        matched, invalid = False, False
        for match_list in arch_cls.keys_detect:
            if all(key in state_dict for key in match_list):
                matched = True
                invalid = any(key in state_dict for key in arch_cls.keys_banned)
                break
        if matched and not invalid: return arch_cls()
    raise RuntimeError("无法识别的模型架构！")

def get_quant_type(name): return getattr(GGMLQuantizationType, name, None)
def get_file_type(name): return getattr(LlamaFileType, name, None)

def test_quantization(qtype):
    try:
        test_data = np.random.randn(256, 256).astype(np.float32)
        gguf.quants.quantize(test_data, qtype)
        return True
    except (NotImplementedError, Exception):
        return False

QUANT_MAP_UNTESTED = {
    key: (get_quant_type(qtype_name), get_file_type(ftype_name) if ftype_name else None, desc)
    for key, qtype_name, ftype_name, desc in [
        ('f32', 'F32', None, "32-bit float (无损)"), ('f16', 'F16', 'MOSTLY_F16', "16-bit float"), ('bf16', 'BF16', 'MOSTLY_BF16', "bfloat16"),
        ('q2_k', 'Q2_K', 'MOSTLY_Q2_K', "2-bit K-quant"), ('q3_k', 'Q3_K', 'MOSTLY_Q3_K_M', "3-bit K-quant"),
        ('q4_0', 'Q4_0', 'MOSTLY_Q4_0', "4-bit (legacy)"), ('q4_1', 'Q4_1', 'MOSTLY_Q4_1', "4-bit (legacy, 更好)"),
        ('q4_k_m', 'Q4_K_M', 'MOSTLY_Q4_K_M', "4-bit K-quant (推荐)"), ('q5_0', 'Q5_0', 'MOSTLY_Q5_0', "5-bit (legacy)"),
        ('q5_1', 'Q5_1', 'MOSTLY_Q5_1', "5-bit (legacy, 更好)"), ('q5_k_m', 'Q5_K_M', 'MOSTLY_Q5_K_M', "5-bit K-quant"),
        ('q6_k', 'Q6_K', 'MOSTLY_Q6_K', "6-bit K-quant"), ('q8_0', 'Q8_0', 'MOSTLY_Q8_0', "8-bit (量化中最佳)")
    ] if get_quant_type(qtype_name) is not None
}
QUANT_MAP_TESTED, QUANT_BROKEN = None, None

def get_tested_quant_map(logger=None):
    global QUANT_MAP_TESTED, QUANT_BROKEN
    if QUANT_MAP_TESTED is None:
        if logger: logger.log("正在测试当前环境支持的量化类型...")
        working, broken = {}, []
        for key, (qtype, ftype, desc) in QUANT_MAP_UNTESTED.items():
            if test_quantization(qtype): working[key] = (qtype, ftype, desc)
            else: broken.append(key)
        QUANT_MAP_TESTED, QUANT_BROKEN = working, broken
        if logger:
            logger.log(f"测试完成。可用类型: {len(working)}个, 不可用: {len(broken)}个。")
            if broken: logger.log(f"不可用类型: {', '.join(broken)}", "WARN")
    return QUANT_MAP_TESTED

def convert_to_gguf(state_dict, dst_path, quant_type='bf16', logger=None, progress_callback=None):
    """
    将 state_dict 转换为 GGUF 并量化。
    核心功能是从 convert_quantize.py 脚本中提取的。
    """
    if logger:
        logger.log("=== 开始 GGUF 转换与量化 ===")
        logger.log(f"量化类型: {quant_type.upper()}")

    # 识别架构
    if progress_callback: progress_callback(0, "识别模型架构...")
    try:
        model_arch = detect_arch(state_dict)
        if logger: logger.log(f"识别到模型架构: {model_arch.arch}")
    except Exception as e:
        raise RuntimeError(f"识别模型架构失败: {e}")

    # 获取量化配置
    working_quants = get_tested_quant_map(logger)
    quant_key = quant_type.lower()
    if quant_key not in working_quants:
        raise ValueError(f"不支持的量化类型: {quant_type}。请先测试可用类型。")
    
    target_qtype, ftype_gguf, desc = working_quants[quant_key]
    if logger: logger.log(f"使用类型: {desc}")

    fallback_qtype = working_quants.get('f16', working_quants.get('bf16', working_quants.get('f32')))[0]
    if logger: logger.log(f"备用量化类型: {fallback_qtype.name}")

    # 创建 GGUF writer
    if progress_callback: progress_callback(10, "创建 GGUF 文件头...")
    writer = GGUFWriter(path=None, arch=model_arch.arch)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    if ftype_gguf is not None: writer.add_file_type(ftype_gguf)

    # 处理张量
    if progress_callback: progress_callback(20, "开始处理张量...")
    stats = {'quantized': 0, 'kept_f32': 0, 'fallback': 0, 'total': 0}
    total_tensors = len(state_dict)
    processed_tensors = 0

    for key, data in state_dict.items():
        processed_tensors += 1
        if progress_callback:
            progress_callback(20 + 70 * processed_tensors // total_tensors, f"处理张量 {processed_tensors}/{total_tensors}")

        stats['total'] += 1
        if any(x in key for x in model_arch.keys_ignore): continue

        if data.dtype == torch.bfloat16: data = data.to(torch.float32).numpy()
        else: data = data.numpy()

        n_dims = len(data.shape)
        if n_dims > MAX_TENSOR_DIMS:
            if logger: logger.log(f"跳过 >4D 张量: {key} {data.shape}", "WARN")
            continue

        n_params = np.prod(data.shape)
        data_qtype = target_qtype
        if n_dims == 1 or n_params <= QUANTIZATION_THRESHOLD or any(x in key for x in model_arch.keys_hiprec):
            data_qtype = GGMLQuantizationType.F32
            stats['kept_f32'] += 1
        else:
            stats['quantized'] += 1

        try:
            quantized_data = gguf.quants.quantize(data, data_qtype)
        except Exception:
            if logger: logger.log(f"量化 {key} 失败, 尝试备用类型 {fallback_qtype.name}", "WARN")
            data_qtype = fallback_qtype
            quantized_data = gguf.quants.quantize(data, data_qtype)
            stats['fallback'] += 1
        
        writer.add_tensor(key, quantized_data, raw_dtype=data_qtype)

    # 写入文件
    if progress_callback: progress_callback(95, "正在写入 GGUF 文件...")
    writer.write_header_to_file(path=dst_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    if logger:
        logger.log("GGUF 文件写入完成。")
        logger.log(f"统计: 总张量={stats['total']}, 量化={stats['quantized']}, 保持F32={stats['kept_f32']}, 使用备用={stats['fallback']}")
    if progress_callback: progress_callback(100, "GGUF 转换完成。")

# ==============================================================================
# GUI 主应用
# ==============================================================================
class WorkflowGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("模型融合与GGUF量化工作流")
        self.geometry("800x650")
        self.resizable(True, True)

        style = ttk.Style(self)
        style.theme_use('clam')

        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(1, weight=1)

        # --- 输入/输出设置 ---
        io_frame = ttk.LabelFrame(main_frame, text="文件路径", padding=10)
        io_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=5)
        io_frame.columnconfigure(1, weight=1)

        self.base_model_path = tk.StringVar()
        self.lora_path = tk.StringVar()
        self.output_path = tk.StringVar()

        ttk.Label(io_frame, text="基础模型:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(io_frame, textvariable=self.base_model_path).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(io_frame, text="浏览...", command=lambda: self.browse_file(self.base_model_path, "选择基础模型", [("Safetensors", "*.safetensors")])).grid(row=0, column=2, padx=5)

        ttk.Label(io_frame, text="LoRA模型:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(io_frame, textvariable=self.lora_path).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(io_frame, text="浏览...", command=lambda: self.browse_file(self.lora_path, "选择LoRA模型", [("Safetensors", "*.safetensors")])).grid(row=1, column=2, padx=5)

        ttk.Label(io_frame, text="输出文件:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(io_frame, textvariable=self.output_path).grid(row=2, column=1, sticky="ew", padx=5)
        ttk.Button(io_frame, text="另存为...", command=self.browse_save_file).grid(row=2, column=2, padx=5)

        # --- 参数设置 ---
        params_frame = ttk.LabelFrame(main_frame, text="参数设置", padding=10)
        params_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=10)
        params_frame.columnconfigure(1, weight=1)

        self.lora_weight = tk.DoubleVar(value=1.0)
        ttk.Label(params_frame, text="LoRA 权重:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Spinbox(params_frame, from_=0.0, to=5.0, increment=0.1, textvariable=self.lora_weight, width=10).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(params_frame, text="量化类型:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.quant_type = tk.StringVar()
        self.quant_combo = ttk.Combobox(params_frame, textvariable=self.quant_type, state="readonly")
        self.quant_combo.grid(row=1, column=1, sticky="ew", padx=5)
        self.test_quant_btn = ttk.Button(params_frame, text="测试并列出可用量化类型", command=self.test_and_list_quants)
        self.test_quant_btn.grid(row=1, column=2, padx=5)
        self.quant_combo['values'] = ['bf16', 'f16', 'f32']
        self.quant_type.set('q4_k_m') # 默认推荐值

        # --- 控制与进度 ---
        self.progress_label = ttk.Label(main_frame, text="就绪")
        self.progress_label.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)

        self.start_btn = ttk.Button(main_frame, text="开始工作流", command=self.start_workflow)
        self.start_btn.grid(row=4, column=0, columnspan=3, pady=10)

        # --- 日志区域 ---
        log_frame = ttk.LabelFrame(main_frame, text="实时日志", padding=10)
        log_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=10)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)

        self.log_text = tk.Text(log_frame, height=10, state='disabled', font=("Consolas", 9), wrap="word")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.logger = GuiLogger(self.log_text)

    def browse_file(self, var, title, filetypes):
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if path:
            var.set(path)
            if var == self.base_model_path:
                self.auto_fill_output()

    def browse_save_file(self):
        path = filedialog.asksaveasfilename(title="保存 GGUF 文件", defaultextension=".gguf", filetypes=[("GGUF", "*.gguf")])
        if path:
            self.output_path.set(path)

    def auto_fill_output(self):
        base_path = self.base_model_path.get()
        if base_path:
            dir_name = os.path.dirname(base_path)
            base_name = os.path.splitext(os.path.basename(base_path))[0]
            quant = self.quant_type.get().replace('_', '')
            suggested_name = f"{base_name}_merged_{quant}.gguf"
            self.output_path.set(os.path.join(dir_name, suggested_name))

    def update_progress(self, step_name, value, status):
        # 总进度条分为3个阶段：LoRA修复(10%) -> 模型融合(40%) -> GGUF转换(50%)
        total_progress = 0
        if step_name == "LORA_FIX":
            total_progress = value * 0.1
        elif step_name == "MERGE":
            total_progress = 10 + (value * 0.4)
        elif step_name == "QUANTIZE":
            total_progress = 50 + (value * 0.5)
        
        self.progress['value'] = total_progress
        self.progress_label.config(text=f"步骤: {step_name} - {status}")
        self.update_idletasks()

    def test_and_list_quants(self):
        self.test_quant_btn.config(state="disabled")
        self.logger.log("开始测试可用的量化类型...")
        
        def run_test():
            try:
                working_quants = get_tested_quant_map(self.logger)
                quant_keys = sorted(working_quants.keys())
                self.quant_combo['values'] = quant_keys
                if 'q4_k_m' in quant_keys:
                    self.quant_type.set('q4_k_m')
                elif quant_keys:
                    self.quant_type.set(quant_keys[0])
                self.logger.log("可用量化类型列表已更新。")
                messagebox.showinfo("测试完成", f"找到 {len(quant_keys)} 个可用的量化类型，已更新下拉列表。")
            except Exception as e:
                self.logger.log(f"测试量化类型时出错: {e}", "ERROR")
                messagebox.showerror("错误", f"测试失败: {e}")
            finally:
                self.test_quant_btn.config(state="normal")

        threading.Thread(target=run_test, daemon=True).start()

    def start_workflow(self):
        base_model = self.base_model_path.get().strip()
        lora = self.lora_path.get().strip()
        output = self.output_path.get().strip()
        weight = self.lora_weight.get()
        quant = self.quant_type.get().strip()

        if not all([base_model, lora, output, quant]):
            messagebox.showerror("错误", "所有路径和参数都必须填写！")
            return

        if not os.path.isfile(base_model) or not os.path.isfile(lora):
            messagebox.showerror("错误", "基础模型或LoRA文件不存在！")
            return

        self.start_btn.config(state="disabled")
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')

        def run_workflow():
            temp_dir = tempfile.mkdtemp()
            fixed_lora_path = os.path.join(temp_dir, "fixed_lora.safetensors")
            merged_model_path = os.path.join(temp_dir, "merged_model.safetensors")
            
            try:
                # --- 步骤 1: 修复 LoRA ---
                self.logger.log("--- 步骤 1/3: 修复/转换 LoRA ---")
                convert_lora_for_comfyui(
                    lora, fixed_lora_path, self.logger,
                    lambda v, s: self.update_progress("LORA_FIX", v, s)
                )

                # --- 步骤 2: 融合模型 ---
                self.logger.log("\n--- 步骤 2/3: 融合基础模型与 LoRA ---")
                merge_lora_A_B_with_base_model(
                    base_model, fixed_lora_path, merged_model_path, weight, self.logger,
                    lambda v, s: self.update_progress("MERGE", v, s)
                )

                # --- 步骤 3: GGUF 转换和量化 ---
                self.logger.log("\n--- 步骤 3/3: 转换为 GGUF 并量化 ---")
                # 加载刚刚融合的模型 state_dict
                merged_state_dict = load_file(merged_model_path)
                convert_to_gguf(
                    merged_state_dict, output, quant, self.logger,
                    lambda v, s: self.update_progress("QUANTIZE", v, s)
                )

                self.logger.log(f"\n🎉🎉🎉 工作流全部完成！🎉🎉🎉")
                self.logger.log(f"最终文件已保存到: {output}")
                messagebox.showinfo("成功", f"工作流执行完毕！\n最终文件保存在:\n{output}")

            except Exception as e:
                error_msg = traceback.format_exc()
                self.logger.log(f"\n❌ 工作流执行失败！\n{error_msg}", "ERROR")
                messagebox.showerror("执行失败", f"工作流中发生错误，请查看日志获取详情。\n\n错误: {e}")
            finally:
                # 清理临时文件
                try:
                    if os.path.exists(temp_dir):
                        import shutil
                        shutil.rmtree(temp_dir)
                        self.logger.log("临时文件已清理。")
                except Exception as e:
                    self.logger.log(f"清理临时文件失败: {e}", "WARN")
                
                self.start_btn.config(state="normal")
                self.progress['value'] = 0
                self.progress_label.config(text="就绪")

        threading.Thread(target=run_workflow, daemon=True).start()


if __name__ == "__main__":
    # 检查依赖是否满足
    try:
        import torch, safetensors, gguf, numpy, tqdm
    except ImportError as e:
        # 如果在主程序中捕获到，说明是首次运行，弹出提示框
        root = tk.Tk()
        root.withdraw() # 隐藏主窗口
        messagebox.showerror("依赖缺失", f"启动失败，缺少必要的库: {e}\n\n请在终端中运行以下命令安装所有必需的库:\npip install torch safetensors gguf numpy tqdm")
        sys.exit(1)

    app = WorkflowGUI()
    app.mainloop()

