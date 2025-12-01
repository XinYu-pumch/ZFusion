#!/usr/bin/env python3
"""
NextDiT LoRA Merger (支持 lora_A/lora_B 格式)
"""

import torch
import argparse
import os
import sys
from pathlib import Path

try:
    from safetensors.torch import load_file, save_file
except ImportError:
    print("Error: safetensors not installed. Please run: pip install safetensors")
    sys.exit(1)

def validate_file_exists(file_path, file_type):
    """验证文件是否存在"""
    if not os.path.exists(file_path):
        print(f"Error: {file_type} file not found: {file_path}")
        sys.exit(1)
    return True

def analyze_lora_format(lora_dict):
    """分析LoRA的格式"""
    formats = {
        'lora_A_B': 0,  # lora_A.weight + lora_B.weight
        'lora_up_down': 0,  # lora_up.weight + lora_down.weight
        'other': 0
    }
    
    prefixes = set()
    
    for key in lora_dict.keys():
        # 收集前缀
        if 'diffusion_model.' in key:
            prefixes.add('diffusion_model.')
        
        # 分析格式
        if '.lora_A.weight' in key:
            formats['lora_A_B'] += 1
        elif '.lora_up.weight' in key:
            formats['lora_up_down'] += 1
        else:
            formats['other'] += 1
    
    main_format = max(formats, key=formats.get)
    return main_format, formats, prefixes

def extract_lora_A_B_pairs(lora_dict, prefix_to_remove='diffusion_model.'):
    """提取lora_A/lora_B权重对并移除前缀"""
    lora_pairs = {}
    
    for key in lora_dict.keys():
        if '.lora_A.weight' in key:
            # 移除前缀并获取基础层名称
            clean_key = key
            if prefix_to_remove and clean_key.startswith(prefix_to_remove):
                clean_key = clean_key[len(prefix_to_remove):]
            
            base_key = clean_key.replace('.lora_A.weight', '.weight')
            lora_b_key = key.replace('.lora_A.weight', '.lora_B.weight')
            
            if lora_b_key in lora_dict:
                lora_pairs[base_key] = {
                    'A': lora_dict[key],  # 对应down
                    'B': lora_dict[lora_b_key],  # 对应up
                    'original_A_key': key,
                    'original_B_key': lora_b_key
                }
    
    return lora_pairs

def apply_lora_A_B_to_weight(base_weight, lora_A, lora_B, strength):
    """将LoRA_A/B增量应用到基础权重"""
    try:
        # LoRA增量 = lora_B @ lora_A (注意顺序！)
        if len(lora_A.shape) == 2 and len(lora_B.shape) == 2:
            lora_delta = torch.mm(lora_B, lora_A) * strength
        elif len(lora_A.shape) == 4 and len(lora_B.shape) == 4:
            lora_delta = torch.nn.functional.conv2d(
                lora_A.permute(1, 0, 2, 3), 
                lora_B
            ).permute(1, 0, 2, 3) * strength
        else:
            print(f"Warning: Unsupported LoRA shape - A: {lora_A.shape}, B: {lora_B.shape}")
            return base_weight
        
        # 验证维度匹配
        if base_weight.shape != lora_delta.shape:
            print(f"Warning: Shape mismatch - base: {base_weight.shape}, delta: {lora_delta.shape}")
            return base_weight
        
        # 应用增量
        return base_weight + lora_delta
    
    except Exception as e:
        print(f"Error applying LoRA: {e}")
        return base_weight

def merge_lora_A_B_with_base_model(base_model_path, lora_path, output_path, lora_strength=1.0, verbose=False):
    """融合lora_A/B格式的LoRA到基础模型"""
    print("=== NextDiT LoRA Merger (lora_A/B格式) ===")
    print(f"Base Model: {base_model_path}")
    print(f"LoRA: {lora_path}")
    print(f"Output: {output_path}")
    print(f"LoRA Strength: {lora_strength}")
    
    # 验证输入文件
    validate_file_exists(base_model_path, "Base model")
    validate_file_exists(lora_path, "LoRA")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 加载基础模型
    print("\n📥 Loading base model...")
    try:
        base_model = load_file(base_model_path)
        print(f"✓ Base model loaded: {len(base_model.keys())} parameters")
    except Exception as e:
        print(f"Error loading base model: {e}")
        sys.exit(1)
    
    # 加载LoRA
    print("\n📥 Loading LoRA...")
    try:
        lora_model = load_file(lora_path)
        print(f"✓ LoRA loaded: {len(lora_model.keys())} parameters")
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        sys.exit(1)
    
    # 分析LoRA格式
    print("\n🔍 Analyzing LoRA format...")
    lora_format, format_counts, prefixes = analyze_lora_format(lora_model)
    print(f"✓ LoRA format detected: {lora_format}")
    print(f"  Format counts: {format_counts}")
    print(f"  Prefixes found: {prefixes}")
    
    if lora_format != 'lora_A_B':
        print(f"❌ Error: This tool is designed for lora_A/B format, but found: {lora_format}")
        sys.exit(1)
    
    # 确定要移除的前缀
    prefix_to_remove = 'diffusion_model.' if 'diffusion_model.' in prefixes else None
    if prefix_to_remove:
        print(f"✓ Will remove prefix: '{prefix_to_remove}'")
    
    # 提取LoRA对
    print("\n🔍 Extracting LoRA A/B pairs...")
    lora_pairs = extract_lora_A_B_pairs(lora_model, prefix_to_remove)
    print(f"✓ Found {len(lora_pairs)} LoRA A/B pairs")
    
    if verbose:
        print("📋 LoRA pairs preview (first 10):")
        for i, (base_key, pair_info) in enumerate(list(lora_pairs.items())[:10]):
            A_shape = pair_info['A'].shape
            B_shape = pair_info['B'].shape
            print(f"  {i+1}. {base_key}")
            print(f"     A: {A_shape} (from {pair_info['original_A_key']})")
            print(f"     B: {B_shape} (from {pair_info['original_B_key']})")
    
    # 创建融合后的模型
    print("\n🔄 Merging models...")
    merged_model = base_model.copy()
    applied_count = 0
    skipped_count = 0
    
    # 应用LoRA权重
    for base_key, pair_info in lora_pairs.items():
        if base_key in merged_model:
            original_weight = merged_model[base_key]
            merged_weight = apply_lora_A_B_to_weight(
                original_weight,
                pair_info['A'],
                pair_info['B'],
                lora_strength
            )
            merged_model[base_key] = merged_weight
            applied_count += 1
            
            if verbose:
                print(f"  ✓ Applied LoRA to: {base_key}")
                
                # 计算变化幅度
                delta = merged_weight - original_weight
                orig_magnitude = torch.abs(original_weight).mean().item()
                delta_magnitude = torch.abs(delta).mean().item()
                relative_change = delta_magnitude / orig_magnitude if orig_magnitude > 0 else 0
                print(f"    Relative change: {relative_change:.6f}")
        else:
            skipped_count += 1
            if verbose:
                print(f"  ⚠ Skipped (not in base): {base_key}")
    
    print(f"✓ Applied LoRA to {applied_count} layers")
    if skipped_count > 0:
        print(f"⚠ Skipped {skipped_count} layers (not found in base model)")
        
        # 显示一些未匹配的层
        missing_keys = [k for k in lora_pairs.keys() if k not in merged_model]
        if missing_keys:
            print(f"📋 Some missing keys (first 5):")
            for key in missing_keys[:5]:
                print(f"  - {key}")
    
    # 验证融合效果
    if applied_count == 0:
        print("\n❌ ERROR: No LoRA layers were applied! This suggests a key mismatch.")
        print("🔧 Debugging info:")
        
        # 显示一些底模的key用于对比
        base_keys = list(base_model.keys())[:10]
        lora_keys = list(lora_pairs.keys())[:5]
        
        print("📋 Sample base model keys:")
        for key in base_keys:
            print(f"  {key}")
        
        print("📋 Sample LoRA target keys:")
        for key in lora_keys:
            print(f"  {key}")
        
        sys.exit(1)
    
    # 保存融合后的模型
    print(f"\n💾 Saving merged model...")
    try:
        save_file(merged_model, output_path)
        file_size = os.path.getsize(output_path) / (1024**3)  # GB
        print(f"✓ Model saved successfully: {output_path}")
        print(f"✓ File size: {file_size:.2f} GB")
    except Exception as e:
        print(f"Error saving model: {e}")
        sys.exit(1)
    
    # 计算整体变化统计
    total_params = sum(tensor.numel() for tensor in merged_model.values())
    print(f"\n📊 Merge Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - LoRA layers applied: {applied_count}")
    print(f"  - Coverage: {applied_count/len(base_model)*100:.1f}% of base model layers")
    
    match_rate = applied_count / len(lora_pairs) if len(lora_pairs) > 0 else 0
    print(f"  - LoRA utilization: {match_rate:.1%}")
    
    if match_rate >= 0.8:
        print("\n🎉 SUCCESS: LoRA merged successfully!")
    elif match_rate >= 0.5:
        print(f"\n⚠️ PARTIAL SUCCESS: Only {match_rate:.1%} of LoRA layers were applied.")
        print("   The model should still show some LoRA effects, but may not be optimal.")
    else:
        print(f"\n❌ LOW SUCCESS: Only {match_rate:.1%} of LoRA layers were applied.")
        print("   The LoRA effects may be very weak or unnoticeable.")
    
    return merged_model

def main():
    parser = argparse.ArgumentParser(
        description='NextDiT LoRA Merger - Support for lora_A/lora_B format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python nextdit_lora_merger_AB.py --base base_model.safetensors --lora my_lora.safetensors --output merged_model.safetensors
  python nextdit_lora_merger_AB.py -b base.safetensors -l lora.safetensors -o merged.safetensors --strength 2.0 --verbose
        '''
    )
    
    parser.add_argument('--base', '-b', required=True,
                        help='Path to base model (.safetensors)')
    parser.add_argument('--lora', '-l', required=True,
                        help='Path to LoRA model (.safetensors)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output path for merged model (.safetensors)')
    parser.add_argument('--strength', '-s', type=float, default=1.0,
                        help='LoRA strength (default: 1.0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.strength < 0 or args.strength > 5:
        print("Warning: LoRA strength outside normal range (0-5)")
    
    # 执行合并
    try:
        merge_lora_A_B_with_base_model(
            args.base,
            args.lora,
            args.output,
            args.strength,
            args.verbose
        )
    except KeyboardInterrupt:
        print("\n\n⚠ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
