# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Modified with full quantization support and pad_token fix
import os
import gguf
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from safetensors.torch import load_file, save_file

QUANTIZATION_THRESHOLD = 1024
REARRANGE_THRESHOLD = 512
MAX_TENSOR_NAME_LENGTH = 127
MAX_TENSOR_DIMS = 4

def get_quant_type(name):
    """安全获取量化类型"""
    return getattr(gguf.GGMLQuantizationType, name, None)

def get_file_type(name):
    """安全获取文件类型"""
    return getattr(gguf.LlamaFileType, name, None)

def test_quantization(qtype):
    """测试量化类型是否真正可用"""
    try:
        # 创建一个小的测试数组
        test_data = np.random.randn(256, 256).astype(np.float32)
        result = gguf.quants.quantize(test_data, qtype)
        return True
    except (NotImplementedError, Exception):
        return False

# 动态构建量化类型映射（只包含当前gguf版本支持的类型）
def build_quant_map():
    """根据当前gguf库版本构建支持的量化类型"""
    quant_definitions = [
        # (key, quant_type_name, file_type_name, description)
        ('f32', 'F32', None, "32-bit float (largest, best quality)"),
        ('f16', 'F16', 'MOSTLY_F16', "16-bit float"),
        ('bf16', 'BF16', 'MOSTLY_BF16', "bfloat16"),
        
        # Q2 系列
        ('q2_k', 'Q2_K', 'MOSTLY_Q2_K', "2-bit K-quant (smallest)"),
        ('q2_k_s', 'Q2_K_S', 'MOSTLY_Q2_K_S', "2-bit K-quant small"),
        
        # Q3 系列
        ('q3_k', 'Q3_K', 'MOSTLY_Q3_K_M', "3-bit K-quant"),
        ('q3_k_s', 'Q3_K_S', 'MOSTLY_Q3_K_S', "3-bit K-quant small"),
        ('q3_k_m', 'Q3_K_M', 'MOSTLY_Q3_K_M', "3-bit K-quant medium"),
        ('q3_k_l', 'Q3_K_L', 'MOSTLY_Q3_K_L', "3-bit K-quant large"),
        
        # Q4 系列
        ('q4_0', 'Q4_0', 'MOSTLY_Q4_0', "4-bit (legacy)"),
        ('q4_1', 'Q4_1', 'MOSTLY_Q4_1', "4-bit (legacy, better)"),
        ('q4_k', 'Q4_K', 'MOSTLY_Q4_K_M', "4-bit K-quant"),
        ('q4_k_s', 'Q4_K_S', 'MOSTLY_Q4_K_S', "4-bit K-quant small"),
        ('q4_k_m', 'Q4_K_M', 'MOSTLY_Q4_K_M', "4-bit K-quant medium (recommended)"),
        
        # Q5 系列
        ('q5_0', 'Q5_0', 'MOSTLY_Q5_0', "5-bit (legacy)"),
        ('q5_1', 'Q5_1', 'MOSTLY_Q5_1', "5-bit (legacy, better)"),
        ('q5_k', 'Q5_K', 'MOSTLY_Q5_K_M', "5-bit K-quant"),
        ('q5_k_s', 'Q5_K_S', 'MOSTLY_Q5_K_S', "5-bit K-quant small"),
        ('q5_k_m', 'Q5_K_M', 'MOSTLY_Q5_K_M', "5-bit K-quant medium"),
        
        # Q6 系列
        ('q6_k', 'Q6_K', 'MOSTLY_Q6_K', "6-bit K-quant (good quality)"),
        
        # Q8 系列
        ('q8_0', 'Q8_0', 'MOSTLY_Q8_0', "8-bit (best quantized quality)"),
        ('q8_1', 'Q8_1', 'MOSTLY_Q8_1', "8-bit (legacy)"),
    ]
    
    quant_map = {}
    quant_map_untested = {}
    
    for key, qtype_name, ftype_name, desc in quant_definitions:
        qtype = get_quant_type(qtype_name)
        if qtype is not None:
            ftype = get_file_type(ftype_name) if ftype_name else None
            quant_map_untested[key] = (qtype, ftype, desc)
    
    return quant_map_untested

def test_all_quants(quant_map):
    """测试所有量化类型是否真正可用"""
    working = {}
    broken = []
    
    print("🔍 Testing quantization implementations...")
    for key, (qtype, ftype, desc) in quant_map.items():
        if test_quantization(qtype):
            working[key] = (qtype, ftype, desc)
        else:
            broken.append(key)
    
    return working, broken

QUANT_MAP_UNTESTED = build_quant_map()
# 延迟测试，只在需要时进行
QUANT_MAP_TESTED = None
QUANT_BROKEN = None

def get_tested_quant_map():
    """获取经过测试的量化映射"""
    global QUANT_MAP_TESTED, QUANT_BROKEN
    if QUANT_MAP_TESTED is None:
        QUANT_MAP_TESTED, QUANT_BROKEN = test_all_quants(QUANT_MAP_UNTESTED)
    return QUANT_MAP_TESTED, QUANT_BROKEN

class ModelTemplate:
    arch = "invalid"
    shape_fix = False
    keys_detect = []
    keys_banned = []
    keys_hiprec = []
    keys_ignore = []

    def handle_nd_tensor(self, key, data):
        raise NotImplementedError(f"Tensor detected that exceeds dims supported by C++ code! ({key} @ {data.shape})")

class ModelFlux(ModelTemplate):
    arch = "flux"
    keys_detect = [
        ("transformer_blocks.0.attn.norm_added_k.weight",),
        ("double_blocks.0.img_attn.proj.weight",),
    ]
    keys_banned = ["transformer_blocks.0.attn.norm_added_k.weight",]

class ModelSD3(ModelTemplate):
    arch = "sd3"
    keys_detect = [
        ("transformer_blocks.0.attn.add_q_proj.weight",),
        ("joint_blocks.0.x_block.attn.qkv.weight",),
    ]
    keys_banned = ["transformer_blocks.0.attn.add_q_proj.weight",]

class ModelAura(ModelTemplate):
    arch = "aura"
    keys_detect = [
        ("double_layers.3.modX.1.weight",),
        ("joint_transformer_blocks.3.ff_context.out_projection.weight",),
    ]
    keys_banned = ["joint_transformer_blocks.3.ff_context.out_projection.weight",]

class ModelHiDream(ModelTemplate):
    arch = "hidream"
    keys_detect = [
        (
            "caption_projection.0.linear.weight",
            "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight"
        )
    ]
    keys_hiprec = [".ff_i.gate.weight", "img_emb.emb_pos"]

class CosmosPredict2(ModelTemplate):
    arch = "cosmos"
    keys_detect = [
        (
            "blocks.0.mlp.layer1.weight",
            "blocks.0.adaln_modulation_cross_attn.1.weight",
        )
    ]
    keys_hiprec = ["pos_embedder"]
    keys_ignore = ["_extra_state", "accum_"]

class ModelHyVid(ModelTemplate):
    arch = "hyvid"
    keys_detect = [
        (
            "double_blocks.0.img_attn_proj.weight",
            "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight",
        )
    ]

    def handle_nd_tensor(self, key, data):
        path = f"./fix_5d_tensors_{self.arch}.safetensors"
        if os.path.isfile(path):
            raise RuntimeError(f"5D tensor fix file already exists! {path}")
        fsd = {key: torch.from_numpy(data)}
        tqdm.write(f"5D key found in state dict! Manual fix required! - {key} {data.shape}")
        save_file(fsd, path)

class ModelWan(ModelHyVid):
    arch = "wan"
    keys_detect = [
        (
            "blocks.0.self_attn.norm_q.weight",
            "text_embedding.2.weight",
            "head.modulation",
        )
    ]
    keys_hiprec = [".modulation"]

class ModelLTXV(ModelTemplate):
    arch = "ltxv"
    keys_detect = [
        (
            "adaln_single.emb.timestep_embedder.linear_2.weight",
            "transformer_blocks.27.scale_shift_table",
            "caption_projection.linear_2.weight",
        )
    ]
    keys_hiprec = ["scale_shift_table"]

class ModelSDXL(ModelTemplate):
    arch = "sdxl"
    shape_fix = True
    keys_detect = [
        ("down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight",),
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight",
            "output_blocks.2.2.conv.weight", "output_blocks.5.2.conv.weight",
        ),
        ("label_emb.0.0.weight",),
    ]

class ModelSD1(ModelTemplate):
    arch = "sd1"
    shape_fix = True
    keys_detect = [
        ("down_blocks.0.downsamplers.0.conv.weight",),
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "input_blocks.9.0.op.weight",
            "output_blocks.2.1.conv.weight", "output_blocks.5.2.conv.weight", "output_blocks.8.2.conv.weight"
        ),
    ]

class ModelLumina2(ModelTemplate):
    arch = "lumina2"
    keys_detect = [
        ("cap_embedder.1.weight", "context_refiner.0.attention.qkv.weight")
    ]
    keys_hiprec = ["pad_token"]

arch_list = [ModelFlux, ModelSD3, ModelAura, ModelHiDream, CosmosPredict2, 
             ModelLTXV, ModelHyVid, ModelWan, ModelSDXL, ModelSD1, ModelLumina2]

def is_model_arch(model, state_dict):
    matched = False
    invalid = False
    for match_list in model.keys_detect:
        if all(key in state_dict for key in match_list):
            matched = True
            invalid = any(key in state_dict for key in model.keys_banned)
            break
    assert not invalid, "Model architecture not allowed for conversion!"
    return matched

def detect_arch(state_dict):
    model_arch = None
    for arch in arch_list:
        if is_model_arch(arch, state_dict):
            model_arch = arch()
            break
    assert model_arch is not None, "Unknown model architecture!"
    return model_arch

def list_quant_types():
    """列出所有支持的量化类型"""
    working, broken = get_tested_quant_map()
    
    print("\n📊 Supported quantization types (tested and working):\n")
    print(f"{'Type':<12} {'Description'}")
    print("-" * 60)
    
    # 按类别分组
    categories = {
        'Float': ['f32', 'f16', 'bf16'],
        'Q2': ['q2_k', 'q2_k_s'],
        'Q3': ['q3_k', 'q3_k_s', 'q3_k_m', 'q3_k_l'],
        'Q4': ['q4_0', 'q4_1', 'q4_k', 'q4_k_s', 'q4_k_m'],
        'Q5': ['q5_0', 'q5_1', 'q5_k', 'q5_k_s', 'q5_k_m'],
        'Q6': ['q6_k'],
        'Q8': ['q8_0', 'q8_1'],
    }
    
    available_count = 0
    for category, types in categories.items():
        available_in_cat = [t for t in types if t in working]
        if available_in_cat:
            print(f"\n{category}:")
            for qtype in available_in_cat:
                _, _, desc = working[qtype]
                print(f"  ✅ {qtype:<10} {desc}")
                available_count += 1
    
    print(f"\n📦 Total working: {available_count} quantization types")
    
    # 显示有类型定义但实现不工作的
    if broken:
        print(f"\n⚠️ Defined but NOT implemented in your gguf version:")
        print(f"   {', '.join(broken)}")
    
    # 显示完全不可用的类型
    all_possible = ['f32', 'f16', 'bf16', 'q2_k', 'q2_k_s', 'q3_k', 'q3_k_s', 'q3_k_m', 'q3_k_l',
                    'q4_0', 'q4_1', 'q4_k', 'q4_k_s', 'q4_k_m', 'q5_0', 'q5_1', 'q5_k', 'q5_k_s', 
                    'q5_k_m', 'q6_k', 'q8_0', 'q8_1']
    missing = [t for t in all_possible if t not in QUANT_MAP_UNTESTED]
    if missing:
        print(f"\n❌ Not defined in your gguf version:")
        print(f"   {', '.join(missing)}")
    
    print("\n💡 Recommendations based on working types:")
    if 'bf16' in working:
        print("  - Best quality:    bf16 (no compression)")
    if 'f16' in working:
        print("  - Good quality:    f16 (2x compression)")
    if 'q8_0' in working:
        print("  - Very good:       q8_0 (4x compression)")
    if 'q5_0' in working or 'q5_1' in working:
        working_q5 = 'q5_1' if 'q5_1' in working else 'q5_0'
        print(f"  - Good balance:    {working_q5} (5-6x compression)")
    if 'q4_0' in working or 'q4_1' in working:
        working_q4 = 'q4_1' if 'q4_1' in working else 'q4_0'
        print(f"  - Small size:      {working_q4} (7-8x compression)")
    
    print("\n🔧 To get more options: pip install --upgrade gguf")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate GGUF files with quantization support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python convert_quantize.py --src model.safetensors --quant q4_0
  python convert_quantize.py --src model.safetensors --dst output.gguf --quant q8_0
  python convert_quantize.py --list-quants
        '''
    )
    parser.add_argument("--src", help="Source model file (.safetensors, .ckpt, .pt)")
    parser.add_argument("--dst", help="Output GGUF file path")
    parser.add_argument("--quant", "-q", default="bf16", 
                       help="Quantization type (use --list-quants to see options)")
    parser.add_argument("--list-quants", "-l", action="store_true",
                       help="List all supported quantization types")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed progress")
    args = parser.parse_args()

    if args.list_quants:
        list_quant_types()
        exit(0)

    if not args.src:
        parser.error("--src is required (use --list-quants to see options)")

    if not os.path.isfile(args.src):
        parser.error(f"Source file not found: {args.src}")

    return args

def strip_prefix(state_dict):
    prefix = None
    for pfx in ["model.diffusion_model.", "model."]:
        if any([x.startswith(pfx) for x in state_dict.keys()]):
            prefix = pfx
            break

    if prefix is None:
        for pfx in ["net."]:
            if all([x.startswith(pfx) for x in state_dict.keys()]):
                prefix = pfx
                break

    if prefix is not None:
        logging.info(f"State dict prefix found: '{prefix}'")
        sd = {}
        for k, v in state_dict.items():
            if prefix not in k:
                continue
            k = k.replace(prefix, "")
            sd[k] = v
    else:
        logging.debug("State dict has no prefix")
        sd = state_dict

    return sd

def load_state_dict(path):
    if any(path.endswith(x) for x in [".ckpt", ".pt", ".bin", ".pth"]):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        for subkey in ["model", "module"]:
            if subkey in state_dict:
                state_dict = state_dict[subkey]
                break
        if len(state_dict) < 20:
            raise RuntimeError(f"pt subkey load failed: {state_dict.keys()}")
    else:
        state_dict = load_file(path)

    return strip_prefix(state_dict)

def safe_bfloat16_to_numpy(tensor, key):
    """安全地将bfloat16张量转换为numpy数组"""
    try:
        if 'pad_token' in key:
            tqdm.write(f"🔧 Special handling for pad_token: {key} {tensor.shape}")
        return tensor.to(torch.float32).numpy()
    except Exception as e:
        tqdm.write(f"Error converting {key}: {e}")
        try:
            return tensor.to(torch.float16).to(torch.float32).numpy()
        except Exception as e2:
            raise e2

def handle_tensors(writer, state_dict, model_arch, target_quant_type, fallback_quant_type, verbose=False):
    """处理张量并应用量化"""
    name_lengths = tuple(sorted(
        ((key, len(key)) for key in state_dict.keys()),
        key=lambda item: item[1],
        reverse=True,
    ))
    if not name_lengths:
        return
    max_name_len = name_lengths[0][1]
    if max_name_len > MAX_TENSOR_NAME_LENGTH:
        bad_list = ", ".join(f"{key!r} ({namelen})" for key, namelen in name_lengths if namelen > MAX_TENSOR_NAME_LENGTH)
        raise ValueError(f"Can only handle tensor names up to {MAX_TENSOR_NAME_LENGTH} characters.")

    stats = {'quantized': 0, 'kept_f32': 0, 'fallback': 0, 'total': 0}
    
    for key, data in tqdm(state_dict.items(), desc="Processing tensors"):
        old_dtype = data.dtype
        stats['total'] += 1

        if any(x in key for x in model_arch.keys_ignore):
            tqdm.write(f"Filtering ignored key: '{key}'")
            continue

        # 转换为numpy
        if data.dtype == torch.bfloat16:
            data = safe_bfloat16_to_numpy(data, key)
        elif data.dtype in [getattr(torch, "float8_e4m3fn", "_invalid"), getattr(torch, "float8_e5m2", "_invalid")]:
            data = data.to(torch.float16).numpy()
        else:
            data = data.numpy()

        n_dims = len(data.shape)
        data_shape = data.shape
        
        # 处理高维张量
        if len(data.shape) > MAX_TENSOR_DIMS:
            model_arch.handle_nd_tensor(key, data)
            continue

        # 计算参数数量
        n_params = 1
        for dim_size in data_shape:
            n_params *= dim_size

        # 决定量化类型
        if n_dims == 1:
            # 1D张量保持F32
            data_qtype = gguf.GGMLQuantizationType.F32
            stats['kept_f32'] += 1
        elif n_params <= QUANTIZATION_THRESHOLD:
            # 小张量保持F32
            data_qtype = gguf.GGMLQuantizationType.F32
            stats['kept_f32'] += 1
        elif any(x in key for x in model_arch.keys_hiprec):
            # 关键张量保持F32
            data_qtype = gguf.GGMLQuantizationType.F32
            stats['kept_f32'] += 1
        else:
            # 其他张量使用目标量化
            data_qtype = target_quant_type
            stats['quantized'] += 1

        # 形状重排（仅适用于某些模型）
        if (model_arch.shape_fix
            and n_dims > 1
            and n_params >= REARRANGE_THRESHOLD
            and (n_params / 256).is_integer()
            and not (data.shape[-1] / 256).is_integer()
        ):
            orig_shape = data.shape
            data = data.reshape(n_params // 256, 256)
            writer.add_array(f"comfy.gguf.orig_shape.{key}", tuple(int(dim) for dim in orig_shape))

        # 执行量化
        try:
            data = gguf.quants.quantize(data, data_qtype)
        except (NotImplementedError, AttributeError, gguf.QuantError) as e:
            # 使用备用量化
            if fallback_quant_type and data_qtype != fallback_quant_type:
                tqdm.write(f"⚠️ {data_qtype.name} failed for {key}, trying {fallback_quant_type.name}")
                try:
                    data_qtype = fallback_quant_type
                    data = gguf.quants.quantize(data, data_qtype)
                    stats['fallback'] += 1
                except Exception as e2:
                    tqdm.write(f"⚠️ Fallback also failed, keeping F32 for {key}")
                    data_qtype = gguf.GGMLQuantizationType.F32
            else:
                tqdm.write(f"⚠️ Quantization failed, keeping F32 for {key}: {e}")
                data_qtype = gguf.GGMLQuantizationType.F32

        if verbose:
            shape_str = f"{{{', '.join(str(n) for n in reversed(data.shape))}}}"
            tqdm.write(f"{f'%-{max_name_len + 4}s' % f'{key}'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

        writer.add_tensor(key, data, raw_dtype=data_qtype)
    
    return stats

def convert_file(path, dst_path=None, quant_type='bf16', verbose=False, interact=True, overwrite=False):
    """转换文件"""
    print(f"🔄 GGUF Converter with Quantization Support")
    print(f"   Input:  {path}")
    print(f"   Quant:  {quant_type.upper()}")
    
    # 获取经过测试的量化映射
    working_quants, broken_quants = get_tested_quant_map()
    
    state_dict = load_state_dict(path)
    model_arch = detect_arch(state_dict)
    print(f"   Arch:   {model_arch.arch}")
    print(f"   Params: {len(state_dict)} tensors")

    # 获取量化配置
    quant_key = quant_type.lower()
    
    # 检查是否可用
    if quant_key not in working_quants:
        if quant_key in broken_quants:
            print(f"❌ Quantization type '{quant_type}' is defined but NOT implemented in your gguf version!")
        else:
            print(f"❌ Unsupported quantization type: {quant_type}")
        print(f"\n   Working types: {', '.join(sorted(working_quants.keys()))}")
        print(f"\n   Use --list-quants to see all options with descriptions")
        return None, None
        
    target_qtype, ftype_gguf, desc = working_quants[quant_key]
    ftype_name = quant_type.upper()
    print(f"   Type:   {desc}")
    
    # 选择备用量化类型
    fallback_qtype = None
    fallback_options = ['f16', 'bf16', 'f32']
    for fb in fallback_options:
        if fb in working_quants and fb != quant_key:
            fallback_qtype = working_quants[fb][0]
            print(f"   Fallback: {fb.upper()}")
            break

    if dst_path is None:
        dst_path = f"{os.path.splitext(path)[0]}-{ftype_name}.gguf"
    elif "{ftype}" in dst_path:
        dst_path = dst_path.replace("{ftype}", ftype_name)

    print(f"   Output: {dst_path}")

    if os.path.isfile(dst_path) and not overwrite:
        if interact:
            response = input("\n⚠️ Output file exists. Overwrite? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                return None, None
        else:
            raise OSError("Output exists and overwriting is disabled!")

    # 创建writer
    print(f"\n📝 Creating GGUF file...")
    writer = gguf.GGUFWriter(path=None, arch=model_arch.arch)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    if ftype_gguf is not None:
        writer.add_file_type(ftype_gguf)

    # 处理张量
    print(f"\n🔧 Processing tensors...")
    stats = handle_tensors(writer, state_dict, model_arch, target_qtype, fallback_qtype, verbose)
    
    # 写入文件
    print(f"\n💾 Writing output file...")
    writer.write_header_to_file(path=dst_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    # 显示结果
    original_size = os.path.getsize(path)
    output_size = os.path.getsize(dst_path)
    compression = original_size / output_size if output_size > 0 else 1
    
    print(f"\n✅ Conversion complete!")
    print(f"")
    print(f"   📊 Statistics:")
    print(f"      Total tensors:     {stats['total']}")
    print(f"      Quantized:         {stats['quantized']}")
    print(f"      Kept F32:          {stats['kept_f32']}")
    if stats['fallback'] > 0:
        print(f"      Used fallback:     {stats['fallback']}")
    print(f"")
    print(f"   📦 File sizes:")
    print(f"      Input:             {original_size / (1024**3):.2f} GB")
    print(f"      Output:            {output_size / (1024**3):.2f} GB")
    print(f"      Compression:       {compression:.2f}x")
    print(f"")
    print(f"   🎉 Output saved to: {dst_path}")

    return dst_path, model_arch

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = parse_args()
    convert_file(args.src, args.dst, args.quant, args.verbose)
