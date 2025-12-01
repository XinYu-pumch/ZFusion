#!/usr/bin/env python3
"""
LoRA Converter for z-image-turbo / Lumina2 (可视化版本)
将分离的 to_q/to_k/to_v LoRA 转换为合并的 qkv 格式
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from collections import defaultdict
import traceback

try:
    import torch
    from safetensors.torch import load_file, save_file
    from safetensors import safe_open
except ImportError as e:
    messagebox.showerror("依赖缺失", f"缺少必要库：{e}\n\n请在终端运行：\npip install torch safetensors")
    sys.exit(1)


def convert_lora(input_path, output_path, progress_callback=None):
    if progress_callback:
        progress_callback(0, "正在加载 LoRA 文件...")

    if input_path.endswith('.safetensors'):
        lora_dict = load_file(input_path)
    elif input_path.endswith(('.pt', '.pth')):
        lora_dict = torch.load(input_path, map_location='cpu')
    else:
        raise ValueError("仅支持 .safetensors / .pt / .pth 文件")

    total_keys = len(lora_dict)
    if progress_callback:
        progress_callback(10, f"加载完成，共 {total_keys} 个键")

    layer_groups = defaultdict(lambda: defaultdict(dict))
    output_dict = {}
    converted_count = 0

    # 收集 alpha（用于后面放大 3 倍）
    alpha_values = {}
    for k, v in lora_dict.items():
        if '.alpha' in k and not k.endswith('.weight'):
            alpha_values[k] = v

    processed = 0
    for key, value in lora_dict.items():
        processed += 1
        if progress_callback and processed % 50 == 0:
            progress_callback(10 + 40 * processed // total_keys, f"正在解析键... ({processed}/{total_keys})")

        # 1. to_out.0 → out 重命名（Lumina2 兼容）
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

        # 2. 跳过单独的 alpha（后面统一处理）
        if '.attention.to_' in key and '.alpha' in key:
            continue

        # 3. 收集需要合并的 q/k/v 注意力层
        if '.attention.to_' in key and any(x in key for x in ('.to_q.', '.to_k.', '.to_v.')):
            parts = key.split('.')
            layer_idx = None
            attn_type = None
            lora_type = None

            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts):
                    layer_idx = parts[i + 1]
                elif p in ('to_q', 'to_k', 'to_v'):
                    attn_type = p[3:]           # q / k / v
                elif p in ('lora_A', 'lora_B'):
                    lora_type = p

            if layer_idx and attn_type and lora_type:
                # 构造不含 to_q/to_k/to_v 的 base_key
                base_parts = []
                skip = False
                for p in parts:
                    if p in ('to_q', 'to_k', 'to_v'):
                        skip = True
                        continue
                    if skip:
                        base_parts.append(p)
                        skip = False
                    else:
                        base_parts.append(p)
                base_key = '.'.join(base_parts[:-2])  # 去掉 lora_X.weight
                layer_groups[base_key][attn_type][lora_type] = value
                continue

        # 其他键直接复制
        output_dict[key] = value

    if progress_callback:
        progress_callback(60, f"找到 {len(layer_groups)} 个需要合并的注意力层")

    # ==================== 合并 qkv ====================
    step = 30.0 / max(len(layer_groups), 1)
    current = 0

    for base_key, qkv_dict in layer_groups.items():
        current += 1
        if progress_callback:
            progress_callback(60 + step * current, f"正在合并层 {current}/{len(layer_groups)}")

        if not all(x in qkv_dict for x in ('q', 'k', 'v')):
            continue

        qB = qkv_dict['q'].get('lora_B')
        kB = qkv_dict['k'].get('lora_B')
        vB = qkv_dict['v'].get('lora_B')
        qA = qkv_dict['q'].get('lora_A')
        kA = qkv_dict['k'].get('lora_A')
        vA = qkv_dict['v'].get('lora_A')

        if None in (qB, kB, vB, qA, kA, vA):
            continue

        try:
            assert qB.shape == kB.shape == vB.shape
            assert qA.shape == kA.shape == vA.shape
            hidden_dim, rank = qB.shape

            # 构造 block-diagonal lora_B
            qkv_B = torch.zeros(3 * hidden_dim, 3 * rank, dtype=qB.dtype)
            qkv_B[:hidden_dim, :rank]                 = qB
            qkv_B[hidden_dim:2*hidden_dim, rank:2*rank] = kB
            qkv_B[2*hidden_dim:, 2*rank:]             = vB

            # 垂直堆叠 lora_A
            qkv_A = torch.cat([qA, kA, vA], dim=0)

            output_dict[f"{base_key}.qkv.lora_B.weight"] = qkv_B
            output_dict[f"{base_key}.qkv.lora_A.weight"] = qkv_A
            converted_count += 1

            # alpha 处理：原来是 alpha / rank，现在 rank×3 → alpha×3
            alpha_key_q = f"{base_key}.to_q.alpha"
            orig_alpha = lora_dict.get(alpha_key_q) or lora_dict.get(f"{base_key}.to_q.lora_A.alpha")
            if orig_alpha is not None:
                output_dict[f"{base_key}.qkv.alpha"] = orig_alpha * 3.0

        except Exception as e:
            print(f"合并失败 {base_key}: {e}")
            traceback.print_exc()

    if progress_callback:
        progress_callback(95, "正在保存文件...")

    # 元数据
    metadata = {}
    if input_path.endswith('.safetensors'):
        try:
            with safe_open(input_path, framework="pt", device="cpu") as f:
                metadata = f.metadata() or {}
        except:
            pass
    metadata['converted_for'] = 'z-image-turbo / Lumina2 (GUI converter)'
    metadata['conversion_script'] = 'convert_lora_to_zimage_gui.py'

    save_file(output_dict, output_path, metadata=metadata)

    if progress_callback:
        progress_callback(100, f"完成！共转换 {converted_count} 个注意力层")

    return converted_count


# ====================== GUI ======================
class LoRAConverterGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LoRA → z-image-turbo / Lumina2 转换器")
        self.geometry("660x360")
        self.resizable(False, False)

        # 样式
        style = ttk.Style(self)
        style.theme_use('clam')

        # 主框架
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Zimage(ToKit2Comy)Fok&PAseer", font=("Segoe UI", 16, "bold")).pack(pady=(0, 20))

        # 输入文件
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=8)
        ttk.Label(input_frame, text="输入 LoRA：", width=12).pack(side=tk.LEFT)
        self.input_path = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_path, width=50).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(input_frame, text="浏览...", command=self.browse_input).pack(side=tk.RIGHT)

        # 输出文件
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill=tk.X, pady=8)
        ttk.Label(output_frame, text="输出路径：", width=12).pack(side=tk.LEFT)
        self.output_path = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(output_frame, text="浏览...", command=self.browse_output).pack(side=tk.RIGHT)

        # 进度条
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=20)

        self.status_label = ttk.Label(main_frame, text="就绪", foreground="gray")
        self.status_label.pack(pady=5)

        # 开始按钮
        self.convert_btn = ttk.Button(main_frame, text="开始转换", command=self.start_conversion)
        self.convert_btn.pack(pady=10)

        # 日志区域
        log_frame = ttk.LabelFrame(main_frame, text="日志")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.log_text = tk.Text(log_frame, height=6, state='disabled', font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.update_idletasks()

    def browse_input(self):
        path = filedialog.askopenfilename(
            title="选择输入 LoRA 文件",
            filetypes=[("LoRA files", "*.safetensors *.pt *.pth"), ("All files", "*.*")]
        )
        if path:
            self.input_path.set(path)
            # 自动填充输出路径
            dir_name = os.path.dirname(path)
            base = os.path.basename(path)
            name, ext = os.path.splitext(base)
            suggested = os.path.join(dir_name, f"{name}_zimage{ext}")
            self.output_path.set(suggested)

    def browse_output(self):
        path = filedialog.asksaveasfilename(
            title="保存转换后的 LoRA",
            defaultextension=".safetensors",
            filetypes=[("SafeTensors", "*.safetensors"), ("PyTorch", "*.pt")]
        )
        if path:
            self.output_path.set(path)

    def update_progress(self, value, status):
        self.progress['value'] = value
        self.status_label.config(text=status)
        self.log(f"[{value:.1f}%] {status}")
        self.update_idletasks()

    def start_conversion(self):
        in_path = self.input_path.get().strip()
        out_path = self.output_path.get().strip()

        if not in_path or not os.path.isfile(in_path):
            messagebox.showerror("错误", "请选择有效的输入 LoRA 文件")
            return
        if not out_path:
            messagebox.showerror("错误", "请指定输出文件路径")
            return

        if os.path.exists(out_path):
            if not messagebox.askyesno("文件已存在", f"文件已存在，是否覆盖？\n{out_path}"):
                return

        self.convert_btn.config(state='disabled')
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')
        self.progress['value'] = 0
        self.update_progress(0, "开始转换...")

        def run():
            try:
                converted = convert_lora(
                    in_path,
                    out_path,
                    progress_callback=self.update_progress
                )
                self.update_progress(100, f"转换完成！共合并 {converted} 个注意力层")
                messagebox.showinfo("成功", f"转换完成！\n已保存到：\n{out_path}")
            except Exception as e:
                error_msg = ''.join(traceback.format_exc())
                self.log(error_msg)
                messagebox.showerror("转换失败", f"转换过程中发生错误：\n{e}")
            finally:
                self.convert_btn.config(state='normal')

        # 在新线程运行，避免界面卡死
        import threading
        threading.Thread(target=run, daemon=True).start()


if __name__ == "__main__":
    app = LoRAConverterGUI()
    app.mainloop()