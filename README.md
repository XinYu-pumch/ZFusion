# Z-Fusion: One-Click LoRA Merger & GGUF Quantizer

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)![License](https://img.shields.io/badge/License-MIT-green.svg)

一个为 Z-Image 等先进图像模型量身打造的一站式工具。它通过一个简单的图形界面，将 LoRA 与基础模型无缝融合，并量化为单个 GGUF 文件，让低显存设备（如 MacBook）也能轻松运行您最喜爱的自定义模型。

---

## 📖 项目的诞生 (The Origin Story)
Z image turbo因其尺寸小、真实性高、出图快的特点，一上线便获得大量摄取支持，ai-toolkit也第一时间支持了lora。

您是否曾想在显存有限的电脑（尤其是 Mac）上，运行强大的 Z-Image 模型并加载您精心训练的 LoRA？

如果您尝试过，可能会迅速发现一个令人沮丧的技术鸿沟：为低功耗设备而生的 GGUF 量化模型，无法动态加载 LoRA。唯一的出路是先将 LoRA 与基础模型合并，然后再对这个庞大的新模型进行量化。

然而，这个“先合并，后量化”的工作流本身就是一个“雷区”，充满了模型键名不匹配、库版本冲突和各种隐蔽的兼容性问题。**Z-Fusion** 正是为了解决这一系列棘手的问题而诞生的。

它的唯一使命就是：提供一个无缝、一键式的解决方案，让任何人都能轻松地将基础模型和 LoRA 转化为一个**直接可用**的、经过量化的 GGUF 文件。

注1：**本项目中处理的z image的lora由ai-toolkit产生，如云部署可参考**：https://www.xiangongyun.com/image/detail/835c05d8-d906-412e-a80f-6694475e5da7

**sd-scripts来源的lora未测试过**

注2：大显存电脑（如M4 pro 48G）无需运行本项目，若不爆显存，自测量化后无法给模型出图加速

## ✨ 核心功能 (Core Features)

*   **一站式工作流**: 自动完成 LoRA 修复、模型融合和 GGUF 量化，将复杂流程简化为一次点击。
*   **为低显存而生**: 专为解决在 MacBook 和其他低 VRAM 设备上运行自定义模型的难题。
*   **GUI界面**: 一目了然，上手简单
*   **智能 LoRA 修复**: 自动处理 `ai-toolkit` 等工具链生成的 LoRA 格式，从根源上解决兼容性问题。
*   **动态量化支持**: 工具能**自动测试**并只显示当前环境可用的 GGUF 量化类型，告别转换失败的烦恼。
*   **实时反馈与日志**: 通过进度条和详细日志，全程清晰掌握任务进展。
*   **零临时文件残留**: 自动在内存和临时目录中处理中间文件，任务结束后不留痕迹。

## 🖼️ 界面截图 (Screenshot)


![image](https://github.com/XinYu-pumch/ZFusion/blob/main/zfusion.png)
## 🎯 解决的痛点 (Pain Points Solved)

| 痛点 (Before)                                                | Z-Fusion 解决方案 (After)                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| GGUF 模型无法加载 LoRA，迫使用户面对复杂的手动流程。         | **自动化“先合并，后量化”** 的完整流程，屏蔽所有技术细节。 |
| 需要按顺序手动运行 3 个或更多不同的 Python 脚本。            | **一个脚本，一个界面**，自动完成所有步骤。                   |
| `ai-toolkit` 的 LoRA 在 ComfyUI 中报错，需要手动修复。       | **自动检测并修复** LoRA 格式，确保兼容性。                   |
| 不确定当前 `gguf` 库支持哪些量化类型，转换时频繁试错。       | **一键测试可用量化类型**，下拉菜单只显示有效选项。           |
| 命令行参数复杂，容易出错，对新手不友好。                     | **图形化操作**，只需点击浏览选择文件，设置简单参数。         |
| 流程中产生多个中间文件，需要手动管理和删除。                 | **自动管理临时文件**，任务结束后自动清理。                   |

## 🛠️ 环境要求 (Environment Requirements)

*   **操作系统**: Windows, macOS, or Linux
*   **Python**: 3.8 或更高版本

## 🚀 安装与使用 (Installation & Usage)

1.  **克隆或下载仓库**
    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2.  **安装依赖**
    项目所需的所有依赖都已在 `requirements.txt` 中列出。使用 pip 一键安装：
    ```bash
    pip install -r requirements.txt
    ```

3.  **运行 Z-Fusion**
    ```bash
    python zfusion.py
    ```

4.  **在 GUI 中操作**
    *   **基础模型**: 选择你的 `bf16` 底模 (`.safetensors`)。
    *   **LoRA模型**: 选择你的 `bf16` LoRA (`.safetensors`)。
    *   **输出文件**: 指定最终生成的 `.gguf` 文件的保存路径。
    *   **LoRA 权重**: 根据需要调整 LoRA 的融合强度。
    *   **量化类型**:
        *   （推荐）首先点击 **"测试并列出可用量化类型"** 按钮，程序会自动更新下拉列表。
        *   从下拉列表中选择一个你想要的量化等级（如 `q4_k_m` 是一个不错的平衡选择）。
    *   点击 **"开始工作流"** 按钮，然后泡杯咖啡，等待奇迹发生！

## ⚙️ 工作流详解 (Workflow Explained)

当您点击“开始”后，Z-Fusion 将在后台执行以下操作：
1.  **LoRA 修复**: 加载用户选择的 LoRA 文件，检查其键名格式。如果检测到是 `ai-toolkit` 的分离式 q/k/v 格式，则将其合并为标准的 `qkv` 格式。
2.  **模型融合**: 加载基础模型和修复后的 LoRA，根据用户设定的权重将 LoRA 的增量应用到基础模型的相应层上，生成一个临时的融合模型。
3.  **GGUF 转换与量化**: 读取融合后的模型，并使用 `gguf` 库将其转换为 GGUF 格式。在此过程中，根据用户选择的量化类型对模型的权重张量进行压缩。
4.  **清理**: 删除所有在流程中产生的临时文件。

## 🙏 致谢 (Acknowledgements)

本工具的核心逻辑基于以下三个脚本，并在此基础上进行了整合与优化。

*   `Zimage从ai-toolkit转换补全层级.py` #来自https://civitai.com/models/2174392?modelVersionId=2448609
*   `nextdit_lora_merger_AB.py` #来自我自己的编写
*   `convert_quantize.py` #基于City96的代码修改

## 📄 许可证 (License)

本项目采用 [MIT License](LICENSE) 授权。
