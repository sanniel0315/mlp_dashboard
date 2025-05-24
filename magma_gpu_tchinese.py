import torch
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoProcessor

def select_image():
    """使用文件對話框選擇圖像文件"""
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title="選擇圖片", filetypes=[("圖片文件", "*.jpg *.jpeg *.png *.gif *.bmp")])


if not torch.cuda.is_available():
    print("警告: 未检测到GPU，将使用CPU模式")
    device = "cpu"
    dtype = torch.float32
else:
    device = "cuda"
    dtype = torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32以提高性能
    
print(f"使用设备: {device}，数据类型: {dtype}")

# 加载模型和处理器
print("正在加载Magma-8B模型...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Magma-8B", 
    trust_remote_code=True, 
    torch_dtype=dtype,
    low_cpu_mem_usage=True
)
processor = AutoProcessor.from_pretrained(
    "microsoft/Magma-8B", 
    trust_remote_code=True
)

model = model.to(device)
print(f"模型已加载并设置为{device}模式")

# 选择图片
print("请选择要分析的图片...")
image_path = select_image()
if not image_path:
    print("未选择图片，程序退出")
    exit()

# 加载图像
image = Image.open(image_path).convert("RGB")
print(f"已加载图片: {image_path}")

# 提示文本
prompt_text = input("请输入提示文本 (预设: '请用繁体中文描述这张图片'): ") or "请用繁体中文描述这张图片"

# 使用繁体中文的对话格式
convs = [
    {"role": "system", "content": "你是一个能看、能说的智能助手。请务必只使用繁体中文回答所有问题。"},
    {"role": "user", "content": f"<image_start><image><image_end>\n{prompt_text}，请务必使用繁体中文回答，不要使用英文。"},
]
prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)

# 处理输入
print("处理输入...")
try:
    # 初始处理
    inputs = processor(images=[image], texts=prompt, return_tensors="pt")
    
    # 确保ID类型为长整型
    for key in ['input_ids', 'attention_mask']:
        if key in inputs:
            inputs[key] = inputs[key].long()
    
    # 确保图像相关的Tensor是正确类型和形状
    if 'pixel_values' in inputs:
        # 检查并修正形状
        if inputs['pixel_values'].dim() == 4:
            inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        
        # 确保数据类型正确
        inputs['pixel_values'] = inputs['pixel_values'].to(dtype)
    
    if 'image_sizes' in inputs:
        # 确保image_sizes是整数类型
        inputs['image_sizes'] = inputs['image_sizes'].long()
        
        # 检查并修正形状
        if inputs['image_sizes'].dim() == 2:
            inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
    
    # 移动到设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print("输入数据形状:")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    
    # 生成参数
    generation_args = { 
        "max_new_tokens": 512,
        "temperature": 0.3,
        "do_sample": True,
        "top_p": 0.9,
        "use_cache": True,
        "num_beams": 1,
    }
    
    # 使用CUDA流来优化GPU使用
    with torch.cuda.stream(torch.cuda.Stream()) if device == "cuda" else nullcontext():
        print("生成回答中...")
        with torch.inference_mode():
            generate_ids = model.generate(**inputs, **generation_args)
        
        # 解码输出
        generate_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
        response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()
    
    print("\n" + "="*50)
    print("分析结果:")
    print("="*50)
    print(response)
    
    # 提供保存选项
    save_option = input("\n是否要保存结果到文件? (y/n): ").lower()
    if save_option == 'y':
        output_file = f"{os.path.splitext(os.path.basename(image_path))[0]}_分析结果.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"图片: {image_path}\n")
            f.write(f"提示: {prompt_text}\n")
            f.write("="*50 + "\n")
            f.write(response + "\n")
        print(f"结果已保存到: {output_file}")

except Exception as e:
    print(f"GPU模式发生错误: {e}")
    import traceback
    traceback.print_exc()
    
    # 自动降级到CPU模式
    print("\n自动降级到CPU模式...")
    try:
        model = model.to("cpu").to(torch.float32)
        inputs = processor(images=[image], texts=prompt, return_tensors="pt")
        
        print("生成回答中 (CPU模式)...")
        with torch.inference_mode():
            generate_ids = model.generate(
                **inputs, 
                max_new_tokens=256,  # 在CPU上减少生成长度以加快速度
                do_sample=False,     # 关闭采样以加快速度
                num_beams=1
            )
        
        generate_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
        response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()
        
        print("\n" + "="*50)
        print("分析结果 (CPU模式):")
        print("="*50)
        print(response)
    except Exception as e2:
        print(f"CPU模式也失败了: {e2}")
        print("建议尝试其他多模态模型或联系Magma模型的维护者")

# 需要引入nullcontext
from contextlib import nullcontext