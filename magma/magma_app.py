import torch
from PIL import Image
import gradio as gr
from transformers import AutoModelForCausalLM, AutoProcessor
import warnings


# 忽略警告訊息
warnings.filterwarnings("ignore")


class MagmaAnalyzer:
    def __init__(self, use_gpu=True):
        
        # 檢查GPU是否可用
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"初始化Magma-8B模型(設備: {self.device})....")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Magma-8B", 
                trust_remote_code=True, 
                torch_dtype=self.dtype,
                
            )
            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Magma-8B", 
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)
            print("模型初始化完成")

        except Exception as e:
            print(f"模型加載失敗: {str(e)}")
            raise e
    
    def analyze_image(self, image, prompt_text="請用繁體中文描述這張圖片", max_tokens=512):
        """分析圖像並返回結果"""
        if image is None:
            return "請上傳圖片",None
        
        # 確保圖像是RGB模式
        try:
            image = image.convert("RGB")
            display_image = image
        except Exception as e:
            return f"無法處理圖像:{str(e)}",None
    
        
        # 使用繁體中文的對話格式
        convs = [
            {"role": "system", "content": "你是一個能看、能說的智能助手。請務必只使用繁體中文回答所有問題。"},
            {"role": "user", "content": f"<image_start><image><image_end>\n{prompt_text}，請務必使用繁體中文回答。"},
        ]
        try:
            prompt = self.processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
            
            # 特殊處理 - 當使用GPU時，先嘗試在CPU上準備輸入
            inputs = self.processor(images=[image], texts=prompt, return_tensors="pt")
              # 修正數據類型
            for key in inputs:
                if key in ['input_ids', 'attention_mask']:
                    inputs[key] = inputs[key].long()
                else:
                    # 保持其他張量在CPU上為float32
                    inputs[key] = inputs[key].float()
                  
                   
            
            # 生成參數
            generation_args = { 
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "use_cache": True,
                "num_beams": 1,
            }
            # 使用try/except捕獲維度錯誤
            try:
                # 將特定張量移至GPU（如果使用）
                if self.device == "cuda":
                    for key in ['input_ids', 'attention_mask']:
                        if key in inputs:
                            inputs[key] = inputs[key].to(self.device)
                else:
                    # 全部移至目標設備
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # 生成回答
                with torch.inference_mode():
                    generate_ids = self.model.generate(**inputs, **generation_args)
                
                # 解碼輸出
                generate_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
                response = self.processor.decode(generate_ids[0], skip_special_tokens=True).strip()
                return response, display_image
            
            except IndexError as e:
                 # 如果遇到維度錯誤，回退到CPU模式
                if "Dimension out of range" in str(e):
                    print("GPU模式遇到維度錯誤，嘗試使用CPU...")
                    
                    # 將模型暫時移至CPU
                    model_cpu = self.model.to("cpu")
                    inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}
                    
                    with torch.inference_mode():
                        generate_ids = model_cpu.generate(**inputs_cpu, **generation_args)
                    
                    # 解碼輸出
                    generate_ids = generate_ids[:, inputs_cpu["input_ids"].shape[-1]:]
                    response = self.processor.decode(generate_ids[0], skip_special_tokens=True).strip()
                    
                    # 將模型移回原設備
                    self.model = self.model.to(self.device)
                    
                    return response, display_image
                else:
                    raise e
                
        except Exception as e:
                print(f"分析圖像時出錯: {e}")
                import traceback
                traceback.print_exc()
                return f"分析圖像時出錯: {str(e)}", display_image
                

            

# 創建全局的分析器實例
analyzer = None

def analyze_image_wrapper(image, prompt, use_gpu, max_tokens):
    """包裝函數，處理圖像分析的主要邏輯"""
    global analyzer
    
    # 延遲加載模型
    if analyzer is None or analyzer.device != ("cuda" if use_gpu and torch.cuda.is_available() else "cpu"):
        try:
            analyzer = MagmaAnalyzer(use_gpu=use_gpu)
        except Exception as e:
            return f"加載模型失敗: {str(e)}",None
    
    try:
        result, display_image = analyzer.analyze_image(image, prompt, int(max_tokens))
        return result, display_image
    except Exception as e:
        return f"分析圖像時出錯: {str(e)}",image

# 創建Gradio界面
def create_interface():
    """創建Gradio界面"""
    with gr.Blocks(title="Magma-8B圖像分析") as demo:
        gr.Markdown("# Magma-8B 圖像分析工具")
        gr.Markdown("上傳圖片並獲取繁體中文描述")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 輸入區域
                input_image = gr.Image(type="pil", label="上傳圖片")
                prompt = gr.Textbox(label="提示文本", value="請用繁體中文詳細描述這張圖片的內容", lines=2)
                
                with gr.Row():
                    # 參數設置
                    use_gpu = gr.Checkbox(label="使用GPU", value=True)
                    max_tokens = gr.Slider(minimum=100, maximum=1024, value=512, step=32, label="最大生成長度")
                
                analyze_btn = gr.Button("分析圖片", variant="primary")
            
            with gr.Column(scale=1):
                # 輸出區域
                output_text = gr.Textbox(label="分析結果", lines=15)
                output_image = gr.Image(label="分析的圖片") # 新增顯示圖片元素
        
        # 綁定分析按鈕事件
        analyze_btn.click(
            fn=analyze_image_wrapper,
            inputs=[input_image, prompt, use_gpu, max_tokens],
            outputs=[output_text,output_image]
        )
        
        # 使用說明
        gr.Markdown("""
        ## 使用說明
        1. 上傳圖片或使用示例圖片
        2. 輸入提示文本（預設為"請用繁體中文詳細描述這張圖片的內容"）
        3. 選擇是否使用GPU（如果可用）
        4. 點擊"分析圖片"按鈕獲取結果
        
        ## 關於模型
        本應用使用Microsoft的Magma-8B多模態模型進行圖像分析。模型可以理解圖像內容並生成詳細的文本描述。
        """)
    
    return demo

# 啟動Gradio應用
if __name__ == "__main__":
    demo = create_interface()
    # 啟動界面，設置share=True可以生成一個公共鏈接分享給他人
    demo.launch(share=True)
else:
    app = create_interface()