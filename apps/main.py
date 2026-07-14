from io import BytesIO
import torch
import litserve as ls
from PIL import Image
from fastapi import UploadFile, File
from unsloth import FastVisionModel
from transformers import AutoProcessor
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from pix2text import Pix2Text
class Prompt(BaseModel):
    text: str
CONFIG = {
    # Data paths
    'train_file': 'datagen_train.txt',  # 1,613 Datagen samples
    'valid_file': 'data_CROHME2019/crohme2019_valid.txt',  # 986 samples
    'test_file': 'data_CROHME2019/crohme2019_test.txt',  # 1,199 samples
    'crohme_base': 'data_CROHME2019',
    
    # Model settings - CONTINUE TRAINING FROM CROHME MODEL
    'base_model_name': 'unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit',  # ⭐ Must match LoRA adapter's base model (3B, not 7B)
    #'model_name': 'Qwen/Qwen2.5-VL-3B-Instruct',  # For processor (use official model name)
    'pretrained_model': os.getenv('MODEL_PATH', '/app/models/lora_model_qwen3vl.zip'),
    'max_seq_length': 2048,
    'use_4bit': True,
    
    # Training settings - LOWER LR for continue training
    'batch_size': 2,
    'gradient_accumulation_steps': 4,  # Effective batch = 2 * 4 = 8
    'learning_rate': 5e-5,  # ⭐ Lower LR (was 2e-4) for fine-tuning pretrained model
    'num_epochs': 3,
    'warmup_steps': 20,  # ⭐ Shorter warmup (was 50)
    'logging_steps': 10,
    'eval_steps': 50,  # Validate every 50 steps
    'save_steps': 100,
    
    # Early Stopping settings
    'early_stopping_patience': 3,  # Stop if no improvement for 3 evals
    'early_stopping_threshold': 0.0,  # Minimum change to qualify as improvement
    
    # LoRA settings
    'lora_r': 16,
    'lora_alpha': 16,
    'lora_dropout': 0.05,
    
    # Output
    'output_dir': 'outputs_datagen_continue',  # ⭐ Different folder
    'plot_dir': 'plots_datagen_continue',
    
    # Image settings
    'img_size': (512, 128),
    'line_width': 2,
}

# Create directories
os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['plot_dir'], exist_ok=True)
def load_unsloth_qwen():
#     model_name = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"
#     processor = AutoProcessor.from_pretrained(model_name)
#     return model, tokenizer, processor
    from peft import PeftModel
    print("🔄 Loading CROHME-trained model for continue training...\n")

    # Check if pretrained LoRA adapter exists
    lora_path = CONFIG['pretrained_model']
    if lora_path.endswith('.zip') or not os.path.isdir(lora_path):
        # Handle zip file
        if os.path.exists(lora_path) and not os.path.isdir(lora_path):
            print(f"⚠️  Found zip file: {lora_path}")
            extracted_path = lora_path + '_extracted'
            if not os.path.exists(extracted_path):
                print(f"📦 Extracting to {extracted_path}...")
                import zipfile
                with zipfile.ZipFile(lora_path, 'r') as zip_ref:
                    zip_ref.extractall(extracted_path)
                print(f"✅ Extracted successfully")
            lora_path = extracted_path
            CONFIG['pretrained_model'] = lora_path

    if not os.path.exists(lora_path):
        print(f"❌ ERROR: Pretrained model not found at '{lora_path}'")
        print(f"\n💡 Please ensure you have the CROHME-trained model.")
        raise FileNotFoundError(f"Model not found: {lora_path}")

    print(f"✅ Found LoRA adapter: {lora_path}")
    print(f"📦 Loading base model + LoRA adapter...\n")

    # Step 1: Load base model
    print("1️⃣ Loading base model: {}".format(CONFIG['base_model_name']))
    model, tokenizer = FastVisionModel.from_pretrained(
        CONFIG['base_model_name'],  # Load base Qwen model first
        load_in_4bit=CONFIG['use_4bit'],
        use_gradient_checkpointing="unsloth",
    )

    # Step 2: Load LoRA adapter from CROHME training
    print(f"2️⃣ Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)  # ⭐ Enable training!

    # Enable gradient for LoRA p
    # The model now has LoRA adapters from CROHME training
    print("\n✅ Model loaded with CROHME LoRA adapters!")
    return model, tokenizer
class AuraSRLitAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device
        self.model, self.tokenizer = load_unsloth_qwen()
        self.model_latex = Pix2Text.from_config(device="cuda", enable_onnx=False)
    def decode_request(self, request):
        if request["Type"] == "Label": 
            image_bytes = bytes.fromhex(request["image_bytes"])
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": request["prompt"]}
                    ]
                }
            ]

            input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
            # --- ép BatchFeature -> dict ---
            if not isinstance(inputs, dict):
                inputs = dict(inputs)
            inputType = request["Type"]
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            inputs = [inputs,inputType]
        else:
            image_bytes = bytes.fromhex(request["image_bytes"])            
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            inputs = [image,request["Type"]]
        return inputs

    def predict(self, inputs):
        print(inputs)
        if inputs[0][1] == "Label":
            with torch.no_grad():
                outputs = self.model.generate(**inputs[0][0], max_new_tokens=256)
            prediction = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            if "assistant\n" in prediction:
                latex_output = prediction.split("assistant\n", 1)[1].strip()
            elif "assistant" in prediction:
                latex_output = prediction.split("assistant", 1)[1].strip()
            else:
                # Fallback: use the whole prediction
                latex_output = prediction.strip('"')
            
            print(f"Extracted LaTeX: {latex_output}")
        else:
            latex_output = self.model_latex.recognize(img=inputs[0][0], return_text=True)

        return {latex_output}  # Return plain text instead of dict        return {prediction}


if __name__ == "__main__":
    api = AuraSRLitAPI()
    server = ls.LitServer(api, max_batch_size=4, timeout=True)
    # Add CORS middleware to allow web app access
    server.app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins (can restrict to specific domains in production)
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
        allow_headers=["*"],  # Allow all headers
    )
    server.run(port=8000)
