import torch
import numpy as np
from PIL import Image
import gradio as gr
from pathlib import Path
from torchvision import transforms

from model import VisionGPT2Model
from transformers import GPT2TokenizerFast
from config import config

# ZeroGPU support: on Hugging Face ZeroGPU hardware, `spaces.GPU` attaches a GPU
# for the duration of each call. Elsewhere (local / CPU) it is a no-op.
try:
    import spaces
    gpu = spaces.GPU
except ImportError:
    def gpu(fn):
        return fn

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
def load_model():
    model = VisionGPT2Model(config)
    state_dict = torch.load("captioner.pt", map_location="cpu", weights_only=True)
    # Some checkpoints omit the causal attention masks (`*.attn.mask`); those are
    # fixed buffers the model already builds in __init__, so they may be absent.
    # Anything else missing or unexpected is a real mismatch.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if not k.endswith(".attn.mask")]
    if missing or unexpected:
        raise RuntimeError(f"captioner.pt does not match the model: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model.to(device)

model = load_model()
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Inference function
@gpu
def generate_caption(image):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    sequence = (torch.ones(1, 1).long() * tokenizer.bos_token_id).to(device)
    
    with torch.no_grad():
        caption_ids = model.generate(
            image_tensor,
            sequence,
            max_tokens=50,
            temperature=1.0,
            deterministic=True
        )
    
    caption = tokenizer.decode(caption_ids.numpy(), skip_special_tokens=True)
    return caption

# Gradio App
interface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Caption Generator (ViT + GPT-2)",
    description="Upload an image and generate a caption using a hybrid Vision Transformer + GPT-2 model."
)

interface.launch()
