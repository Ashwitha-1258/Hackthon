# ============================================================
# ECHOVERSE – GRANITE 3.3 2B AUDIOBOOK CREATOR (FIXED VERSION)
# ============================================================

!pip install --upgrade transformers accelerate sentencepiece safetensors --quiet
!pip install torch torchaudio torchvision --quiet
!pip install gradio pillow --quiet
!pip install bark==0.1.5 scipy --quiet

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gradio as gr
import numpy as np
from bark import SAMPLE_RATE, generate_audio
import scipy.io.wavfile as wavfile
from PIL import Image

# ============================================================
# LOAD GRANITE 3.3 2B INSTRUCT (FIXED TOKENIZER)
# ============================================================

MODEL_NAME = "ibm-granite/granite-3.3-2b-instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=False    # <-- FIXED
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Manual generation function
def granite_generate(messages):
    prompt = ""
    for m in messages:
        prompt += f"{m['role'].upper()}: {m['content']}\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# ============================================================
# IMAGE CAPTIONING MODEL (BLIP)
# ============================================================

vision = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

# ============================================================
# TEXT REWRITING FUNCTION
# ============================================================

def rewrite_text(text, tone, language):
    messages = [
        {
            "role": "user",
            "content": f"Rewrite the text in a {tone} tone. Output language: {language}.\n\n{text}"
        }
    ]
    return granite_generate(messages)

# ============================================================
# AUDIO GENERATION FUNCTION
# ============================================================

def create_audio(text, voice, speed, pitch):
    audio = generate_audio(text, history_prompt=voice)

    # Adjust speed
    indices = np.arange(0, len(audio), speed)
    indices = indices[indices < len(audio)]
    audio = np.interp(indices, np.arange(len(audio)), audio)

    # Adjust pitch
    audio = audio * pitch

    filename = "audiobook.wav"
    wavfile.write(filename, SAMPLE_RATE, audio.astype(np.float32))
    return filename

# ============================================================
# IMAGE DESCRIPTION
# ============================================================

def describe_image(img):
    if img is None:
        return "No image uploaded."
    return vision(img)[0]["generated_text"]

# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================

def process(text, tone, lang, voice, speed, pitch, img):
    rewritten = rewrite_text(text, tone, lang)
    audio = create_audio(rewritten, voice, float(speed), float(pitch))
    caption = describe_image(img)
    return rewritten, audio, caption

# ============================================================
# GRADIO UI
# ============================================================

ui = gr.Interface(
    fn=process,
    title="EchoVerse – IBM Granite 3.3 2B Audiobook Creator",
    inputs=[
        gr.Textbox(lines=6, label="Text"),
        gr.Radio(["Neutral", "Inspiring", "Suspenseful"], label="Tone"),
        gr.Dropdown(["English", "Hindi", "Tamil", "Spanish", "French", "German", "Chinese"], label="Language"),
        gr.Dropdown(["v2/en_speaker_1", "v2/en_speaker_6", "v2/en_speaker_9", "v2/hi_speaker_7"], label="Voice"),
        gr.Slider(0.5, 2.0, 1.0, label="Speed"),
        gr.Slider(0.5, 2.0, 1.0, label="Pitch"),
        gr.Image(label="Image (Optional)")
    ],
    outputs=[
        gr.Textbox(label="Rewritten Text"),
        gr.Audio(label="Generated Audiobook"),
        gr.Textbox(label="Image Description")
    ]
)

ui.launch(debug=True)
