import gradio as gr
from transformers import pipeline
import torch

# YOUR CORRECT USERNAME
MODEL_ID = "jayantgurushrivastava/phi2-oasst-grpo"

load_error = "Unknown error"
pipe = None

try:
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        trust_remote_code=True
    )
except Exception as e:
    load_error = str(e)
    print(f"Error loading model: {e}")

def chat(prompt):
    if pipe is None:
        return f"SYSTEM ERROR: The model failed to load.\n\nReason: {load_error}\n\n(Check requirements.txt or model status)"
    
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n" if "###" not in prompt else prompt

    try:
        outputs = pipe(formatted_prompt, max_new_tokens=200)
        return outputs[0]["generated_text"].split("### Response:")[-1].strip()
    except Exception as e:
        return f"Generation Error: {e}"

iface = gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="Phi-2 OpenAssistant GRPO",
    description=f"Model: {MODEL_ID}"
)

iface.launch()
