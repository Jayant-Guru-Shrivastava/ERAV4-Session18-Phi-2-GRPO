import gradio as gr
from transformers import pipeline

# Replace with your actual model path on Hugging Face Hub
MODEL_ID = "YOUR_USERNAME/phi2-oasst-grpo"

try:
    pipe = pipeline(
        "text-generation",
        model=MODEL_ID,
        trust_remote_code=True
    )
except Exception as e:
    pipe = None
    print(f"Error loading model: {e}")

def chat(prompt):
    if pipe is None:
        return "Error: Model not loaded. Did you train and push the model to HF?"
    
    return pipe(prompt, max_new_tokens=200)[0]["generated_text"]

iface = gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="Phi-2 OpenAssistant GRPO (QLoRA)",
    description="A fine-tuned Phi-2 model using GRPO and QLoRA on OpenAssistant dataset."
)

if __name__ == "__main__":
    iface.launch()
