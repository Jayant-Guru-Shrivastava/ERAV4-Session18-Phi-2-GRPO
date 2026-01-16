from transformers import pipeline

def main():
    # Replace 'YOUR_USERNAME/phi2-oasst-grpo' with your actual HF model ID after training
    model_id = "YOUR_USERNAME/phi2-oasst-grpo"
    
    print(f"Loading model from {model_id}...")
    try:
        pipe = pipeline(
            "text-generation",
            model=model_id,
            trust_remote_code=True
        )
        
        prompt = "Explain transformers simply"
        print(f"Generating response for: '{prompt}'")
        
        result = pipe(prompt, max_new_tokens=150)
        print("\nGenerated Text:")
        print(result[0]["generated_text"])
        
    except Exception as e:
        print(f"Error loading model or generating text: {e}")
        print("Note: Make sure you have trained the model and pushed it to Hugging Face Hub first.")

if __name__ == "__main__":
    main()
