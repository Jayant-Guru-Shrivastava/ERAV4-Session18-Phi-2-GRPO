import sys
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM

# REPLACE THIS WITH YOUR USERNAME
USERNAME = "Jayant-Guru-Shrivastava" 
REPO_NAME = "phi2-oasst-grpo"
MODEL_ID = f"{USERNAME}/{REPO_NAME}"

print(f"Checking model: {MODEL_ID}...")

# 1. Check if repo exists
api = HfApi()
try:
    repo_info = api.repo_info(repo_id=MODEL_ID)
    print(f"✅ Repository exists! (Private: {repo_info.private})")
except Exception as e:
    print(f"❌ Repository NOT found: {e}")
    sys.exit(1)

# 2. Check for required files
files = api.list_repo_files(repo_id=MODEL_ID)
required_files = ["config.json", "model.safetensors"] # Basic check
missing = [f for f in required_files if f not in files and not any(rf in f for rf in files)] # loose check
if missing:
    print(f"⚠️ Warning: Some files might be missing: {missing}")
    print(f"Found files: {files}")
else:
    print("✅ Key files found.")

# 3. Try Loading Tokenizer
try:
    print("⏳ Attempting to load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("✅ Tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Tokenizer failed to load: {e}")

# 4. Try Loading Model (Float16 to save memory)
try:
    print("⏳ Attempting to load model (this might take a minute)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, device_map="auto")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model failed to load: {e}")
