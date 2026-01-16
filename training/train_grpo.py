import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_from_disk
from trl import GRPOTrainer, GRPOConfig
from reward import length_reward
import config

# QLoRA Config
bnb = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_NAME,
    quantization_config=bnb,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

lora = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora)

dataset = load_from_disk("./processed_oasst")

training_args = GRPOConfig(
    output_dir=config.OUTPUT_DIR,
    learning_rate=config.LR,
    per_device_train_batch_size=config.BATCH_SIZE,
    gradient_accumulation_steps=config.GRAD_ACC,
    max_steps=config.MAX_STEPS,
    num_generations=2, # Reduce from default 8 to 2 for speed
    logging_steps=5,
    push_to_hub=True,
    hub_model_id=config.HF_REPO_NAME,
    max_prompt_length=1024,
    max_completion_length=512,
    gradient_checkpointing=True,
    per_device_eval_batch_size=1,
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[length_reward]
)

# Enable gradient checkpointing for the model explicitly
model.gradient_checkpointing_enable()

trainer.train()
trainer.model.merge_and_unload().push_to_hub(config.HF_REPO_NAME)
tokenizer.push_to_hub(config.HF_REPO_NAME)
