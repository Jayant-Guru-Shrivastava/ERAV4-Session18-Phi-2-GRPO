MODEL_NAME = "microsoft/phi-2"
DATASET_NAME = "OpenAssistant/oasst1"
MAX_SEQ_LEN = 1024

LR = 2e-5
BATCH_SIZE = 1
GRAD_ACC = 8
MAX_STEPS = 25  # Reduced to 25 for fast assignment submission

OUTPUT_DIR = "./outputs"
HF_REPO_NAME = "phi2-oasst-grpo"
