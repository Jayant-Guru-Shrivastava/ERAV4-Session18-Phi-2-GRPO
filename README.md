# Phi-2 GRPO Fine-Tuning (Assignment Submission)

This repository contains everything you need to fine-tune Phi-2 using GRPO and QLoRA, and deploy it to Hugging Face Spaces.

**Current Mode:** ⚡ Speed Run (100 Steps) - Takes ~20 mins on T4 GPU.

---

## 🏃‍♂️ Step-by-Step Guide: How to Train on Google Colab

### 1. Requirements
- A Google account (for Colab)
- A Hugging Face account (with a Write Access Token)

### 2. Prepare the Files
1.  **Zip** this entire folder (`phi2-oasst-grpo`).
2.  Go to [Google Colab](https://colab.research.google.com/).
3.  Click **File > Upload Notebook** (or just create a new notebook).
4.  **Important:** Go to **Runtime > Change runtime type** and select **T4 GPU**.

### 3. Upload & Unzip
In the Colab sidebar (Files icon), upload your `phi2-oasst-grpo.zip`.
Run this cell to unzip it:
```python
!unzip phi2-oasst-grpo.zip
%cd phi2-oasst-grpo
```

### 4. Install Dependencies
Run this cell:
```python
!pip install -r requirements.txt
```

### 5. Login to Hugging Face
You need to authenticate to upload your trained model.
```python
from huggingface_hub import notebook_login
notebook_login()
```
*Paste your Write Access Token when prompted.*

### 6. Prepare Data
Process the dataset:
```python
!python training/prepare_data.py
```

### 7. Run Training 🚀
Start the GRPO training. This is configured to run for 100 steps (approx. 20 mins).
```python
!python training/train_grpo.py
```
*Wait for it to finish. It will automatically push the model to your Hugging Face account.*

---

## 🌐 How to Deploy to Hugging Face Spaces

Once training is finished, your model is online! Now create the app.

1.  Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2.  **Name:** `phi2-grpo-demo` (or similar).
3.  **SDK:** Select **Gradio**.
4.  **Hardware:** CPU Basic (Free) is fine.
5.  Click **Create Space**.

**Adding the App Code:**
1.  Go to the **Files** tab of your new Space.
2.  Click **Add file > Upload files**.
3.  Drag and drop the files from the `space/` folder (`app.py` and `requirements.txt`).
4.  **Before Committing:** Click on `app.py` to edit it.
5.  Change line 5:
    ```python
    MODEL_ID = "YOUR_USERNAME/phi2-oasst-grpo"
    ```
    *(Replace `YOUR_USERNAME` with your actual HF username)*.
6.  Click **Commit changes**.

Your app will build and be ready in a few minutes! 
