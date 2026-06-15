# py -3.11 -m venv gemma311.venv
# .\gemma311.venv\Scripts\Activate.ps1
# python -m pip install --upgrade pip
# pip install -U transformers accelerate sentencepiece protobuf huggingface_hub pillow
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E2B-it"

processor = AutoProcessor.from_pretrained(MODEL_ID)

# Loading model. This may take a few minutes the first time 
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto"
)

print("Type 'exit' to stop.\n")

messages = [
    {
        "role": "system",
        "content": (
            "You are a concise assistant helping PhD students in psychological science "
            "understand Python and data science."
        )
    }
]


def extract_text(response):
    """
    Gemma 4 processors can parse the structured response.
    This fallback keeps the demo robust if the installed transformers version
    returns a slightly different object.
    """
    try:
        parsed = processor.parse_response(response)
        if isinstance(parsed, str):
            return parsed
        if isinstance(parsed, dict):
            return parsed.get("text", str(parsed))
        return str(parsed)
    except Exception:
        return response.replace("<eos>", "").strip()


while True:
    user_text = input("You: ")

    if user_text.lower().strip() in {"exit", "quit", "q"}:
        print("Stopping.")
        break

    messages.append({"role": "user", "content": user_text})

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = processor(
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=64
        )

    raw_response = processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=False
    )
    
    print(raw_response)

    messages.append({"role": "assistant", "content": raw_response})
