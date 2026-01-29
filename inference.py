import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 1. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load pretrained model (for now)
MODEL_NAME = "google/flan-t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()


# 3. Inference function
def humanize_text(text):
    input_text = "Rewrite the following text in natural English:\n\n " + text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            min_length=30,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 4. Local test
if __name__ == "__main__":
    text = "Artificial intelligence is transforming the world rapidly."
    print("Original:", text)
    print("Humanized:", humanize_text(text))
    print(tokenizer.name_or_path)
    print(model.name_or_path)
