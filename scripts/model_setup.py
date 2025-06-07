from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization")

def download_and_save_model():
    """
    Downloads and saves the model to the local directory.
    """
    model_name = "Falconsai/text_summarization"
    save_dir = "models/summarization"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Downloading model from {model_name} to {save_dir}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer saved to {save_dir}")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

    if __name__ == "__main__":
        download_and_save_model()
