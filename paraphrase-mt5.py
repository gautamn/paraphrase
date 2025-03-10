from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import torch
import nltk
from nltk.tokenize import sent_tokenize
from pathlib import Path
import logging

# Logger setup
logger = logging.getLogger(__name__)

MODEL_NAME = "google/mt5-base"
MODEL_PATH = Path(f"/opt/ai-models/{MODEL_NAME}")

# Ensure NLTK tokenizer is available
nltk.download("punkt")

class ParaphraseServiceMT5:
    def __init__(self):
        """Initializes and loads the MT5 model and tokenizer."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer, self.model = self.check_and_load_model()
        self.model.to(self.device)

    def check_and_load_model(self):
        """Loads the MT5 model, downloading it if necessary."""
        try:
            if not MODEL_PATH.exists():
                logger.info(f"Model {MODEL_NAME} not found locally. Downloading...")

                tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
                model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

                # Save locally
                MODEL_PATH.mkdir(parents=True, exist_ok=True)
                tokenizer.save_pretrained(MODEL_PATH)
                model.save_pretrained(MODEL_PATH)

                logger.info(f"Model {MODEL_NAME} downloaded and saved to {MODEL_PATH}.")
            else:
                logger.info(f"Model {MODEL_NAME} found locally. Loading from disk...")
                tokenizer = MT5Tokenizer.from_pretrained(MODEL_PATH)
                model = MT5ForConditionalGeneration.from_pretrained(MODEL_PATH)

            return tokenizer, model

        except Exception as e:
            logger.error(f"Error loading model {MODEL_NAME}: {str(e)}")
            raise e

    def paraphrase(self, text, max_length=100):
        """Paraphrases text using the MT5 model."""
        if not text.strip():
            raise ValueError("Input text cannot be empty.")

        sentences = sent_tokenize(text)
        paraphrased_sentences = []

        for sentence in sentences:
            input_text = f"paraphrase: {sentence} </s>"
            input_ids = self.tokenizer(
                input_text, return_tensors="pt", max_length=512, truncation=True
            ).input_ids.to(self.device)

            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                min_length=20,
                num_return_sequences=1,
                do_sample=True,
                temperature=1.0,
                repetition_penalty=2.0,
            )

            paraphrased_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            paraphrased_sentences.append(paraphrased_text)

        logger.debug(f"Paraphrased Sentences: {paraphrased_sentences}")
        return " ".join(paraphrased_sentences)


if __name__ == "__main__":
    paraphrase_service = ParaphraseServiceMT5()

    # Example usage
    text = """India is the seventh-largest country in the world by land area and the second-most populous, 
    with over 1.4 billion people. It is a land of immense cultural diversity, with over 2,000 ethnic groups 
    and hundreds of languages spoken across its 28 states and 8 Union Territories. India is the birthplace 
    of major religions such as Hinduism, Buddhism, Jainism, and Sikhism, which have influenced cultures worldwide. 
    The country is also home to magnificent historical monuments, including the Taj Mahal, one of the Seven Wonders of the World."""

    paraphrased = paraphrase_service.paraphrase(text)
    print("Original:\n", text)
    print("\nParaphrased:\n", paraphrased)