import torch
import logging
import nltk
from nltk.tokenize import sent_tokenize
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor  # For parallelization

# Logger setup
logger = logging.getLogger(__name__)

MODEL_NAME = "tuner007/pegasus_paraphrase"
MODEL_PATH = Path(f"/opt/ai-models/{MODEL_NAME}")

# Ensure NLTK tokenizer is available
nltk.download("punkt")

class ParaphraseService:
    def __init__(self):
        """
        Initializes the Paraphrase Service by checking and loading the model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer, self.model = self.check_and_load_model()
        self.model.to(self.device)  # Move model to GPU if available

    def check_and_load_model(self):
        """
        Loads Pegasus paraphrase model, downloading it if not available locally.
        """
        try:
            if not MODEL_PATH.exists():
                logger.info(f"Model {MODEL_NAME} not found locally. Downloading...")
                tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
                model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME)
                MODEL_PATH.mkdir(parents=True, exist_ok=True)
                tokenizer.save_pretrained(MODEL_PATH)
                model.save_pretrained(MODEL_PATH)
                logger.info(f"Model {MODEL_NAME} downloaded and saved to {MODEL_PATH}.")
            else:
                logger.info(f"Model {MODEL_NAME} found locally. Loading from disk...")
                tokenizer = PegasusTokenizer.from_pretrained(MODEL_PATH)
                model = PegasusForConditionalGeneration.from_pretrained(MODEL_PATH)

            return tokenizer, model
        except Exception as e:
            logger.error(f"Error loading model {MODEL_NAME}: {str(e)}")
            raise e

    def paraphrase_sentence(self, sentence, max_length=100):
        """Paraphrases a single sentence."""
        input_ids = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=512
        ).input_ids.to(self.device)

        output_ids = self.model.generate(
            input_ids,
            max_length=max_length,
            min_length=20,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.8,  # Lower temperature for more stable output
            repetition_penalty=1.5  # Lower penalty for slight diversity
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def paraphrase(self, text, max_length=100):
        """
        Paraphrases the input text using Pegasus.
        Uses multi-threading for better performance.
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty.")

        sentences = sent_tokenize(text)  # Split into sentences

        with ThreadPoolExecutor(max_workers=4) as executor:
            paraphrased_sentences = list(executor.map(lambda s: self.paraphrase_sentence(s, max_length), sentences))

        return " ".join(paraphrased_sentences)  # Reconstruct full paragraph


if __name__ == "__main__":
    paraphrase_service = ParaphraseService()

    # Example usage
    text = """India is the seventh-largest country in the world by land area and the second-most populous, 
    with over 1.4 billion people. It is a land of immense cultural diversity, with over 2,000 ethnic groups 
    and hundreds of languages spoken across its 28 states and 8 Union Territories. India is the birthplace 
    of major religions such as Hinduism, Buddhism, Jainism, and Sikhism, which have influenced cultures worldwide. 
    The country is also home to magnificent historical monuments, including the Taj Mahal, one of the Seven Wonders of the World."""

    paraphrased = paraphrase_service.paraphrase(text)
    print("Original:\n", text)
    print("\nParaphrased:\n", paraphrased)
