import torch
import logging
import nltk
from nltk.tokenize import sent_tokenize
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor  # For parallelization
import time

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
    #text = """Artificial Intelligence (AI) is a branch of computer science that enables machines to mimic human intelligence. It involves technologies like **machine learning, deep learning, and natural language processing** to solve complex problems, recognize patterns, and automate tasks. AI is widely used in applications such as **virtual assistants, self-driving cars, healthcare diagnostics, fraud detection, and recommendation systems**. Machine learning allows AI models to improve over time by learning from data, while deep learning uses neural networks to process vast amounts of information. AI-powered chatbots and speech recognition systems, like Siri and Alexa, enhance human-computer interaction. Businesses use AI for **predictive analytics, automation, and personalized marketing**. However, AI also raises concerns about **job displacement, data privacy, and ethical decision-making**. As AI continues to evolve, it holds the potential to revolutionize industries and improve daily life, making it one of the most significant technological advancements of the modern era."""

    text = "Artificial Intelligence (AI) is a branch of computer science that enables machines to mimic human intelligence."
    start_time = time.time()  # Start the timer

    paraphrased = paraphrase_service.paraphrase(text)

    end_time = time.time()  # End the timer

    print("Original:\n", text)
    print("\nParaphrased:\n", paraphrased)
    print(f"\nExecution Time: {end_time - start_time:.4f} seconds")
