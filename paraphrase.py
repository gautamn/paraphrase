import torch
import logging
import nltk
import onnxruntime as ort
from nltk.tokenize import sent_tokenize
from transformers import PegasusTokenizer
from pathlib import Path
from optimum.onnxruntime import ORTModelForSeq2SeqLM


# Logger setup
logger = logging.getLogger(__name__)

MODEL_NAME = "tuner007/pegasus_paraphrase"
MODEL_PATH = Path(f"/opt/ai-models/{MODEL_NAME}")
ONNX_MODEL_PATH = MODEL_PATH / "pegasus.onnx"

# Ensure NLTK tokenizer is available
nltk.download("punkt")

class ParaphraseService:
    def __init__(self):
        """
        Initializes the Paraphrase Service by checking and loading the model.
        Uses ONNX Runtime for optimized inference.
        """
        self.device = "cpu"  # Force CPU usage
        self.tokenizer, self.session = self.check_and_load_model()

    from pathlib import Path
    import logging
    from transformers import PegasusTokenizer
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    import onnxruntime as ort  # Import ONNX Runtime

    logger = logging.getLogger(__name__)

    MODEL_NAME = "tuner007/pegasus_paraphrase"
    MODEL_PATH = Path(f"/opt/ai-models/{MODEL_NAME}")

    def check_and_load_model(self):
        """
        Loads the Pegasus paraphrase model and tokenizer, exporting to ONNX if needed.
        """
        try:
            if not MODEL_PATH.exists():
                logger.info(f"Model {MODEL_NAME} not found locally. Downloading...")

                tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
                model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_NAME, export=True)

                # Save model and tokenizer locally
                MODEL_PATH.mkdir(parents=True, exist_ok=True)
                tokenizer.save_pretrained(MODEL_PATH)
                model.save_pretrained(MODEL_PATH)

                logger.info(f"Model {MODEL_NAME} downloaded and saved to {MODEL_PATH}.")
            else:
                logger.info(f"Model {MODEL_NAME} found locally. Loading from disk...")
                tokenizer = PegasusTokenizer.from_pretrained(MODEL_PATH)
                model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_PATH, export=False)

            # ✅ Debugging logs
            logger.info("Tokenizer and model loaded successfully!")

            # ✅ ONNX Session Configuration
            onnx_model_path = MODEL_PATH / "decoder_model.onnx"  # Ensure correct ONNX file
            if not onnx_model_path.exists():
                raise FileNotFoundError(f"ONNX model file not found at {onnx_model_path}")

            session = ort.InferenceSession(str(onnx_model_path), providers=["CPUExecutionProvider"])

            logger.info("ONNX Inference Session created successfully!")

            return tokenizer, session

        except Exception as e:
            logger.error(f"Error loading model {MODEL_NAME}: {str(e)}")
            raise e

    def paraphrase(self, text, max_length=100):
        """
        Paraphrases the input text using ONNX Runtime.
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty.")

        sentences = sent_tokenize(text)  # Split into sentences
        paraphrased_sentences = []

        for sentence in sentences:
            # Tokenize input properly
            inputs = self.tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                padding="longest",
                max_length=512
            )

            input_ids = inputs.input_ids  # Tensor format
            input_ids_np = input_ids.numpy().astype("int64")  # Convert to int64 for ONNX

            # ✅ Validate input indices before sending to ONNX
            if input_ids.max() >= self.tokenizer.vocab_size:
                logger.error(
                    f"Token index out of range! Max index: {input_ids.max()}, Vocab size: {self.tokenizer.vocab_size}")
                continue  # Skip this sentence

            # ONNX inference
            outputs = self.session.run(None, {"input_ids": input_ids_np})
            output_ids = torch.tensor(outputs[0])  # Convert back to tensor for decoding

            paraphrased_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            paraphrased_sentences.append(paraphrased_text)

        logger.debug(f"Paraphrased Sentences: {paraphrased_sentences}")
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
