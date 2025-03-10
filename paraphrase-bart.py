from transformers import BartForConditionalGeneration, BartTokenizer

# Load BART model
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def paraphrase_text(text, max_length=100):
    """
    Paraphrases input text using BART transformer.
    """
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)

    # Generate paraphrased text
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        min_length=20,
        num_return_sequences=1,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        repetition_penalty=2.0
    )

    # Decode output
    paraphrased_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return paraphrased_text

# Example usage
text = "AI-generated content is becoming more common, making detection crucial."
paraphrased = paraphrase_text(text)
print("Original:", text)
print("Paraphrased:", paraphrased)
