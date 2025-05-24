from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Function to load the pre-fine-tuned model from Hugging Face
def load_model(model_name="philschmid/flan-t5-base-samsum"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Function to generate a response from the model
def generate_response(input_text, model_name="philschmid/flan-t5-base-samsum"):
    tokenizer, model = load_model(model_name)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
