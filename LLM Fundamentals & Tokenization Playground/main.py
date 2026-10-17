from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")      #Loading GPT - 2 Tokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")   #Loading GPT - 2 Model

print(model)

# For Multiple Sentences in Multiple Languages
sentences = ["Hello, how are you?", "నమస్కారం, మీరు ఎలా ఉన్నారు?", "नमस्ते, आप कैसे हैं?"]

for text in sentences:

    print("Original Text:", text)

    # Convert text into Token IDs
    token_ids = tokenizer.encode(text)
    print("Token IDs:", token_ids)

    # Internal representation of Token Ids to text for understanding
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    print("Internal Representation:", tokens)

    # Tokens IDs to Actual text
    decoded_text = tokenizer.decode(token_ids)
    print("Decoded Text:", decoded_text)

    # Comparing number of words with number of tokens
    print("Number of Words:", len(text.split()))
    print("Number of Tokens:", len(token_ids))

    print("-" * 100)