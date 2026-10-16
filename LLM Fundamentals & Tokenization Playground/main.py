from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")      #Loading GPT - 2 Tokenizer

text = "Hello, how are you?"

# Convert text into Token IDs
token_ids = tokenizer.encode(text)
print("Token IDs:", token_ids)

# Internal representation of Token Ids to text for understanding
tokens = tokenizer.convert_ids_to_tokens(token_ids)
print("Internal Representation:", tokens)

# Tokens IDs to Actual text
decoded_text = tokenizer.decode(token_ids)
print("Decoded Text:", decoded_text)