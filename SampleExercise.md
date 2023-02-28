Sentence Completion with a Pre-Trained Transformer
==================================================

In this exercise you will use a "prompt" sentence to probe the top 4 probable words to complete that sentence.  You'll use the distilbert model (a transformer trained on English Wikipedia and BookCorpus datasets).  The prompt ends with a mask token "[MASK]". You can use the index of this mask to identify the most probable words that would replace the mask, according to the model.

Both the distilbert `tokenizer` and `model` are preloaded for you.

Instructions
------------

* Using the distilbert `tokenizer` to tokenize the prompt and assign the result to `tokens`

* Use the `torch.where` function to find the position of the mask in the tokenized prompt and assign it to `mask_token_index`

* Use the model to get the logits for all tokens and pull out those for the mask and assign them to `mask_token_logits`

* Finally use `torch.topk` to select the four highest logits (assign them to `most_probable_tokens`) and then print out the corresponding words

Solution Code
-------------

```py
import torch

prompt = "I laugh when [MASK]."

# use the `tokenizer` to tokenize the prompt
tokens = tokenizer(prompt, return_tensors="pt")

# Use `torch.where` to find the position of [MASK]
mask_token_index = torch.where(tokens["input_ids"] == tokenizer.mask_token_id)[1]

# get the logits from the model and pull out those for the [MASK]
mask_token_logits = model(**tokens).logits[0, mask_token_index, :]

# Choose the four highest logit [MASK]
most_probable_tokens = torch.topk(mask_token_logits, 4, dim=1).indices[0].tolist()

# print out the most probable words
print(prompt)
for token in most_probable_tokens:
    word = tokenizer.decode([token])
    print(f" -- {word}")
```

Expected output
---------------

```
I laugh when [MASK].
 -- embarrassed
 -- angry
 -- necessary
 -- nervous
```
