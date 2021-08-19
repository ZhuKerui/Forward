import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import pdb

# Initialize model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever = retriever)

# Initialize data
inputs = tokenizer(["How many people live in Paris?", 'how old are you?'], return_tensors="pt", padding=True, truncation=True)
with tokenizer.as_target_tokenizer():
   targets = tokenizer(["In Paris, there are 10 million people.", 'I am 22 years old.'], return_tensors="pt", padding=True, truncation=True)
input_ids = inputs["input_ids"]
labels = targets["input_ids"]

pdb.set_trace()
# 1. Step by step code
# 1.1. Encode
question_hidden_states = model.question_encoder(input_ids)[0]

# 1.2. Retrieve
docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)).squeeze(1)

# 1.3. Train
outputs = model(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores, labels=labels)

# 1.4. Forward to generator
outputs = model(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores, decoder_input_ids=labels)

# 2. Direct code
inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
outputs = model(input_ids=inputs["input_ids"])