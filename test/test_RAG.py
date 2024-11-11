from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import torch

# 1. Load the sentence transformer model for retrieval
retriever_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Example documents to search from (simulating a document store)
documents = [
    "Python is a programming language known for its simplicity and readability.",
    "RAG (Retrieval-Augmented Generation) is a method that combines retrieval and generation.",
    "The Hugging Face library provides a variety of pre-trained language models.",
    "Transformers have revolutionized the field of natural language processing."
]

# 2. Encode the documents for retrieval
document_embeddings = retriever_model.encode(documents, convert_to_tensor=True)

# 3. Set up the generation model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
generator_model = AutoModelForCausalLM.from_pretrained("gpt2")
generator = pipeline("text-generation", model=generator_model, tokenizer=tokenizer)

def retrieve(query, top_k=2):
    query_embedding = retriever_model.encode(query, convert_to_tensor=True)
    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)
    # Find the top k documents
    top_k_indices = torch.topk(similarities, k=top_k)[1][0]
    retrieved_docs = [documents[idx] for idx in top_k_indices]
    return retrieved_docs

def generate_answer(query):
    # Step 1: Retrieve relevant documents
    retrieved_docs = retrieve(query)
    context = " ".join(retrieved_docs)

    # Step 2: Combine query and context, then generate response
    input_text = f"{context}\nQuestion: {query}\nAnswer:"
    response = generator(input_text, max_length=50, num_return_sequences=1)
    return response[0]["generated_text"]

# Test the RAG pipeline
query = "How does RAG work?"
answer = generate_answer(query)
print("Generated Answer:", answer)
