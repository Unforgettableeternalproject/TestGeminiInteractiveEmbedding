import sys
import os
from transformers import pipeline, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'llm_module')))

from memory_manager import MemoryManager

# Initialize MemoryManager with a test index
memory_manager = MemoryManager(index_file="test_index")

# Initialize summarizer and tokenizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def chunk_and_summarize_memories(query, interactions, max_tokens=1024):
    """Retrieve relevant memories based on the query, then summarize in chunks."""
    # Retrieve memories relevant to the query
    retrieved_memories = memory_manager.retrieve_memory(query, top_k=10)
    print("Retrieved memories:", retrieved_memories)
    
    # Summarize in chunks
    summaries = []
    current_chunk, current_tokens = [], 0
    
    for interaction in retrieved_memories:
        interaction_text = f"Customer said: {interaction['user']} | Response: {interaction['response']}"
        token_count = len(tokenizer.encode(interaction_text))

        # Check if adding this interaction exceeds the token limit
        if current_tokens + token_count > max_tokens:
            # Summarize current chunk if token limit is reached
            chunk_text = " ".join(current_chunk)
            summary = summarizer(chunk_text, max_length=60, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
            
            # Reset chunk
            current_chunk, current_tokens = [], 0

        # Add interaction to current chunk
        current_chunk.append(interaction_text)
        current_tokens += token_count

    # Summarize any remaining interactions in the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        summary = summarizer(chunk_text, max_length=60, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    # Combine all summaries
    combined_summary = " ".join(summaries)
    return combined_summary

# Example test interactions
example_chats = [
    {"user": "Hello, how are you?", "response": "I'm good, thank you! How can I assist you today?"},
    {"user": "Can you tell me a joke?", "response": "Why don't scientists trust atoms? Because they make up everything!"},
    {"user": "What's the weather like today?", "response": "It's sunny and warm outside."},
    {"user": "Can you help me with my homework?", "response": "Sure, what subject do you need help with?"},
    {"user": "What is the capital of France?", "response": "The capital of France is Paris."},
    {"user": "Can you recommend a good book?", "response": "I recommend 'To Kill a Mockingbird' by Harper Lee."},
    {"user": "How does a computer work?", "response": "A computer processes information using binary data and performs calculations with its CPU."},
    {"user": "What's the latest news?", "response": "I'm sorry, I can't access real-time news at the moment."},
    {"user": "How can I improve my productivity?", "response": "Try breaking tasks into smaller goals and using a time management technique like Pomodoro."},
    {"user": "Tell me something interesting.", "response": "Did you know that honey never spoils? Archaeologists found pots of honey in ancient Egyptian tombs that were over 3,000 years old and still edible."},
    {"user": "What languages can you speak?", "response": "I can understand and respond in several languages including English, Spanish, and French."},
    {"user": "Can you help me with Python programming?", "response": "Sure, I'd be happy to help with Python. What specifically do you need help with?"},
    {"user": "What are some healthy eating tips?", "response": "Focus on whole foods, eat a variety of fruits and vegetables, and avoid processed foods."},
    {"user": "What's the best way to learn a new language?", "response": "Practice daily, immerse yourself in the language, and try language learning apps like Duolingo."},
    {"user": "Do you know any facts about space?", "response": "Yes! One interesting fact is that space is completely silent, as there is no atmosphere to carry sound."},
    {"user": "How do airplanes stay in the air?", "response": "Airplanes stay in the air due to lift created by the wings, which counteracts the force of gravity."},
    {"user": "Can you help me write a resume?", "response": "Certainly! Start with a clear header, list your experience, and focus on achievements."},
    {"user": "What is quantum computing?", "response": "Quantum computing uses quantum bits or qubits and can perform complex calculations much faster than classical computers."},
    {"user": "How does climate change affect the planet?", "response": "Climate change leads to rising temperatures, more extreme weather events, and impacts ecosystems."},
    {"user": "Give me a motivational quote.", "response": "Sure! 'The only way to do great work is to love what you do.' - Steve Jobs"},
    {"user": "What's your favorite movie?", "response": "I don't watch movies, but I know that classics like 'The Godfather' and 'Shawshank Redemption' are highly recommended."},
    {"user": "Can you do basic math?", "response": "Yes, I can help with basic calculations. Just tell me what you need solved."},
    {"user": "What's the meaning of life?", "response": "That's a deep question! Many believe it’s about finding happiness, growth, and connection with others."},
    {"user": "Can you play music?", "response": "I can't play music directly, but I can recommend songs or tell you about popular genres."},
    {"user": "What's the tallest building in the world?", "response": "The tallest building is the Burj Khalifa in Dubai, standing at 828 meters."},
]


# Add example interactions to memory
for chat in example_chats:
    memory_manager.add_memory(chat["user"], chat)

# Test the retrieval and summarization function
query = "Can you"
retrieved_and_summarized_memory = chunk_and_summarize_memories(query, example_chats)

# Print the summarized memory
print("Summarized Memory:")
print(retrieved_and_summarized_memory)
