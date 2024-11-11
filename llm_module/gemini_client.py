import vertexai
from transformers import pipeline, AutoTokenizer
from tts_module.api_caller import TTSClient
from .memory_manager import MemoryManager
from vertexai.generative_models import GenerativeModel
from .config import SYSTEM_INSTRUCTION, GENERATION_CONFIG, SAFETY_SETTINGS

class GeminiClient:
    def __init__(self, project_id, location="asia-east1"):
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel("gemini-1.5-pro-002", system_instruction=[SYSTEM_INSTRUCTION])
        self.memory_manager = MemoryManager()
        self.load_faiss_index()
        self.tts_client = TTSClient()
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    def load_faiss_index(self):
        if self.memory_manager.faiss_index_exists():
            self.memory_manager.load_index()
        else:
            print("FAISS index not found. Creating a new index.")
            self.memory_manager.create_index()

    def chunk_and_summarize_memories(self, interactions, max_tokens=512):
        """Divide interactions into chunks based on token limits and summarize each chunk."""
        summaries = []
        current_chunk, current_tokens = [], 0
        
        for interaction in interactions:
            # Create text for the interaction
            interaction_text = f"User: {interaction['user']} | Response: {interaction['response']}"
            
            # Calculate token count for this interaction
            token_count = len(self.tokenizer.encode(interaction_text))
            
            # Check if adding this interaction exceeds the max token limit for the current chunk
            if current_tokens + token_count > max_tokens:
                # Summarize the current chunk if token limit is reached
                chunk_text = " ".join(current_chunk)
                summary = self.summarizer(chunk_text, max_length=60, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])
                
                # Reset for the next chunk
                current_chunk, current_tokens = [], 0
            
            # Add interaction to the current chunk
            current_chunk.append(interaction_text)
            current_tokens += token_count

        # Summarize any remaining interactions in the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            summary = self.summarizer(chunk_text, max_length=60, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        
        # Combine all chunk summaries into a single context
        return " ".join(summaries)
    
    def start_chat(self):
        self.chat = self.model.start_chat()

    def send_labeled_message(self, message, label):
        # Retrieve relevant past interactions
        past_interactions = self.memory_manager.retrieve_memory(message, top_k=10)  # Retrieve more to ensure enough context
        
        # Summarize retrieved memories using sliding window
        if past_interactions:
            memory_summary = self.chunk_and_summarize_memories(past_interactions, max_tokens=512)
        else:
            memory_summary = "No relevant past interactions found."
        
        # Customize system instruction based on message type
        if label == "chat":
            system_instruction = SYSTEM_INSTRUCTION + "\n\nYou are having a friendly chat."
        elif label == "command":
            system_instruction = SYSTEM_INSTRUCTION + "\n\nPlease assist with the task as requested."
        elif label == "non-sense":
            system_instruction = SYSTEM_INSTRUCTION + "\n\nRespond playfully, as the input doesn't make sense."
        else:
            system_instruction = SYSTEM_INSTRUCTION

        # Combine the system instruction and memory summary into a single prompt
        full_prompt = f"{system_instruction}\n\nRelevant Memory Summary:\n{memory_summary}\n\nUser: {message}"
        
        # Generate response with summarized memory context
        model_with_memory = GenerativeModel(
            "gemini-1.5-pro-002",
            system_instruction=[full_prompt]
        )
        chat_with_memory = model_with_memory.start_chat()
        
        response = chat_with_memory.send_message(
            [message],
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS
        )
        
        # Synthesize the response and play it
        try:
            self.tts_client.synthesize_and_play(response.text)
        except Exception as e:
            print("< TTS service not available. >")

        # Add the current message and response to memory
        self.memory_manager.add_memory(message, {"user": message, "response": response.text})
        return response
