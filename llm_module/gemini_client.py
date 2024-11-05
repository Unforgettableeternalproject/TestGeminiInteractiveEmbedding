import vertexai
from transformers import pipeline
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
        
    def load_faiss_index(self):
        if self.memory_manager.faiss_index_exists():
            self.memory_manager.load_index()  # Load the FAISS index if it exists
        else:
            print("FAISS index not found. Creating a new index.")
            self.memory_manager.create_index()  # Create a new FAISS index
            
    def summarize_context(self, interactions):
        """Summarize a list of retrieved interactions."""
        context_text = "Chat context as below:" + "\n".join(
            f"User said {interaction['user']}\n and you responded with {interaction['response']}\n"
            for interaction in interactions
        )
        
        # Generate a summary with the summarizer pipeline
        summary = self.summarizer(context_text, max_length=60, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    
    def start_chat(self):
        self.chat = self.model.start_chat()

    def send_labeled_message(self, message, label):
        # Retrieve relevant past interactions
        past_interactions = self.memory_manager.retrieve_memory(message, top_k=5)
        
        # Summarize the retrieved interactions
        if past_interactions:
            memory_summary = self.summarize_context(past_interactions)
        else:
            memory_summary = "No relevant past interactions found."
            
        print("\nNLP results: ", label, '\n')

        # Customize system instruction based on message type
        if label == "chat":
            system_instruction = SYSTEM_INSTRUCTION + "\n\nYou are having a friendly chat."
        elif label == "command":
            system_instruction = SYSTEM_INSTRUCTION + "\n\nPlease assist with the task as requested."
        elif label == "non-sense":
            system_instruction = SYSTEM_INSTRUCTION + "\n\nRespond playfully, as the input doesn't make sense."
        else:
            system_instruction = SYSTEM_INSTRUCTION
            
        print("Past memory summary: ", memory_summary, '\n')

        # Create the prompt with memory summary
        full_prompt = f"{system_instruction}\n\nRelevant Memory Summary:\n{memory_summary}\n\nUser: {message}"
        
        # Send the message with the summarized memory context
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
        
        self.tts_client.synthesize_and_play(response.text)

        # Add the current message and response to memory
        self.memory_manager.add_memory(message, {"user": message, "response": response.text})
        return response
