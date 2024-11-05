import vertexai
from vertexai.generative_models import GenerativeModel, Part
from .config import SYSTEM_INSTRUCTION, GENERATION_CONFIG, SAFETY_SETTINGS

class GeminiClient:
    def __init__(self, project_id, location="asia-east1"):
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel("gemini-1.5-pro-002", system_instruction=[SYSTEM_INSTRUCTION])

    def start_chat(self):
        self.chat = self.model.start_chat()
       
    def send_message(self, message):
        return self.chat.send_message(
            [message],
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS
        )
