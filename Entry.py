from calendar import c
import datetime
from llm_module.gemini_client import GeminiClient
from nlp_module.simpleNLP import SimpleNLP
from llm_module.memory_manager import MemoryManager
from stt_module.stt import STT

def main():
    # Initialize the Gemini client
    project_id = "u-e-p-440612"
    gemini_client = GeminiClient(project_id=project_id)
    memoryManager = MemoryManager()
    
    # Start a chat session
    gemini_client.start_chat()
    
    # Initialize the NLP model
    nlp_model = SimpleNLP()
    
    # Initialize the STT module
    stt_process = STT()
    
    # Send sample messages and print responses
    try:
        while True:
            message = input("You: ")
            
            if(message is None): continue
            if message.startswith(">"):
                match(message[1:]):
                    case "listen":
                        message = stt_process.onetime_speech_recognize()
                    case "check":
                        memoryManager.check_memory()
                        continue
                    case "clear":
                        memoryManager.clear_memory()
                        continue
                    case "exit":
                        break
                    case _:
                        print("Unknown command.")
                        continue
            
            label = nlp_model.classify_message(message)
            response = gemini_client.send_labeled_message(message, label)
            print("U.E.P: {}".format(response.text))
    except KeyboardInterrupt:
        print(datetime.time + "\nChat ended.")
        return
    
    print(datetime.time + "Chat ended.")

if __name__ == "__main__":
    main()
