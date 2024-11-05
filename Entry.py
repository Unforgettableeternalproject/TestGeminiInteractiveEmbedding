import datetime
from llm_module.gemini_client import GeminiClient
from nlp_module.simpleNLP import SimpleNLP
from llm_module.memory_manager import MemoryManager

def main():
    # Initialize the Gemini client
    project_id = "u-e-p-440612"
    gemini_client = GeminiClient(project_id=project_id)
    memoryManager = MemoryManager()
    
    # Start a chat session
    gemini_client.start_chat()
    
    # Initialize the NLP model
    nlp_model = SimpleNLP()
    
    # Send sample messages and print responses
    try:
        while True:
            message = input("You: ")
            
            if(message is None): continue
            
            if (message == ">check"): 
                memoryManager.check_memory()
                continue
            if (message == ">clear"): 
                memoryManager.clear_memory()
                continue
            if (message == ">exit"): break
            
            label = nlp_model.classify_message(message)
            response = gemini_client.send_labeled_message(message, label)
            print("U.E.P: {}".format(response.text))
    except KeyboardInterrupt:
        print(datetime.time + "\nChat ended.")
        return
    
    print(datetime.time + "Chat ended.")

if __name__ == "__main__":
    main()
