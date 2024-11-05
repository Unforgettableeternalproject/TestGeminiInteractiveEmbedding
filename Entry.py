from llm_module.gemini_client import GeminiClient

def main():
    # Initialize the Gemini client
    project_id = "u-e-p-440612"
    gemini_client = GeminiClient(project_id=project_id)
    
    # Start a chat session
    gemini_client.start_chat()
    
    # Send sample messages and print responses
    try:
        while True:
            message = input("You: ")
            response = gemini_client.send_message(message)
            print("U.E.P: {}".format(response.text))
    except KeyboardInterrupt:
        print("Chat ended.")
        return

if __name__ == "__main__":
    main()
