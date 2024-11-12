# TestGeminiInteractiveEmbedding

## Overview

This project is a simple chatbot implementation that integrates speech-to-text (STT) and text-to-speech (TTS) functionalities. It leverages various components to provide real-time interaction with users through voice commands and responses.

## Features

- **Real-Time Speech-to-Text (STT)**: Convert spoken language into text using a microphone in real-time.
- **Natural Language Processing (NLP)**: Classify the transcribed text using a pre-trained DistilBERT model.
- **Text-to-Speech (TTS)**: Convert the chatbot's text responses back into speech for audio playback.
- **Memory Management**: Store and retrieve past interactions to provide context-aware responses.

## Project Structure

```graphql
¢x SimpleChatbot-STT-TTS/ 
¢u¢w¢w llm_module/
¢x   ¢u¢w¢w config.py
¢x   ¢u¢w¢w memory_manager.py 
¢x   ¢|¢w¢w gemini_client.py 
¢x
¢u¢w¢w nlp_module/
¢x   ¢|¢w¢w simpleNLP.py 
¢x
¢u¢w¢w stt_module/ 
¢x   ¢|¢w¢w stt.py 
¢x
¢u¢w¢w tts_module/
¢x   ¢|¢w¢w api_caller.py 
¢x
¢u¢w¢w Entry.py 
¢|¢w¢w README.md
```

## Components

### Speech-to-Text (STT)

The `stt.py` script handles real-time speech-to-text conversion using the `speech_recognition` library. It captures audio from a microphone, transcribes it to text, and places the text in a queue for NLP processing.

### NLP Processing

The `simpleNLP.py` script uses a pre-trained DistilBERT model to classify the transcribed text into one of three categories: command, chat, or non-sense. It integrates with the STT component to process real-time transcriptions.

### Text-to-Speech (TTS)

The `api_caller.py` script sends text to a TTS API and plays the synthesized audio. It splits long texts into manageable chunks and handles the audio playback.

### Memory Management

The `memory_manager.py` script manages the storage and retrieval of past interactions using a FAISS index. It embeds text interactions, stores them, and retrieves relevant memories to provide context-aware responses.

### Chatbot Client

The `gemini_client.py` script integrates all components to create a functional chatbot. It initializes the necessary models, manages the chat sessions, retrieves past interactions, and generates responses using the Gemini model.

## To be implemented

- Real-time interaction with the chatbot.
- Integration of all components for a seamless user experience.
---

## Disclaimer

This project is **not** intended for public use.

The model within this project is not public, and the server IP and port are fake.

For most part, it is for demonstration purposes only.
