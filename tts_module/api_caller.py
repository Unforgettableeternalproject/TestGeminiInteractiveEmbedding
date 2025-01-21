import requests
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
import re

class TTSClient:    
    def __init__(self, ip="26.87.187.124", port="5000"):
        # Set up the base URL for the TTS API
        self.url = f"http://{ip}:{port}/synthesize"
        
    def split_text(self, text, max_chars=200):
        """Split text into chunks within the max character limit, based on sentence boundaries."""
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?]) +', text)
        
        # Combine sentences into chunks within max character limit
        chunks, current_chunk = [], ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence

        # Add any remaining text as a final chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def synthesize_and_play(self, text, f0_up_key=7, f0_method="rmvpe", index_rate=0, protect=0.33):
        """Send text to TTS API and play the synthesized audio."""
        
        chunks = self.split_text(text) # Split text into chunks
        
        for i, chunk in enumerate(chunks):
        # Data payload for the TTS request
            data = {
                "text": chunk,
                "f0_up_key": f0_up_key,
                "f0_method": f0_method,
                "index_rate": index_rate,
                "protect": protect
            }

            # Send the POST request to the TTS API
            #print(f"Sending TTS request to {self.url}...")
            response = requests.post(self.url, json=data)

            # Handle the response
            if response.status_code == 200:
                # Load audio from the binary content of the response
                audio = AudioSegment.from_wav(BytesIO(response.content))
                #print("Playing audio...")
                play(audio)  # Play the audio directly
            else:
                # Print error if the synthesis fails
                print(f"Failed to synthesize audio. Status code: {response.status_code}")
                try:
                    print("Error:", response.json().get("error"))
                except ValueError:
                    print("Error: No JSON content in response.")
