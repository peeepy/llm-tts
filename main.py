import sys
sys.path.append('./imports')
from KoboldAILLMApi import llm
from langchain_variables import prompt, memory
from langchain import LLMChain

import re
import json
import requests
import time

from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

import speech_recognition as sr
from scipy.io.wavfile import write
import pygame

with open('config.json') as file:
    config_data = json.load(file)

VOICE = config_data['VOICE']
TTS_KEY = config_data['TTS_KEY']
char = config_data['char']

s = requests.Session()

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=False,
    memory=memory,
)

"""fs = 44100
sd.default.samplerate = fs
sd.default.channels = 2"""

# instantiate pipeline
pipeline = FlaxWhisperPipline("openai/whisper-medium", dtype=jnp.float16)

r = sr.Recognizer()
r.dynamic_energy_threshold = False

while True:
        print(char, " is listening...")
        with sr.Microphone() as source:
            audio = r.listen(source)
        print("Finished recording.")
        start = time.time()
        if audio is not None:
            with open("output.wav", "wb") as f:
                f.write(audio.get_wav_data())
        text = pipeline("output.wav")
        predicted_text = text['text']
        if predicted_text:
            print("You said: " + predicted_text)
            response = llm_chain.predict(human_input=predicted_text)
            stripped_response = re.sub(r'\*[^\*]*?(\*|$)', '', response).strip()
            if stripped_response:
                print(char, " is thinking...")

                url = "https://api.elevenlabs.io/v1/text-to-speech/" + VOICE

                headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": TTS_KEY
                }

                data = {
                    "text": stripped_response,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                    }
                    }

                speech = s.post(url, json=data, headers=headers)
                with open('output.mp3', 'wb+') as f:
                    f.write(speech.content)

                if pygame.mixer.get_init() is None:
                    pygame.mixer.init()
                pygame.mixer.music.load('output.mp3')
                end = time.time()
                time_taken = end - start
                print("Speech generated. Time taken: {0:.2f} seconds.".format(time_taken))
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.delay(100)
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            else:
                print(char, " only used asterisks, or an error occurred. Nothing to generate.")
