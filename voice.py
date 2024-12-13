# from transformers import pipeline

# pipe = pipeline("text-to-speech", model="microsoft/speech5_tts")

# def convert_to_speech(text, voice_name):
#     audio = pipe(text, voice=voice_name)
#     return audio
# def save_audio(audio, filename):
#     audio.save(filename)

# text = "Hello, I'm learning about natural language processing."
# voice_name = "v2/en_speaker_6"
# audio = convert_to_speech(text, voice_name)
# save_audio(audio, "output.wav")

# from gtts import gTTS
# from io import BytesIO
# mp3_fp = BytesIO()
# tts = gTTS('hello how are you', lang="en")

# tts.write_to_fp(mp3_fp)

from gtts import gTTS
from time import sleep
import os
import pyglet

tts = gTTS(text='Hello World', lang='en')
filename = './tmp/temp.mp3'
tts.save(filename)

music = pyglet.media.load(filename, streaming=False)
music.play()

sleep(music.duration)
os.remove(filename)