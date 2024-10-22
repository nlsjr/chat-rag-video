import os

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

_ = load_dotenv(find_dotenv())

open_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=open_api_key)


def transcribe_video(video_path):
    audio_file = open(video_path, "rb")
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=["segment"]
    )

    return transcript.to_dict()
