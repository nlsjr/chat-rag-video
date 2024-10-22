import base64
import glob
import io

from PIL import Image
from pytubefix import YouTube
from moviepy.editor import *
import os


def get_frames_path():
    if not os.path.exists('frames'):
        os.makedirs('frames')
    return os.path.join(os.getcwd(), "frames")


def get_output_path():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    return os.path.join(os.getcwd(), "outputs")


def download(youtube_url):
    yt = YouTube(youtube_url)
    print(yt.title)
    ys = yt.streams.get_highest_resolution()
    ys.download(output_path=get_output_path(), filename="output.mp4")


def generate_frames(start_time, end_time, path_frames, path_video):
    cwd = os.getcwd()

    clip = VideoFileClip(os.path.join(cwd, path_video)).subclip(start_time, end_time)
    clip.write_images_sequence(
        os.path.join(path_frames, "frame%04d.png"), fps=0.2  # configure this for controlling frame rate.
    )


def image2base64(file):
    with Image.open(file) as image:
        buffer = io.BytesIO()
        image.save(buffer, format=image.format)
        img_str = base64.b64encode(buffer.getvalue())
        return img_str.decode("utf-8")


def get_mp4_filename(directory):
    mp4_files = glob.glob(os.path.join(directory, '*.mp4'))
    mp4_filename = [os.path.basename(file) for file in mp4_files]

    return mp4_filename


def remove_all_files_from_frames():
    for root, dirs, files in os.walk(os.path.join(os.getcwd(), "frames")):
        for file in files:
            os.remove(os.path.join(root, file))
