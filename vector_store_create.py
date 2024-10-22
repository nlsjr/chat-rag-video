import os
import shutil
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from utils_model import get_embedding_model
from chain import chain_summarize_scene, chain_describe_image, chain_summarize_images
from client_openai import transcribe_video
from utils_video import get_output_path, generate_frames, get_mp4_filename, remove_all_files_from_frames

PATH_VIDEO = Path(__file__).parent / 'outputs'


def get_segments():
    return transcribe_video(PATH_VIDEO / get_mp4_filename(PATH_VIDEO)[0])['segments']


def generate_frames_from_segments(segments):
    paths = []

    for segment in segments:
        path_frames = os.path.join(os.getcwd(), "frames", f"{segment['start']}_{segment['end']}")
        if not os.path.exists(path_frames):
            os.makedirs(path_frames)

        generate_frames(segment['start'], segment['end'], path_frames,
                        os.path.join(get_output_path(), get_mp4_filename(PATH_VIDEO)[0]))

        path_segment = {
            "path": path_frames,
            "start": segment['start'],
            "end": segment['end'],
            "text": segment['text']
        }
        paths.append(path_segment)

    return paths


def delete_index_dir():
    index_path = os.path.join(os.getcwd(), "index")
    if os.path.exists(index_path):
        shutil.rmtree(index_path)


def add_to_vector_store(scene_summary, path_segment):
    vector_store = Chroma(
        collection_name="mm_rag_clip_photos",
        persist_directory=os.getcwd() + "/index",
        embedding_function=get_embedding_model()
    )

    i = str(uuid.uuid4())
    document = Document(
        page_content=scene_summary,
        metadata={
            "id": i,
            "type": "segment",
            "start_time": path_segment['start'],
            "end_time": path_segment['end'],
            "text": path_segment['text']
        })

    vector_store.add_documents([document])

    return vector_store


def index_vector_store():
    remove_all_files_from_frames()
    delete_index_dir()

    segments = get_segments()
    path_segments = generate_frames_from_segments(segments)

    for path_segment in path_segments:
        imgs_described = chain_describe_image(path_segment['path'])
        chain_response = chain_summarize_scene().run({
            "image_summary": '\n'.join(imgs_described),
            "video_transcription": path_segment['text']
        })

        add_to_vector_store(chain_response, path_segment)
