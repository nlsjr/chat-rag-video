import glob

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from utils_model import get_llm
from utils_video import image2base64
from vector_store_search import get_filtered_vector_store


def chain_describe_image(frames_path):
    files = glob.glob(frames_path + "/*.png")
    imgs_summary = []

    for file in files:
        image_str = image2base64(file)
        response = get_llm().invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text",
                         "text": "please give a summary of the image provided, be descriptive and smart. Respond in pt-br"},
                        {"type": "image_url", "image_url":
                            {
                                "url": f"data:image/png;base64,{image_str}"
                            },
                         },
                    ]
                )
            ]
        )
        imgs_summary.append(response.content)

    return imgs_summary


def chain_summarize_images(images_describe):
    llm = get_llm()
    prompt = (
        'You are analyzing several images, you have been given a description of the images of movie scenes.'
        'Summarize the entire context and summarize the scene.'
        'Be clear and objective, respond only to the analysis.'
        '\n\n'
        '{context}'
    )

    prompt_template = PromptTemplate.from_template(prompt)

    chain = LLMChain(llm=llm, prompt=prompt_template)

    return chain.run({"context": images_describe})


def chain_summarize_scene():
    llm = get_llm()
    prompt_template_str = (
        'You are analyzing a video, you have been given a context describing what is happening in the scene.'
        'You will receive a transcript of the video and a description of the image.'
        'Your role is to analyze the film scene and briefly explain what is happening.'
        'Be clear and objective, respond only to the analysis.'
        '\n\n'
        'Image Summary: {image_summary}'
        '\n'
        'Video Transcription: {video_transcription}'
    )

    prompt_template = PromptTemplate(
        input_variables=["image_summary", "video_transcription"], template=prompt_template_str
    )

    return LLMChain(llm=llm, prompt=prompt_template)


def retrieval_chain():
    llm = get_llm()

    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key='chat_history',
        output_key='answer'
    )

    metadata_filter = {
        "$and": [
            {"type": {"$eq": "segment"}},
            {"start_time": {"$gte": 0}},
            {"end_time": {"$lte": 77}}
        ]
    }
    vector_store = get_filtered_vector_store(metadata_filter)

    retriever = vector_store.as_retriever()

    system_prompt = (
        "You are a friendly Chatbot that helps interpret videos provided to you. The context provided contains information from the users video. "
        "Use the context to answer the users questions. If you don't know the answer, just say you don't know and don't try to make up the answer."
        "Your job is to analyze the scene in the film and briefly explain what is happening. Answer in pt-br."
        "\n"
        "Contexto: {context}"
        "\n"
        "Conversa atual: {chat_history}"
        "\n"
        "Human: {question}"
        "\n"
        "AI: "
    )

    prompt = PromptTemplate.from_template(system_prompt)
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

    return chat_chain
