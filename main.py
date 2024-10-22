from pathlib import Path
import streamlit as st

from chain import retrieval_chain
from vector_store_create import index_vector_store

PATH_OUTPUT = Path(__file__).parent / 'outputs'


def sidebar():
    uploaded_video = st.file_uploader(
        'Adicione o vídeo para análise',
        type=['.mp4'],
        accept_multiple_files=False
    )
    if not uploaded_video is None:
        for arquivo in PATH_OUTPUT.glob('*.mp4'):
            arquivo.unlink()
        with open(PATH_OUTPUT / uploaded_video.name, 'wb') as f:
            f.write(uploaded_video.read())

    label_botao = 'Indexar Vídeo'
    if 'vector_store' in st.session_state:
        label_botao = 'Atualizar indexação'
    if st.button(label_botao, use_container_width=True):
        if len(list(PATH_OUTPUT.glob('*.mp4'))) == 0:
            st.error('Adicione arquivo .mp4 para inicializar o chatbot')
        else:
            st.success('Inicializando o ChatBot...')
            st.session_state['vector_store'] = index_vector_store()
            st.rerun()


def chat_window():
    st.header('🤖 Bem-vindo ao Chat com Vídeo', divider=True)

    if not 'vector_store' in st.session_state:
        st.error('Faça o upload do seu vídeo para começar!')
        st.stop()

    if not 'chain' in st.session_state:
        st.session_state['chain'] = retrieval_chain()

    chain = st.session_state['chain']
    memory = chain.memory

    mensagens = memory.load_memory_variables({})['chat_history']

    container = st.container()
    for mensagem in mensagens:
        chat = container.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    nova_mensagem = st.chat_input('Converse com seu vídeo...')
    if nova_mensagem:
        chat = container.chat_message('human')
        chat.markdown(nova_mensagem)
        chat = container.chat_message('ai')
        chat.markdown('Gerando resposta')

        resposta = chain.invoke({'question': nova_mensagem})
        st.session_state['ultima_resposta'] = resposta
        st.rerun()


def main():
    with st.sidebar:
        sidebar()
    chat_window()


if __name__ == '__main__':
    main()
