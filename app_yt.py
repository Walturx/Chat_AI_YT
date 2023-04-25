import streamlit as st
from streamlit_chat import message
from utils import (
    ingest_youtube_video_url,
    text_to_docs,
    embed_docs,
    search_docs,
    get_answer
    )
from openai.error import OpenAIError

def clear_submit():
    st.session_state["submit"] = False

def set_openai_api_key(api_key: str):
    st.session_state["OPENAI_API_KEY"] = api_key

st.markdown('<h1>YT GPT 🤖<small> by <a href="https://www.youtube.com/@nicocmw8501">Nico CMW</a></small></h1>', unsafe_allow_html=True)

# Sidebar
index = None
doc = None
with st.sidebar:
    user_secret = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Pega tu OpenAI key aqui (sk-...)",
        help="Puedes encontrar tu key aqui: https://platform.openai.com/account/api-keys.",
        value=st.session_state.get("OPENAI_API_KEY", ""),
    )
    if user_secret:
        set_openai_api_key(user_secret)
    def get_text_YT(key):
        if user_secret:
            st.header("Introduce el link del video:")
            input_text = st.text_area(key=key,label="Tu:", on_change=clear_submit)
            return input_text
    user_input_YT = get_text_YT("user_input_YT")

    if user_input_YT is not None:
        doc = ingest_youtube_video_url(user_input_YT)
        text = text_to_docs(doc)
        try:
            with st.spinner("Indexing ... esto puede llevar un rato⏳"):
                index = embed_docs(text)
                st.session_state["api_key_configured"] = True
        except OpenAIError as e:
            st.error(e._message)

tab1, tab2 = st.tabs(["Intro", "Habla con el video"])
with tab1:
    st.markdown("### Como funciona?")
    st.markdown('Para enterarte como funciona te dejo aqui un video: [Video YT](https://www.youtube.com/watch?v=vjRhxXDFtxk&ab_channel=NicoCMW)')
    st.write("### YT GPT esta hecho con las siguientes herramientas:")
    st.markdown("#### Streamlit")
    st.write("La interfaz se ha creado con [Streamlit]('https://streamlit.io/').")
    st.markdown("#### LangChain")
    st.write("Para responder preguntas con contexto [Langchain QA]('https://langchain.readthedocs.io/en/latest/use_cases/question_answering.html#adding-in-sources').")
    st.markdown("#### Embedding")
    st.write('[Embedding]("https://platform.openai.com/docs/guides/embeddings") con la API de OpenAI "text-embedding-ada-002"')
    st.markdown("""---""")
    st.write('Autor: [Nicolas Cort](https://www.linkedin.com/in/nicolas-cort-manubens-194124216/)')
    st.write('Repo: [Github](https://github.com/nicolas-cort/YT_GPT)')


with tab2:
    st.write('Para conseguir una API key es necesario que te crees una cuenta de OpenAI: https://openai.com/api/')
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    def get_text(key):
        if user_secret:
            st.header("Pregunta lo que quieras:")
            input_text = st.text_area(key=key,label="Tu:", on_change=clear_submit)
            return input_text
    user_input = get_text("user_input")

    button = st.button("Submit")
    if button or st.session_state.get("submit"):
        if not user_input:
            st.error("Please enter a question!")
        else:
            st.session_state["submit"] = True
            sources = search_docs(index, user_input)
            try:
                answer = get_answer(sources, user_input)
                st.session_state.past.append(user_input)
                st.session_state.generated.append(answer["output_text"].split("SOURCES: ")[0])
            except OpenAIError as e:
                st.error(e._message)
            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    message(st.session_state["generated"][i], key=str(i))
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')