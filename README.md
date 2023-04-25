# README.md
## Introduction
This is a Streamlit app for chatting with YouTube videos using Langchain, a Python library for building AI-powered chatbots that can answer questions based on documents. The app allows users to input a YouTube video URL, which is then transcribed and indexed to allow users to ask questions about the video. The app uses OpenAI for text embeddings and Langchain for question-answering with context.

<div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
    <img src="https://user-images.githubusercontent.com/110467001/234264788-87a19c24-0c10-4dfd-8c28-fd793615308d.png" alt="Red Professional Business YouTube Thumbnail" style="max-width: 80%; max-height: 80%; margin: auto;">
</div>



## Requirements
- Python 3.7 or above
- Required packages: `langchain`, `streamlit`, `openai`
- A valid OpenAI API key

## Usage
1. Clone the repository
2. Install the required packages using `pip install -r requirements.txt`
3. Set your OpenAI API key by running the command `streamlit run main.py` and entering your key in the sidebar.
4. Run the app using `streamlit run main.py`
5. Enter a YouTube video URL in the sidebar and click submit.
6. Ask a question about the video in the text area and click submit.

## Files
- `main.py`: The main file containing the Streamlit app.
- `yt_loader.py`: Contains the function `ingest_youtube_video_url` for transcribing YouTube videos.
- `utils.py`: Contains helper functions for processing the video transcript and embedding the text.
- `template.py`: Contains the prompt template used for Langchain's question-answering model.

## References
- [Streamlit](https://streamlit.io/)
- [OpenAI API](https://openai.com/api/)
- [Langchain Documentation](https://langchain.readthedocs.io/)
