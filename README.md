## 1. System Requirements

| Item          | Description                                              |
|---------------|----------------------------------------------------------|
| OS            | Recommended: Windows 10 or above / macOS / Linux         |
| Python Version| 3.8 or later (Python 3.10 recommended)                   |
| RAM           | At least 8GB (16GB+ recommended for large batch jobs)   |
| GPU           | Not required (this tool runs on CPU)                    |

---

## 2. Required Packages (Install Before First Use)

Run the following commands in your terminal to install the necessary packages:

```
pip install yt-dlp
pip install opencc
pip install requests
pip install beautifulsoup4
pip install funasr
```
 If you're using a virtual environment (e.g., venv, conda), make sure it is activated before installing.


 ## 3. Install FFmpeg
After installation, add the folder containing ffmpeg.exe to your system PATH.

Linux (Ubuntu):

```
sudo apt install ffmpeg
```

## 4. Folder Structure & Configuration
D:\
└── csmu\
    └── icare\
        ├── videos     ← Stores downloaded audio files
        └── output     ← Stores transcribed .txt files

In the script, find this line:

python
base_folder = r"D:\csmu\icare"
Change the path to match your local directory.

## 5. How to Run
Save the full script as yt_to_text_channel.py (or any name you prefer), then run it in the terminal:

```
python yt_to_text_channel.py
```

**all models are pull from ollama**


**for streamlit_taide.py:**
```
pip install streamlit chromadb ollama
```
pull embedding model / llm in ollama:
```
ollama pull mxbai-embed-large
ollama pull TAIDE-Medicine-QA-TW-Q6
```
**for gradio_breeze.py:**
```
pip install gradio chromadb ollama
```
pull embedding model / llm in ollama:
```
ollama pull mxbai-embed-large
ollama pull ycchen/breeze-7b-instruct-v1_0
```

**for lang.py:**

using LangChain
```
pip install langchain langchain-community langchain-nomic langchain-ollama chromadb ollama 
```
pull llm in ollama:
```
ollama pull mistral
```
