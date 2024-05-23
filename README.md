# Build summarizer

## Description

This is a Python script that:

- takes an mp4 video as input
- separates the audio from the video (mp3 files in the chunks folder; chunks of 25MB)
- transcribes the audio to text with a whisper model in Azure
- summarizes the text with a GPT-4o model in Azure
- saves the summary as a markdown file in the same folder as the video and the same name as the video but with a `.md` extension

**Note:** the creation of the summary is basic and can be improved.

## Add a .env file

```bash
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_API_KEY_4o=your_api_key_4o_here
AZURE_OPENAI_ENDPOINT_4o=your_endpoint_4o_here
```

## Requirements

Create a Python virtual environment and install the requirements. Some requirements might require additional dependencies to be installed on your system.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add mp4 videos to a `videos` folder.

Run the app with:

```bash
python main.py
```

You might need to modify the `main.py` file with the name of your whisper and GPT-4o model deployments.

Select the video to start processing. When a video is processed, there will be a file with the same name as the video but with a `.json` extension. This file will contain the transcript of the video as JSON. Each entry is a part of the transcription.

When such a file exists, the video will not be processed again. The summary, however, will be generated again. This makes it easier to experiment with other prompts.

## Assistant

The file `assistant.py` uses the Azure Assistant API v2 to add all *.md files to a vector search so you can chat with the assistant about the content of the files.
