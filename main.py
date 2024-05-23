from moviepy.editor import VideoFileClip
import os
from openai import AzureOpenAI
from pydub import AudioSegment
import json
from InquirerPy import prompt
from promptflow.tracing import trace, start_trace

start_trace() # this will display a link to the tracing ui


# load environment variables
from dotenv import load_dotenv
load_dotenv()

# setup logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

azure_openai_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_openai_key_4o = os.getenv('AZURE_OPENAI_API_KEY_4o')
azure_openai_endpoint_4o = os.getenv('AZURE_OPENAI_ENDPOINT_4o')

whisper_client = AzureOpenAI(
    api_key=azure_openai_key,
    api_version="2024-02-01",
    azure_endpoint=azure_openai_endpoint
)

gpt_client = AzureOpenAI(
    api_key=azure_openai_key_4o,
    api_version="2024-02-01",
    azure_endpoint=azure_openai_endpoint_4o
)

whisper_deployment = 'whisper'
gpt_deployment = 'gpt-4o'

def split_audio(file_path, chunk_size_mb=25):
    logger.info(f"Splitting audio file {file_path} into chunks of {chunk_size_mb} MB")
    
    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    # Calculate the number of bytes per millisecond
    audio_bitrate = audio.frame_rate * audio.frame_width * audio.channels
    bytes_per_millisecond = audio_bitrate / 8 / 1000

    # Convert chunk size from MB to bytes
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    chunk_size_ms = int(chunk_size_bytes / bytes_per_millisecond)

    # Get the file extension
    file_ext = os.path.splitext(file_path)[1]

    # Create output directory
    output_dir = "chunks"
    os.makedirs(output_dir, exist_ok=True)

    # Split the audio and save chunks
    for i in range(0, len(audio), chunk_size_ms):
        logger.info(f"Writing chunk {i // chunk_size_ms + 1}")
        chunk = audio[i:i + chunk_size_ms]
        chunk.export(os.path.join(output_dir, f"chunk_{i // chunk_size_ms + 1}{file_ext}"), format=file_ext[1:])
        logger.info(f"Chunk {i // chunk_size_ms + 1} written to disk")

def extract_sections(text):
    logger.info("Extracting sections from text")
    result = gpt_client.chat.completions.create(
        model=gpt_deployment,
        response_format={ "type": "json_object" },
        messages=[
            {
                "role": "system",
                "content":  """
                            You are an expert in creating sections from a video transcript. Identify the main sections with a limit of 5.
                            Return the sections as JSON with the following format:
                            {
                                "sections": [
                                    {
                                        "title": "Few words describing the content of the section",
                                        "summary": "One sentence summary of the content of section 1"
                                    },
                                    {
                                        "title": "Few words describing the content of the section",
                                        "summary": "One sentence summary of the content of section 2"
                                    }
                                ]
                            }
                            """
            },
            {
                "role": "user",
                "content": text
            }
        ],
        max_tokens=4000
    )
    
    return result.choices[0].message.content

@trace
def create_summary(transcriptions):
    logger.info("Creating summary from sections")
    
    # iterate over JSON string sections and create a summary
    summary = ''
    for t in transcriptions:
        # get gpt response for summary of this section
        result = gpt_client.chat.completions.create(
            model=gpt_deployment,
            messages=[
                {
                    "role": "system",
                    "content":  """
                                You are an expert in summarizing video transcripts. When the user
                                provides a transcript, summarize the content of the transcript.
                                Make the summary as detailed as possible. If there are new releases or solutions, call them out.
                                If you know the name of the speaker, use that name.
                                Use markdown and provide a single heading. The heading should not contain the word "summary" and
                                should be descriptive of the content.
                                Use bullet points sparingly. End the summary with 3 key takeaways.
                                """
                },
                {
                    "role": "user",
                    "content":  f"""
                                Section summary= {t}
                                """
                }
            ],
            max_tokens=4000
        )
        section_summary = result.choices[0].message.content
        summary += f"{section_summary}\n\n"
        
    return summary
        
    
    
    
def extract_audio_from_mp4(mp4_file):
    logger.info(f"Extracting audio from {mp4_file}")
    video = VideoFileClip(mp4_file)
    audio = video.audio
    
    # Output file is the same as the input file but with .mp3 extension
    output_file = mp4_file.replace('.mp4', '.mp3')
    
    # does output file already exist?
    if not os.path.exists(output_file):
        logger.info(f"Writing audio to {output_file}")
        audio.write_audiofile(output_file)
    
    return output_file


def transcribe_audio(audio_file):
    # check file size of audio file
    file_size = os.path.getsize(audio_file)
    
    # if file is greater than 25MB, break up into smaller files
    if file_size > 25e6:
        logger.info(f"Audio file {audio_file} is larger than 25MB. Splitting into chunks")
        split_audio(audio_file)
        audio_files = os.listdir('./chunks')
        
        # order audio files based on number at the end
        audio_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        transcriptions = []
        for audio_file in audio_files:
            logger.info(f"Transcribing {audio_file}")
            result = whisper_client.audio.transcriptions.create(
                file=open(os.path.join('./chunks', audio_file), 'rb'),
                model=whisper_deployment
            )
            transcriptions.append(result.text)
            
        # delete all chunk files
        logger.info("Deleting chunk files")
        for audio_file in audio_files:
            os.remove(os.path.join('./chunks', audio_file))
            
    else:
        logger.info(f"File under 25 MB. Transcribing {audio_file}")
        result = whisper_client.audio.transcriptions.create(
            file=open(audio_file, 'rb'),
            model=whisper_deployment
        )
        transcriptions = []
        transcriptions.append(result.text)
        
    return transcriptions

def list_mp4_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.mp4')]

def select_mp4_file(directory):
    files = list_mp4_files(directory)
    
    if not files:
        print("No MP4 files found in the specified directory.")
        return None

    choices = [{'name': file, 'value': file} for file in files]

    questions = [
        {
            'type': 'list',
            'name': 'selected_file',
            'message': 'Select an MP4 file:',
            'choices': choices
        }
    ]

    answers = prompt(questions)
    selected_file = answers['selected_file']

    app_path = os.path.abspath(os.path.dirname(__file__))
    relative_path = os.path.relpath(os.path.join(directory, selected_file), app_path)
    return relative_path


if __name__ == '__main__':
    input_file = select_mp4_file("./videos")
    if not input_file:
        logger.info("No MP4 files found in the specified directory. Exiting.")
        exit()
    
    print(f"Selected file: {input_file}")
    input("Press enter to continue...")
    
    output_file = input_file.replace('.mp4', '.json')  # this is the raw transcription file
    
    if not os.path.exists(output_file):
        mp3_file = extract_audio_from_mp4(input_file)
        transcriptions = transcribe_audio(mp3_file)
        
        # save the transcriptions list as json to output_file
        logger.info(f"Writing transcriptions to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(transcriptions, f)
            
    else:
        logger.info(f"Transcription already exists for {input_file}. Skipping transcription")
        
        # read transcription from file
        logger.info(f"Reading transcription from {output_file}")
        with open(output_file, 'r') as f:
            transcriptions = json.load(f)
    
    # create a summary based on the transcriptions
    summary = create_summary(transcriptions)
    
    # save summary to file with same name as input file but with .md extension
    summary_file = input_file.replace('.mp4', '.md')
    logger.info(f"Writing summary to {summary_file}")
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    