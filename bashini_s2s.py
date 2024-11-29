import gradio as gr
import requests
import uuid
import base64
import numpy as np
import tempfile
import os
import time
import shutil
import subprocess
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, vfx
import logging
import langcodes
import logging
from pydub.utils import mediainfo
import moviepy.editor as mp
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import http.client
import json
import random


logging.basicConfig(level=logging.INFO)

load_dotenv()

text_to_speech_subscription_key = os.environ.get("TEXT_TO_SPEECH_SUBSCRIPTION_KEY")
text_to_speech_region = os.environ.get("TEXT_TO_SPEECH_REGION")

# Load service configuration
def load_config_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

config_file_path = "service.json"  # Adjust the path as necessary
config_json = load_config_json(config_file_path)

def find_service_id(task_type, source_language, target_language=None):
    logging.info(f"Looking for service ID for task: {task_type}, source language: {source_language}, target language: {target_language}")
    for config in config_json["pipelineResponseConfig"]:
        if config["taskType"] == task_type:
            for task_config in config["config"]:
                if task_config["language"]["sourceLanguage"] == source_language:
                    if target_language:
                        if task_config["language"]["targetLanguage"] == target_language:
                            logging.info(f"Found service ID: {task_config['serviceId']} for source language: {source_language} and target language: {target_language}")
                            return task_config["serviceId"]
                    else:
                        logging.info(f"Found service ID: {task_config['serviceId']} for source language: {source_language}")
                        return task_config["serviceId"]
    logging.error(f"No service ID found for task: {task_type}, source language: {source_language}, target language: {target_language}")
    return None


# Bhashini ASR function
def bhashini_speech_to_text(audio_file_path, language_code):
    # Find ASR service ID
    asr_service_id = find_service_id("asr", language_code)
    
    if not asr_service_id:
        return "Error: ASR service ID not found for the specified language."

    with open(audio_file_path, "rb") as audio_file:
        audio_content_base64 = base64.b64encode(audio_file.read()).decode()

    payload = json.dumps({
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {"sourceLanguage": language_code},
                    "serviceId": asr_service_id,
                    "preProcessors": ["vad"],
                    "audioFormat": os.path.splitext(audio_file_path)[1][1:],  # Extract file extension without dot
                    "samplingRate": 16000
                }
            }
        ],
        "inputData": {"audio": [{"audioContent": audio_content_base64}]}
    })

    headers = {
        'Accept': '*/*',
        'Authorization': config_json["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"],
        'Content-Type': 'application/json'
    }

    logging.debug(f"Sending ASR request with payload: {payload}")

    conn = http.client.HTTPSConnection("dhruva-api.bhashini.gov.in")
    conn.request("POST", "/services/inference/pipeline", payload, headers)
    res = conn.getresponse()
    data = res.read()

    if not data:
        logging.error("Received empty response from Bhashini API")
        return "Error: Received empty response from Bhashini API"

    try:
        response_json = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON response: {e}")
        return f"Error: Failed to decode JSON response: {e}"

    if res.status == 200:
        try:
            return response_json["pipelineResponse"][0]["output"][0]["source"]
        except KeyError as e:
            logging.error(f"Key error in response JSON: {e}")
            return f"Error: Key not found in response JSON: {e}"
        except IndexError as e:
            logging.error(f"Index error in response JSON: {e}")
            return f"Error: Index out of range in response JSON: {e}"
    else:
        error_message = response_json.get('message', 'Unknown error')
        logging.error(f"ASR request failed with status {res.status}: {error_message}")
        return f"Error: ASR request failed with status {res.status} - {error_message}"


def validate_input(input_text, target_languages, source_language):
    if not input_text or not isinstance(input_text, str):
        raise ValueError("Input text must be a non-empty string")
    if not target_languages or not isinstance(target_languages, list):
        raise ValueError("Target languages must be a non-empty list")
    if not source_language or not isinstance(source_language, str):
        raise ValueError("Source language must be a non-empty string")
    
    # Check if languages are supported
    supported_languages = set(lang["sourceLanguage"] for lang in config_json["languages"])
    if source_language not in supported_languages:
        raise ValueError(f"Unsupported source language: {source_language}")
    for lang in target_languages:
        if lang not in supported_languages:
            raise ValueError(f"Unsupported target language: {lang}")


def translate_text(input_text, target_languages, source_language, max_retries=3):
    logging.info(f"Translating text: {input_text}")
    logging.info(f"Source language: {source_language}, Target languages: {target_languages}")

    results = []

    for target_language in target_languages:
        nmt_service_id = find_service_id("translation", source_language, target_language)
        if not nmt_service_id:
            logging.error(f"No NMT service ID found for translation from {source_language} to {target_language}")
            results.append({"text": "Translation service not available", "to": target_language})
            continue

        headers = {
            'Accept': '*/*',
            'Authorization': config_json["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"],
            'Content-Type': 'application/json'
        }
        
        payload = {
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        "language": {
                            "sourceLanguage": source_language,
                            "targetLanguage": target_language
                        },
                        "serviceId": nmt_service_id
                    }
                }
            ],
            "inputData": {
                "input": [
                    {
                        "source": input_text
                    }
                ]
            }
        }

        payload_json = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        logging.info(f"Translation payload for {target_language}: {payload_json.decode('utf-8')}")

        for attempt in range(max_retries):
            try:
                logging.info(f"Attempt {attempt + 1} of {max_retries} for translation to {target_language}")
                conn = http.client.HTTPSConnection("dhruva-api.bhashini.gov.in")
                conn.request("POST", "/services/inference/pipeline", payload_json, headers)
                res = conn.getresponse()
                data = res.read()

                if res.status == 200:
                    response_json = json.loads(data.decode("utf-8"))
                    logging.info(f"Translation response for {target_language}: {response_json}")
                    if "pipelineResponse" in response_json and response_json["pipelineResponse"]:
                        translated_text = response_json["pipelineResponse"][0]["output"][0]["target"]
                        results.append({"text": translated_text, "to": target_language})
                        break
                    else:
                        logging.error(f"Unexpected response structure for {target_language}")
                        results.append({"text": "Unexpected response structure", "to": target_language})
                        break
                else:
                    logging.error(f"Translation request failed with status {res.status}: {data.decode('utf-8')}")
                    if attempt == max_retries - 1:
                        results.append({"text": f"Error: Server returned {res.status}", "to": target_language})
                    else:
                        wait_time = (attempt + 1) * 2 + random.uniform(0, 1)
                        logging.info(f"Retrying in {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
            except Exception as e:
                logging.error(f"Exception occurred during translation request to {target_language}: {str(e)}")
                if attempt == max_retries - 1:
                    results.append({"text": "An error occurred during translation", "to": target_language})
                else:
                    wait_time = (attempt + 1) * 2 + random.uniform(0, 1)
                    logging.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
            finally:
                conn.close()

    return results

voice_name = {
    'ta': {'male': 'ta-IN-ValluvarNeural', 'female': 'ta-IN-PallaviNeural'},
    'en': {'male': 'en-US-GuyNeural', 'female': 'en-US-AriaNeural'},
    'hi': {'male': 'hi-IN-MadhurNeural' ,'female': 'hi-IN-SwaraNeural'},
    'kn': {'male': 'kn-IN-GaganNeural' ,'female': 'kn-IN-SapnaNeural'},
    'ml': {'male': 'ml-IN-MidhunNeural' ,'female': 'ml-IN-SobhanaNeural'},
    'bn': {'male': 'bn-IN-BashkarNeural' ,'female': 'bn-IN-TanishaaNeural'},
    'te': {'male': 'te-IN-MohanNeural' ,'female': 'te-IN-ShrutiNeural'}
}

def text_to_speech(text, language_code, gender):
    # Ensure gender is in lowercase and default to 'female' if not provided
    gender = gender.lower() if gender else 'female'

    # Get the voice options for the given language code
    voice_options = voice_name.get(language_code, {})

    # If no voice options found, default to English female voice
    if not voice_options:
        logging.info(f"Warning: No voices found for language code '{language_code}'. Defaulting to English female voice.")
        voice_options = ('en-US', 'en-US-AriaNeural')
    
    # Check if the language code is supported
    lang_code = language_code if language_code in voice_name else 'en-US'

    # Select the voice based on the provided gender, default to 'en-US-AriaNeural' if not found
    selected_voice = voice_options.get(gender, 'en-US-AriaNeural')

    # Debug print selected voice
    logging.info(f"Debug selected voice: {selected_voice}")

    # Define the SSML payload
    ssml_payload = f'<speak version="1.0" xml:lang="{lang_code}"><voice xml:lang="{lang_code}" xml:gender="{gender}" name="{selected_voice}">{text}</voice></speak>'

    # Define the headers for the API request
    headers = {
        "Ocp-Apim-Subscription-Key": text_to_speech_subscription_key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
        "User-Agent": "curl",
    }

    # Make a POST request to the Text-to-Speech API
    response = requests.post(
        f"https://{text_to_speech_region}.tts.speech.microsoft.com/cognitiveservices/v1",
        headers=headers,
        data=ssml_payload.encode('utf-8'),
    )

    if response.status_code == 200:
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
            # Write the response content to the temporary audio file
            temp_audio_file.write(response.content)
            temp_audio_file_path = temp_audio_file.name
            print(f"Generated audio file {temp_audio_file_path} with length {len(response.content)} bytes.")
        return temp_audio_file_path
    else:
        logging.info(f"Failed to convert text to speech. Status code: {response.status_code}, Response: {response.text}")
        return None

language_code_map = {
    'Tamil (ta-IN)': 'ta',
    'English (en-US)': 'en',
    'Hindi (hi-IN)': 'hi',
    'Kannada (kn-IN)': 'kn',
    'Malayalam (ml-IN)': 'ml',
    'Bengali (bn-IN)': 'bn',
    'Telugu (te-IN)': 'te'
}



import os
import tempfile
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
import logging
import subprocess
import shutil

logging.basicConfig(level=logging.INFO)

def replace_audio_in_video(video_file_path, new_audio_file_path):
    video = VideoFileClip(video_file_path)
    new_audio = AudioFileClip(new_audio_file_path)
    video_duration = video.duration
    audio_duration = new_audio.duration
    speed_factor = video_duration / audio_duration

    # Speed up or slow down video
    if speed_factor != 1:
        video = video.fx(vfx.speedx, speed_factor)

    # Set the new audio to the video
    video = video.set_audio(new_audio)

    # Save the edited video
    output_path = tempfile.mktemp(suffix=".mp4", prefix="output_")
    video.write_videofile(output_path, codec='libx264', audio_codec='aac')

    return output_path



def convert_audio_to_wav_format(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    output_path = audio_file_path.replace(audio_file_path.split('.')[-1], 'wav')
    audio.export(output_path, format="wav")
    return output_path

def extract_audio_from_video(video_file_path):
    try:
        video = mp.VideoFileClip(video_file_path)
        audio_file_path = video_file_path + ".mp3"
        video.audio.write_audiofile(audio_file_path)
        converted_audio_path = convert_audio_to_wav_format(audio_file_path)
        video.close()
        return converted_audio_path
    except Exception as e:
        logging.error(f"Error extracting audio from video: {str(e)}")
        raise


with open("Asianet_news_logo.png", "rb") as img_file:
    b64_string = base64.b64encode(img_file.read()).decode('utf-8')
def gradio_interface(file, input_language, file_type):
    input_language_code = input_language.split('(')[1].split(')')[0].split('-')[0]

    if file_type == 'audio':
        audio_file = file
    elif file_type == 'video':
        try:
            audio_file = extract_audio_from_video(file)
        except Exception as e:
            logging.error(f"Error extracting audio from video: {str(e)}")
            return f"Error extracting audio from video: {str(e)}"
    else:
        return "Unsupported file type"

    recognized_text = bhashini_speech_to_text(audio_file, input_language_code)
    if recognized_text.startswith("Error:"):
        return recognized_text

    return recognized_text

def process_translation_and_tts(edited_text, input_language, languages, file_type, gender, original_file):
    input_language_code = input_language.split('(')[1].split(')')[0].split('-')[0]
    language_codes = [lang.split(' ')[1][1:-1].split('-')[0] for lang in languages]
    
    outputs = []
    translation_results = translate_text(edited_text, language_codes, input_language_code)

    all_languages = ['ta', 'en', 'hi', 'kn', 'ml', 'bn', 'te']
    
    for lang in all_languages:
        translation_result = next((r for r in translation_results if r['to'] == lang), None)
        
        if translation_result:
            translated_text = translation_result.get('text', 'Translation not available')
            outputs.append(translated_text)

            if translated_text != 'Translation not available':
                tts_audio_path = text_to_speech(translated_text, lang, gender)
                if file_type == 'video' and tts_audio_path:
                    video_with_new_audio = replace_audio_in_video(original_file, tts_audio_path)
                    outputs.append(video_with_new_audio)
                else:
                    outputs.append(tts_audio_path)
            else:
                outputs.append(None)
        else:
            outputs.extend([None, None])  # Add placeholder for text and audio/video

    return outputs


# Define the Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Audio to Multilingual Text & Audio Converter")
    gr.Markdown(f'<img src="data:image/png;base64,{b64_string}" alt="Header Image" style="display: block; margin: 0 auto; width: 100px; height: auto;">')
    gr.Markdown("User can convert the target language | Developed InHouse")

    with gr.Row():
        file_input = gr.File(label="Upload an audio or video file")
        input_language_input = gr.Radio(choices=[
            'Tamil (ta-IN)', 
            'English (en-US)', 
            'Hindi (hi-IN)', 
            'Kannada (kn-IN)', 
            'Malayalam (ml-IN)', 
            'Bengali (bn-IN)', 
            'Telugu (te-IN)'
        ], label="Select Input Language")
        file_type_input = gr.Radio(choices=["audio", "video"], label="File Type")

    transcribe_button = gr.Button("Transcribe")
    transcribed_text = gr.Textbox(label="Transcribed Text (Edit if needed)")

    with gr.Row():
        language_input = gr.CheckboxGroup(choices=[
            'Tamil (ta)', 
            'English (en)', 
            'Hindi (hi)', 
            'Kannada (kn)', 
            'Malayalam (ml)', 
            'Bengali (bn)', 
            'Telugu (te)'
        ], label="Select Languages to convert")
        gender_input = gr.Radio(choices=['male', 'female'], label="Select Gender")

    process_button = gr.Button("Process Translation and TTS")

    output_boxes = []
    for lang in ['Tamil', 'English', 'Hindi', 'Kannada', 'Malayalam', 'Bengali', 'Telugu']:
        with gr.Row():
            output_boxes.append(gr.Textbox(label=f"Translated Text ({lang})"))
            output_boxes.append(gr.Audio(label=f"Generated Audio ({lang})"))

    transcribe_button.click(
        gradio_interface,
        inputs=[file_input, input_language_input, file_type_input],
        outputs=transcribed_text
    )

    process_button.click(
        process_translation_and_tts,
        inputs=[transcribed_text, input_language_input, language_input, file_type_input, gender_input, file_input],
        outputs=output_boxes
    )


iface.launch()