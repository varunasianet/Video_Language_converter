import gradio as gr
import requests
import uuid
import azure.cognitiveservices.speech as speechsdk
import numpy as np
import tempfile
import os
import base64
from moviepy.editor import CompositeVideoClip
import time
import shutil
import subprocess
from dotenv import load_dotenv
import subprocess
import os
from moviepy.editor import VideoFileClip, AudioFileClip, vfx
import langcodes
import logging
from pydub.utils import mediainfo
import moviepy.editor as mp
from moviepy.editor import VideoFileClip
from moviepy.editor import concatenate_videoclips

logging.basicConfig(level=logging.INFO)

load_dotenv()

speech_key = os.environ.get("SPEECH_KEY")
speech_region = os.environ.get("SPEECH_REGION")
translator_key = os.environ.get("TRANSLATOR_KEY")
translator_endpoint = os.environ.get("TRANSLATOR_ENDPOINT")
translator_location = os.environ.get("TRANSLATOR_LOCATION")
text_to_speech_subscription_key = os.environ.get("TEXT_TO_SPEECH_SUBSCRIPTION_KEY")
text_to_speech_region = os.environ.get("TEXT_TO_SPEECH_REGION")

from pydub import AudioSegment

temp_dir = '/tmp/gradio'
os.makedirs(temp_dir, exist_ok=True)

def speech_to_text(audio_file_path, language_code):
    # Convert audio file to the required format
    converted_audio_path = convert_audio_to_wav_format(audio_file_path)
    
    if not speech_key or not speech_region:
        return "Please set the 'SPEECH_KEY' and 'SPEECH_REGION' environment variables for Speech SDK."

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    print("Setting language code to:", language_code)
    speech_config.speech_recognition_language = language_code
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Recognizing speech from the audio file...")
    
    done = False
    result_text = []

    def stop_cb(evt):
        """callback that stops continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True
    
    def canceled_cb(evt):
        """callback that stops continuous recognition upon receiving an event `evt`"""
        print('CANCELED: {}'.format(evt))
        if evt.reason == speechsdk.CancellationReason.Error:
            print('Error details: {}'.format(evt.error_details))
        nonlocal done
        done = True



    speech_recognizer.recognized.connect(lambda evt: result_text.append(evt.result.text))
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(canceled_cb)

    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)

    speech_recognizer.stop_continuous_recognition()
    
    if result_text:
        return " ".join(result_text)
    else:
        return "No speech recognized"
    
def translate_text(input_text, target_languages, source_language):
    path = '/translate'
    constructed_url = translator_endpoint + path

    params = {
        'api-version': '3.0',
        'from': source_language,
        'to': ','.join(target_languages)
    }

    headers = {
        'Ocp-Apim-Subscription-Key': translator_key,
        'Ocp-Apim-Subscription-Region': translator_location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    body = [{
        'text': input_text
    }]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()

    print("Translation API Response:", response)  # Add this line for debugging

    translation_results = []
    if response and isinstance(response, list) and 'translations' in response[0]:
        for lang, translation in zip(target_languages, response[0]['translations']):
            translation_results.append(translation)
    else:
        for lang in target_languages:
            translation_results.append({"text": "Translation not available", "to": lang})

    return translation_results

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

def replace_audio_in_video(video_file_path, new_audio_file_path):
    """
    Replace the audio track in a video file with a new audio track and adjust the video speed to match the audio length.
    """
    video = VideoFileClip(video_file_path)
    new_audio = AudioFileClip(new_audio_file_path)

    # Calculate durations
    video_duration = video.duration
    audio_duration = new_audio.duration

    # Calculate the speed factor
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

def gradio_interface(file, input_language, languages, file_type, gender):
    input_language_code = input_language.split('(')[1].split(')')[0]

    # Prepare a default response for all outputs
    default_output = [None] * (2 * len(languages))  # Two outputs per language

    if not file:
        # The first output is always the recognized text, followed by pairs of outputs for each language
        return ["No file uploaded"] + default_output

    if file_type == 'audio':
        audio_file = file.name
    elif file_type == 'video':
        if file is None or file.name is None:
            return ["Invalid video file"] + [None] * (2 * len(languages))
        audio_file = extract_audio_from_video(file.name)
    else:
        return ["Unsupported file type"] + [None] * (2 * len(languages))

    recognized_text = speech_to_text(audio_file, input_language_code)
    outputs = [recognized_text]

    # Initialize default outputs for each language
    default_outputs = {'ta': (None, None), 'en': (None, None), 'hi': (None, None),
                       'kn': (None, None), 'ml': (None, None), 'bn': (None, None), 'te': (None, None)}

    if recognized_text != "No speech recognized":
        language_codes = [lang.split(' ')[1][1:-1] for lang in languages]
        
        translation_results = translate_text(recognized_text, language_codes, input_language_code)

        for translation_result, language_code in zip(translation_results, language_codes):
            translated_text = translation_result.get('text', 'Translation not available')
            default_outputs[language_code] = (translated_text, None)

            if translated_text != 'Translation not available':
                tts_audio_path = text_to_speech(translated_text, language_code, gender)
                if file_type == 'video':
                    video_with_new_audio = replace_audio_in_video(file, tts_audio_path)
                    default_outputs[language_code] = (translated_text, video_with_new_audio)
                else:
                    default_outputs[language_code] = (translated_text, tts_audio_path)

    # Add all language outputs to the final output
    final_outputs = [recognized_text]
    for lang_code in ['ta', 'en', 'hi', 'kn', 'ml', 'bn', 'te']:
        final_outputs.extend(default_outputs[lang_code])

    return final_outputs

def convert_audio_to_wav_format(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    output_path = audio_file_path.replace(audio_file_path.split('.')[-1], 'wav')
    audio.export(output_path, format="wav")
    return output_path

def extract_audio_from_video(video_file_path):
    if not video_file_path:
        raise ValueError("Video file path is None")
    try:
        video = mp.VideoFileClip(video_file_path)
        audio_file_path = video_file_path + ".mp3"
        video.audio.write_audiofile(audio_file_path)
        converted_audio_path = convert_audio_to_wav_format(audio_file_path)
        return converted_audio_path
    except Exception as e:
        logging.error(f"Error in extract_audio_from_video: {e}")
        return None

with open("Asianet_news_logo.png", "rb") as img_file:
    b64_string = base64.b64encode(img_file.read()).decode('utf-8')

# Define an audio interfaceß
file_input = gr.File(label="Upload an audio or video file")
audio_input = gr.Audio(type="filepath", label="Upload an audio file")
input_language_input = gr.Radio(choices=[
    'Tamil (ta-IN)', 
    'English (en-US)', 
    'Hindi (hi-IN)', 
    'Kannada (kn-IN)', 
    'Malayalam (ml-IN)', 
    'Bengali (bn-IN)', 
    'Telugu (te-IN)'
], label="Select Input Language")
language_input = gr.CheckboxGroup(choices=[
    'Tamil (ta)', 
    'English (en)', 
    'Hindi (hi)', 
    'Kannada (kn)', 
    'Malayalam (ml)', 
    'Bengali (bn)', 
    'Telugu (te)'
], label="Select Languages to convert")
file_type_input = gr.Radio(choices=["audio", "video"], label="File Type")
gender_input = gr.Radio(choices=['male', 'female'], label="Select Gender")

# Define inputs as before
inputs = [file_input, input_language_input, language_input, file_type_input, gender_input]

# Define a function to dynamically generate output components   a a
def dynamic_outputs(languages, file_type):
    outputs = [gr.Textbox(label="Recognized Text")]
    for lang in languages:
        lang_name = lang.split(' ')[0]
        outputs.append(gr.Textbox(label=f"Translated Text {lang_name}"))
        if file_type == 'audio':
            outputs.append(gr.Audio(label=f"Generated Audio {lang_name}"))
        elif file_type == 'video':
            outputs.append(gr.Video(label=f"Video with {lang_name} Audio"))
    return outputs


iface = gr.Interface(
    fn=gradio_interface,
    inputs=inputs,
    outputs=dynamic_outputs(['Tamil (ta)', 'English (en)', 'Hindi (hi)', 'Kannada (kn)', 'Malayalam (ml)', 'Bengali (bn)', 'Telugu (te)'], 'video'),
    title="Video to Multilingual Text & Video Converter",
    description=f'<img src="data:image/png;base64,{b64_string}" alt="Header Image" style="display: block; margin: 0 auto; width: 100px; height: auto;"><p>User can convert the target language | Developed InHouse | <p>',
)

# iface.queue()
iface.launch(share=True, enable_queue=False,server_name="0.0.0.0" ,debug=True, inbrowser=True , server_port=int(os.environ.get('PORT', 8888))) #changed server 


