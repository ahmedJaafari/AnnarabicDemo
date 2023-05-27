import random
from google.cloud import storage
import numpy as np
import soundfile as sf
import subprocess
import gradio as gr
import json
import os
import requests
import openai
from google.cloud import speech_v1p1beta1 as speech
import yt_dlp as youtube_dl


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./annarabic-google-key.json"


def download_yt_audio(yt_url, filename):
    info_loader = youtube_dl.YoutubeDL()
    try:
        info = info_loader.extract_info(yt_url, download=False)
    except youtube_dl.utils.DownloadError as err:
        raise gr.Error(str(err))
    file_length = info["duration_string"]
    file_h_m_s = file_length.split(":")
    file_h_m_s = [int(sub_length) for sub_length in file_h_m_s]
    if len(file_h_m_s) == 1:
        file_h_m_s.insert(0, 0)
    if len(file_h_m_s) == 2:
        file_h_m_s.insert(0, 0)
    file_length_s = file_h_m_s[0] * 3600 + file_h_m_s[1] * 60 + file_h_m_s[2]
    ydl_opts = {"outtmpl": filename,
                "format": "worstvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([yt_url])
        except youtube_dl.utils.ExtractorError as err:
            raise gr.Error(str(err))


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client(project="annarabic")
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    # os.remove(source_file_name)
    return f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"


def audio_download(audio_url):
    response = requests.get(audio_url)
    audio_name = audio_url.split("/")[-1]
    open(f"./tempAudio/{audio_name}", "wb").write(response.content)
    return f"./tempAudio/{audio_name}"


def upload_audio_cloud(filepath):
    return upload_blob("storage-annarabic", filepath, filepath.split("/")[-1])


def preprocess_audio(audio_path):
    audio, samplerate = sf.read(audio_path)
    if audio.ndim > 1 and audio.shape[1] > 1:
        # Convert stereo audio to mono by averaging the channels
        audio = np.mean(audio, axis=1)
    audio_name_wav = audio_path.split(
        "/")[-1].split(".")[0] + str(random.randint(0, 10_000_000)) + ".wav"
    sf.write(f"./tempAudio/{audio_name_wav}", audio, samplerate, format='WAV')
    os.remove(audio_path)
    return f"./tempAudio/{audio_name_wav}"


def video_download(yt_url):
    audio_name = f"audio{random.randint(0,1_000_000)}"
    audio_path = f"./tempAudio/audio{audio_name}.mp4"
    download_yt_audio(yt_url, audio_path)

    subprocess.call(
        ['ffmpeg', '-i', f'{audio_path}', f'./tempAudio/{audio_name}.wav'])
    os.remove(audio_path)
    return f"./tempAudio/{audio_name}.wav"


headers = {"Authorization": "Bearer " +
           "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", "Content-type": "audio/x-audio"}
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./annarabic-google-key.json"


def speech_recognition_MA_ar(link):
    #filename = audio_download(link)
    API_URL = "https://llmqasgvjanvx005.us-east-1.aws.endpoints.huggingface.cloud"

    with open(link, "rb") as f:
        data = f.read()
    response = requests.request("POST", API_URL, headers=headers, data=data)

    # os.remove(link)

    res = json.loads(response.content.decode("utf-8"))
    print(res)
    return res["text"]


def speech_recognition_EG_ar(audio_uri: str):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=audio_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="ar-EG",
        enable_word_time_offsets=True,
        enable_speaker_diarization=True,
    )
    operation = client.long_running_recognize(config=config, audio=audio)
    gresponse = operation.result(timeout=10000)
    # print(gresponse.results[0])
    text = ""
    # for result in gresponse.results:
    print(gresponse.results)
    text += gresponse.results[0].alternatives[0].transcript

    return text


def openai_analysis(system, prompt):
    openai.api_key = "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"

    result = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        # temperature=0,
        # presence_penalty=0.48
    )

    return result["choices"][0]["message"]["content"]


def summarize_MA_ar(text, format, context):

    prompt = """
    
    
    """

    pass


def qa_MA_ar(text, question, context, format):

    system = """
    You are a helpful question answering assistant for Moroccan Arabic (Darija) audios. 

    you get the {transcript} which is the written text from the audio in Darija.

    you get the {question} which is what you should do with the transcript of the the audio

    you get the {context} which is additional information you need in order to answer a question.

    you get {format} which is the format your answers need absolutely to be in.

    if you are not sure, answer that you don't know
    
    """

    prompt = f"""
    transcript: {text}
    
    question: {question}

    context: {context}

    format: {format}
    
    """

    result = openai_analysis(system, prompt)

    print(result)

    return result


def qa_EG_ar(text, question, context, format):

    system = """
    You are a helpful question answering assistant for Egyptian Arabic audios. 

    you get the {transcript} which is the written text from the audio in Darija.

    you get the {question} which is what you should do with the transcript of the the audio

    you get the {context} which is additional information you need in order to answer a question.

    you get {format} which is the format your answers need absolutely to be in.

    if you are not sure, answer that you don't know
    
    """

    prompt = f"""
    transcript: {text}
    
    question: {question}

    context: {context}

    format: {format}
    
    """

    result = openai_analysis(system, prompt)

    print(result)

    return result


def upload_audio(microphone, file_upload, language):
    warn_output = ""
    if (microphone is not None) and (file_upload is not None):
        warn_output = (
            "WARNING: You've uploaded an audio file and used the microphone. "
            "The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
        )

    elif (microphone is None) and (file_upload is None):
        raise gr.Error(
            "You have to either use the microphone or upload an audio file")

    file = microphone if microphone is not None else file_upload

    link = preprocess_audio(file)

    gs_url = upload_audio_cloud(link)

    uri = gs_url.replace("https://storage.googleapis.com/", "gs://")

    if language == "Moroccan Dialect":
        text = speech_recognition_MA_ar(link)

    elif language == "Egyptian Dialect":
        text = speech_recognition_EG_ar(uri)

    else:
        text = "Language Not Defined"

    return text, "✅ Audio Successfuly Loaded"


def download_audio(audio_link, language):

    if audio_link.startswith("https://www.youtube.com/"):
        link = video_download(audio_link)

    else:
        file = audio_download(audio_link)

        link = preprocess_audio(file)

    print(link)

    gs_url = upload_audio_cloud(link)

    uri = gs_url.replace("https://storage.googleapis.com/", "gs://")

    if language == "Moroccan Dialect":
        text = speech_recognition_MA_ar(link)

    elif language == "Egyptian Dialect":
        text = speech_recognition_EG_ar(uri)
    else:
        text = "Language Not Defined"

    return text, "✅ Successfuly Loaded"


def do_all(file_upload, question, context, format, language):
    transcript, _ = upload_audio(None, file_upload, language)
    output = interpret(transcript, question, context, format, language)
    return transcript, output

def do_all_link(audio_link, question, context, format, language):
    transcript, _ = download_audio(audio_link, language)
    output = interpret(transcript, question, context, format, language)
    return transcript, output


def interpret(transcript, question, context, format, language):
    if language == "Moroccan Dialect":
        output = qa_MA_ar(transcript, question, context, format)

    elif language == "Egyptian Dialect":
        output = qa_EG_ar(transcript, question, context, format)

    else:
        output = "Language Not Defined"

    return output


Title = '<img src="https://upload-image-jshop.s3.eu-west-3.amazonaws.com/annaX-2.png" alt="Logo" width="400">'

Description = 'Introducing **Annarabic X**: Unlock the power to understand and transform Arabic audios like never before. With cutting-edge technology and our customized large language model (LLM), Annarabic X empowers you to understand and mold audio content in various Arabic dialects effortlessly. Seamlessly tailor the output format to suit your specific needs. Experience the future of Arabic audio understanding with Annarabic X.'

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(Title)
    gr.Markdown(Description)
    with gr.Tab(label="Upload Audio"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    record = gr.Audio(source="microphone", type="filepath")
                    upload = gr.Audio(source="upload", type="filepath")

                language = gr.Radio(
                    ["Moroccan Dialect", "Egyptian Dialect"], label="Language")
                transcript = gr.Textbox(
                    lines=7, placeholder="", label="Transcript", visible=False)
                fake_transcript = gr.Label(
                    value="🔥 Record or Upload an Audio", label="Status")
                upload_btn = gr.Button(value="Upload")

        with gr.Row():
            with gr.Column():
                output = gr.Textbox(lines=7, placeholder="", label="Output")
            with gr.Column():
                question = gr.Textbox(
                    lines=1, placeholder="", label="Action/Question")
                context = gr.Textbox(lines=3, placeholder="", label="Context")
                format = gr.Textbox(lines=1, placeholder="", label="Format")
                generate_btn = gr.Button(value="Generate")

                upload_btn.click(
                    upload_audio,
                    inputs=[
                        record,
                        upload,
                        language,
                    ],
                    outputs=[transcript, fake_transcript],
                )

                generate_btn.click(
                    interpret,
                    inputs=[
                        transcript,
                        question,
                        context,
                        format,
                        language,
                    ],
                    outputs=[output],
                )
        with gr.Row():
            ex = gr.Examples([
                [
                    "hello.wav", "What is happening", "no context", "no format", "nothing",
                ],
                [
                    "awb.mp4", "What is the problem of the customer?", "This is a call center recording of a bank with the client. The client recording quality is mediocre so try to deduce the problem trom the agent's clear voice", "JSON: problem", "Moroccan Dialect",
                ],
                [
                    "hello.wav", "What is happening", "no context", "no format", "nothing",
                ],
            ],
                fn=do_all, inputs=[
                upload,
                question,
                context,
                format,
                language,
            ], outputs=[transcript, output], cache_examples=True)

    with gr.Tab(label="Audio Link"):
        with gr.Row():
            with gr.Column():
                audio = gr.Textbox(lines=1, placeholder="",
                                   label="Link to your Audio")
                language = gr.Radio(
                    ["Moroccan Dialect", "Egyptian Dialect"], label="Language")
                transcript2 = gr.Textbox(
                    lines=7, placeholder="", label="Transcript", visible=False)
                fake_transcript2 = gr.Label(
                    value="🔥 Enter the Link to your Audio or Video", label="Status")
                download_btn = gr.Button(value="Download")
        with gr.Row():
            with gr.Column():
                output2 = gr.Textbox(lines=7, placeholder="", label="Output")
            with gr.Column():
                question = gr.Textbox(
                    lines=1, placeholder="", label="Action/Question")
                context = gr.Textbox(lines=3, placeholder="", label="Context")
                format = gr.Textbox(lines=1, placeholder="", label="Format")
                generate_btn = gr.Button(value="Generate")

                download_btn.click(
                    download_audio,
                    inputs=[
                        audio,
                        language,
                    ],
                    outputs=[transcript2, fake_transcript2],
                )

                generate_btn.click(
                    interpret,
                    inputs=[
                        transcript2,
                        question,
                        context,
                        format,
                        language
                    ],
                    outputs=[output2],
                )
        with gr.Row():
            ex = gr.Examples([
                [
                    "hello.wav", "What is happening", "no context", "no format", "nothing",
                ],
                [
                    "hello.wav", "What is happening", "no context", "no format", "nothing",
                ],
                [
                    "hello.wav", "What is happening", "no context", "no format", "nothing",
                ],
            ],
                fn=do_all_link, inputs=[
                audio,
                question,
                context,
                format,
                language,
            ], outputs=[transcript2, output2], cache_examples=True)


demo.launch(debug=True, show_api=False,
            server_port=3000, favicon_path="favicon.ico")
