from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate 
from langchain_openai import ChatOpenAI
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
from datasets import load_dataset
import requests
import os
import streamlit as st
import torch
import soundfile as sf

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUBHUB_API_TOKEN")

#img2text
def img2text(url):
    image_to_text = pipeline("image-to-text", model = "Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]["generated_text"]

    print(text)
    return text

#llm
def generate_story(scenario):
    template="""
    You are a srory teller;
    You can generate a short story based on a simple narrative, the story should be no more than 40 words;
    
    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template,input_variables=["scenario"])

    story_llm=LLMChain(llm=ChatOpenAI(
            model_name="gpt-3.5-turbo",temperature=1),prompt=prompt, verbose=True)
    
    story=story_llm.predict(scenario=scenario, max_new_tokens=5000)
    print(story)
    return story

# text to speech
def text2speech(story_text):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # Prepare the input text
    inputs = processor(text=story_text, return_tensors="pt")

    # Load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    # Generate speech
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    # Save the generated speech to a WAV file
    sf.write("story_speech.wav", speech.numpy(), samplerate=16000)
    print("The generated speech has been saved to 'story_speech.wav'")

# Generate a story using the generate_story function
scenario = img2text("photo.png")
story = generate_story(scenario)

# Convert the generated story text to speech
text2speech(story)

def main():
    st.set_page_config(page_title = "AI story Teller", page_icon ="🤖")

    st.header("We turn images to story!")
    upload_file = st.file_uploader("Choose an image...", type = 'jpg')  #uploads image

    if upload_file is not None:
        print(upload_file)
        binary_data = upload_file.getvalue()
        
        # save image
        with open (upload_file.name, 'wb') as f:
            f.write(binary_data)
        st.image(upload_file, caption = "Image Uploaded", use_column_width = True) # display image

        scenario = img2text(upload_file.name) #text2image
        story = generate_story(scenario) # create a story
        text2speech(story) # convert generated text to audio

        # display scenario and story
        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        
        # display the audio - people can listen
        st.audio("audio.flac")

# the main
if __name__ == "__main__":
    main()

scenario = img2text("photo.png")
story = generate_story(scenario)
text2speech(story)

