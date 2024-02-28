# import required libraries
from dotenv import load_dotenv 
load_dotenv()  # to load all the env variables

import gradio as gr
import os
import google.generativeai as genai
from PIL import Image
import numpy as np

# Configure api key
genai.configure(api_key= os.getenv("GOOGLE_API_KEY"))

# creating function to load gemini pro model
model = genai.GenerativeModel("gemini-pro")
model1 = genai.GenerativeModel("gemini-pro-vision")

def get_response(text_input, image_input):
    # Convert NumPy array to PIL Image
    if isinstance(image_input, np.ndarray):
        image_input = Image.fromarray(np.uint8(image_input))

    if text_input == '' and image_input is None:
        return "Please provide a text and an image."

    if text_input != '' and image_input is not None:
        response = model1.generate_content([text_input, image_input])
    elif image_input is None:
        response = model.generate_content(text_input)
    elif text_input == '':
        response = model1.generate_content(image_input)

    if response is not None:
        return response.text
    else:
        return "No response available"
      
# Correct the function name in the gr.Interface
demo = gr.Interface(fn=get_response, inputs=['text', gr.Image()], outputs="text", title='Artificial Intelligence API')
demo.launch(debug=True, share=True)