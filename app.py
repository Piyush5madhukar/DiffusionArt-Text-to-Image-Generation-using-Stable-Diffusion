import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    @classmethod
    def get_generator(cls):
        return torch.Generator(cls.device).manual_seed(cls.seed)

    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9

CFG.generator = CFG.get_generator()

# Load the model
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='auth token'
)
image_gen_model = image_gen_model.to(CFG.device)

def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image

# Streamlit app
st.title("Text to Image Generator")

prompt = st.text_input("Enter text prompt:", value="A sunny beach with palm trees")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image = generate_image(prompt, image_gen_model)
        st.image(image, caption="Generated Image")

        # Create a download button
        image.save("generated_image.png")
        with open("generated_image.png", "rb") as file:
            btn = st.download_button(
                label="Download image",
                data=file,
                file_name="generated_image.png",
                mime="image/png"
            )
