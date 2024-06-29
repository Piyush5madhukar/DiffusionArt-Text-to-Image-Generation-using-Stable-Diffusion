#Text to image generation using  stable diffusion
![Screenshot (607)](https://github.com/Piyush5madhukar/DiffusionArt-Text-to-Image-Generation-using-Stable-Diffusion/assets/105438331/87f344d0-1fdf-4b1e-af68-89c9177e9559)


STEPS
1)import dependencies
2)Define Configuration Class
-to store configuration settings for device, random number generator, image generation parameters, and prompt generation parameters.
3)Instantiate Configuration Class
4)Load Pre-Trained Model
-stable diffusion model
-prompt generating model
5)define image generation function
6)generate image


libraries
-diffusers-to build and deploy diffusion models
-transformers  developed by Hugging Face that provides general-purpose architectures for Natural Language Understanding (NLU) and Natural Language Generation (NLG) with thousands of pretrained models.

-pathlib: module, provides classes to work with filesystem paths in an object-oriented way.
-tqdm: This is a module that provides fast, extensible progress bars for loops and other iterables in Python.

-torch:  open-source ML library, used for applications such as computer vision and natural language processing
-pandas: This is an open-source data analysis and manipulation library for Python.



-StableDiffusionPipeline(CLASS): This is a class within the diffusers library that provides functionalities for generating images using a stable diffusion model.

-pipeline: This is a high-level API within the transformers library that provides easy-to-use interfaces for various NLP tasks.

#model used is-stabilityai/stable-diffusion-2
#prompt generation model used-gpt2

#The term "guidance scale"  refers to a parameter that controls the trade-off between adherence to the input prompt and the diversity or creativity of the generated images.


###
-torch_dtype: This argument specifies the data type for the tensors used in the model.

-torch.float16: This specifies that the model should use 16-bit floating-point precision for its tensors, which can reduce memory usage and speed up computations on compatible hardware.



#revision: This argument specifies the version of the model to use
