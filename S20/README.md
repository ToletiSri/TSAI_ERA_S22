# The School of AI - ERA(Extensive & Reimagined AI Program) - Assignment 20

This folder consists of Assignment-20 from ERA course offered by - TSAI(The school of AI). 
Follow https://theschoolof.ai/ for more updates on TSAI

Assignment-20
Read the textual inversion section on this [Link](https://github.com/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb) . There's a mention of a "community-created SD concepts library" and a download of the learned_embeds.bin file. There is also a mention of "blue_loss" in the Guidance Section.

select 5 different styles of your choice and show output for the same prompt using these 5 different styles. Remember the seeds as you'll use them later. Keep seeds different for each 5 types.
now apply your own variant of "blue_loss" (it cannot be red, green, or blue loss) on the same prompts with each concept library and store the results. 
Convert this into an app on Huggingface (and share it on LinkedIn to get 500 extra points)

### Implementation and Results:
Used 4 different styles of text inversions for a puppy. - madubani art style, line style, pokymon style and concept-art style. (Refer notebook for images)

Applied saturation loss to pokemon style puppy. (tried converting RGB image to HSV image to fetch saturation, but, this resulted in loss of gradients. So, saturations calculated using formula)
We see that the custom saturation loss function nudges the image to have less saturation while creating the image.
(Results in notebook)

Implemented a simple gradio interface on higgingface: [TBD]
HF Link: https://huggingface.co/spaces/ToletiSri/TSAI_S20







