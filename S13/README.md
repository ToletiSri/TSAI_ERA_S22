# The School of AI - ERA(Extensive & Reimagined AI Program) - Assignment 13

This folder consists of Assignment-13 from ERA course offered by - TSAI(The school of AI). 
Follow https://theschoolof.ai/ for more updates on TSAI

Assignment-13
Move the given YOLO V3 code to PytorchLightning
Train the model to reach such that all of these are true:
- Class accuracy is more than 75%
- No Obj accuracy of more than 95%
- Object Accuracy of more than 70% (assuming you had to reduce the kernel numbers, else 80/98/78)
- Ideally trailed till 40 epochs

Add these training features:
- Add multi-resolution training - the code shared trains only on one resolution 416
- Add Implement Mosaic Augmentation only 75% of the times
- Train on float16

GradCam must be implemented.
Things that are allowed due to HW constraints:
- Change of batch size
- Change of resolution
- Change of OCP parameters

Once done:
Move the app to HuggingFace Spaces
- Allow custom upload of images
- Share some samples from the existing dataset
- Show the GradCAM output for the image that the user uploads as well as for the samples.
- Mention things like: classes that your model support link to the actual model

### RESULTS:

Train Accuracy:
Class accuracy is: 82.711205%
No obj accuracy is: 98.512054%
Obj accuracy is: 63.684509%

Test Accuracy:
Class accuracy is: 79.258514%
No obj accuracy is: 98.677887%
Obj accuracy is: 55.932331%


### Hugging Face - Spaces app:
https://huggingface.co/spaces/ToletiSri/TSAI_S13
