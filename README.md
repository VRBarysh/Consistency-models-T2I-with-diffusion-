# Consistency-models-T2I-with-diffusion-
In this project we try to accelerate inference for the Stable Diffusion 1.5 "text to image" model. This model can produce those beautiful images of a "sad puppy with large eyes" in 50 steps: ![SD1.5 50 step sample](result_images/SD_50steps.jpg) 

However, generating those takes noticeable time. If we specify just 4 steps, image quality becomes clearly unacceptable: ![SD1.5 4 step sample](result_images/SD_4steps.jpg) 

In this project we present 3 slightly different approaches to improve 4 step generation quality and we end up with very nice results like those: ![Multi-boundary consistensy distillation](result_images/MBCD.jpg)

Our 4 final step model produces nice images for other prompts too: ![Multi-boundary consistensy distillation samples for other prompts](result_images/MBCD_other_prompts.jpg) 
