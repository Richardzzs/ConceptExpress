import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel
# import torch.nn

path_ori = "./uce_images/69/img.jpg"
path_gen = "./ckpts/69/69_con_ori_abs/images/A-photo-of-<asset0>-and-<asset1>-step-500.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

image_generated = Image.open(path_ori)
with torch.no_grad():
    inputs_generated = processor(images=image_generated, return_tensors="pt").to(device)
    image_features_generated = model.get_image_features(**inputs_generated)


image_original = Image.open(path_gen)
with torch.no_grad():
    inputs_original = processor(images=image_original, return_tensors="pt").to(device)
    image_features_original = model.get_image_features(**inputs_original)

cos = torch.nn.CosineSimilarity(dim=0)
sim = cos(image_features_generated[0], image_features_original[0]).item()
sim = (sim+1)/2
print("Similarity: ", sim)