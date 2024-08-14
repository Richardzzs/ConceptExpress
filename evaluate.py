import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel
# import torch.nn

path_ori = "./uce_images/03/img.jpg"
path_gen = "./output_03/03_con_asset_1/a-photo-of-<asset1>-on-the-road.png"

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