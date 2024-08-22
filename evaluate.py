import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel
# import torch.nn

path_gen = "/root/ConceptExpress/output_65/65_con_kl_1000_asset_0/a-photo-of-<asset0>/0.png"
path_ori = "/root/ConceptExpress/ckpts/65/65_con_kl_1000/attention/0-step/final_masked3.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

image_generated = Image.open(path_gen)
with torch.no_grad():
    inputs_generated = processor(images=image_generated, return_tensors="pt").to(device)
    image_features_generated = model.get_image_features(**inputs_generated)

image_original = Image.open(path_ori)
with torch.no_grad():
    inputs_original = processor(images=image_original, return_tensors="pt").to(device)
    image_features_original = model.get_image_features(**inputs_original)

cos = torch.nn.CosineSimilarity(dim=0)
sim = cos(image_features_generated[0], image_features_original[0]).item()
sim = (sim+1)/2
print("Similarity: ", sim)