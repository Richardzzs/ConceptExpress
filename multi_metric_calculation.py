import torch
from PIL import Image
from transformers import AutoProcessor, AutoImageProcessor, CLIPModel, AutoModel
import os
import torch.nn
from torch import nn

# 原图路径
path_ori = "/root/ConceptExpress/SAM/03/asset1/sam_generated_masked_0.png"
# 图片文件夹路径
folder_path = "/root/ConceptExpress/output_03/03_con_kl_assset_1/a-photo-of-<asset1>-on-the-road"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载原图并获取特征
image_original = Image.open(path_ori)

# 1. Calculate CLIP Compositional Similarity

processor_1 = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model_1 = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

with torch.no_grad():
    inputs_original_1 = processor_1(images=image_original, return_tensors="pt").to(device)
    image_features_original_1 = model_1.get_image_features(**inputs_original_1)

# 初始化相似度列表
similarities_1 = []

# 2. Calculate DINO Compositional Similarity

processor_2 = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model_2 = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

with torch.no_grad():
    inputs_original_2 = processor_2(images=image_original, return_tensors="pt").to(device)
    outputs_original_2 = model_2(**inputs_original_2)
    image_features_original_sample_2 = outputs_original_2.last_hidden_state
    image_features_original_2 = image_features_original_sample_2.mean(dim=1)

# 初始化相似度列表
similarities_2 = []

# 遍历文件夹中的所有图片
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):  # 确保是PNG格式的图片
        # 图片路径
        path_gen = os.path.join(folder_path, filename)
        # 加载图片并获取特征
        image_generated = Image.open(path_gen)

        # 1. Calculate CLIP Compositional Similarity
        with torch.no_grad():
            inputs_generated_1 = processor_1(images=image_generated, return_tensors="pt").to(device)
            image_features_generated_1 = model_1.get_image_features(**inputs_generated_1)
        
        # 2. Calculate DINO Compositional Similarity
        with torch.no_grad():
            inputs_generated_2 = processor_2(images=image_generated, return_tensors="pt").to(device)
            outputs_generated_2 = model_2(**inputs_generated_2)
            image_features_generated_sample_2 = outputs_generated_2.last_hidden_state
            image_features_generated_2 = image_features_generated_sample_2.mean(dim=1)

        # 计算相似度
        cos = torch.nn.CosineSimilarity(dim=0)
        sim_1 = cos(image_features_generated_1[0], image_features_original_1[0]).item()
        sim_1 = (sim_1 + 1) / 2  # 将相似度标准化到[0, 1]
        similarities_1.append(sim_1)

        sim_2 = cos(image_features_generated_2[0], image_features_original_2[0]).item()
        sim_2 = (sim_2+1) / 2
        similarities_2.append(sim_2)

# 计算平均相似度
average_similarity_1 = sum(similarities_1) / len(similarities_1)
average_similarity_2 = sum(similarities_2) / len(similarities_2)

print("CLIP Average Similarity: ", average_similarity_1)
print("DINO Average Similarity: ", average_similarity_2)