import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel
import os

# 原图路径
path_ori = "/root/ConceptExpress/ckpts/65/65_con_kl_1000/attention/0-step/final_masked3.png"
# 图片文件夹路径
folder_path = "/root/ConceptExpress/output_65/65_con_kl_1000_asset_3/a-photo-of-<asset3>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# 加载原图并获取特征
image_original = Image.open(path_ori)
with torch.no_grad():
    inputs_original = processor(images=image_original, return_tensors="pt").to(device)
    image_features_original = model.get_image_features(**inputs_original)

# 初始化相似度列表
similarities = []

# 遍历文件夹中的所有图片
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):  # 确保是PNG格式的图片
        # 图片路径
        path_gen = os.path.join(folder_path, filename)
        # 加载图片并获取特征
        image_generated = Image.open(path_gen)
        with torch.no_grad():
            inputs_generated = processor(images=image_generated, return_tensors="pt").to(device)
            image_features_generated = model.get_image_features(**inputs_generated)
        
        # 计算相似度
        cos = torch.nn.CosineSimilarity(dim=0)
        sim = cos(image_features_generated[0], image_features_original[0]).item()
        sim = (sim + 1) / 2  # 将相似度标准化到[0, 1]
        similarities.append(sim)

# 计算平均相似度
average_similarity = sum(similarities) / len(similarities)
print("Average Similarity: ", average_similarity)