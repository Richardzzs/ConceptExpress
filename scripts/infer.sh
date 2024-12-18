python infer.py \
  --embed_path ckpts/69/learned_embeds_final.bin \
  --prompt "a photo of <asset0> on the beach" \
  --save_path output/69 \
  --seed 0

python infer.py \
  --embed_path ckpts/69/fusion/learned_embeds_final.bin \
  --prompt "a photo of <asset*>" \
  --save_path output_69/seed_0/69_fusion \
  --seed 0

python infer.py \
  --embed_path ckpts/83/fusion_new4/learned_embeds_final.bin \
  --prompt "a photo of <asset*a>" \
  --save_path output_83/83_fusion_new4* \
  --seed 0

python infer.py \
  --embed_path ckpts/69/69_con_ori_abs/learned_embeds_final.bin \
  --prompt "a photo of <asset0> on the beach" \
  --save_path output_69/seed_0/69_con_ori_abs_beach \
  --seed 0

python infer.py \
  --embed_path ckpts/69/69_con/learned_embeds_final.bin \
  --prompt "a photo of <asset0> and <asset1>" \
  --save_path output_69/seed_0/69_con_01 \
  --seed 0

python infer.py \
  --embed_path ckpts/69/69_con_kl_1000/learned_embeds_final.bin \
  --prompt "a photo of <asset0> on the beach" \
  --save_path output_69/seed_0/69_con_kl_1000_beach \
  --seed 0

python infer.py \
  --embed_path ckpts/03/03_con_abs/learned_embeds_final.bin \
  --prompt "a photo of <asset0> on the road" \
  --save_path output_03/03_con_abs_asset_0 \
  --seed 0

python infer.py \
  --embed_path ckpts/03/03_con_ori_abs/learned_embeds_final.bin \
  --prompt "a photo of <asset0> on the road" \
  --save_path output_03/03_con_ori_abs_assset_0 \
  --seed 0

python infer.py \
  --embed_path ckpts/65/65_con_kl_1000/learned_embeds_final.bin \
  --prompt "a photo of <asset0>" \
  --save_path output_65/65_con_kl_1000_asset_0 \
  --seed 0

python infer.py \
  --embed_path ckpts/03/03_con_ori_abs/learned_embeds_final.bin \
  --prompt "a photo of <asset0> and <asset1>" \
  --save_path output_03/03_con_ori_abs_01 \
  --seed 0