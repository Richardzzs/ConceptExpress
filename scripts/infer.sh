python infer.py \
  --embed_path ckpts/69/learned_embeds_final.bin \
  --prompt "a photo of <asset0> on the beach" \
  --save_path output/69 \
  --seed 0

python infer.py \
  --embed_path ckpts/69/learned_embeds_final.bin \
  --prompt "a photo of <asset1> in the snow" \
  --save_path output/69 \
  --seed 0

python infer.py \
  --embed_path ckpts/03/03_con_abs/learned_embeds_final.bin \
  --prompt "a photo of <asset0> on the road" \
  --save_path output_03/03_con_abs_asset_0 \
  --seed 0

python infer.py \
  --embed_path ckpts/65/65_con/learned_embeds_final.bin \
  --prompt "a photo of <asset0>" \
  --save_path output_65/65_con_asset_0 \
  --seed 0