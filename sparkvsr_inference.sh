# Model Path (Update as needed)
# Stage 1
# MODEL_PATH="checkpoints/sparkvsr-s1/ckpt-10000-sft" 
# Stage 2
# MODEL_PATH="checkpoints/sparkvsr-s2/ckpt-500-sft" 

MODEL_PATH="checkpoints/SparkVSR" # 现在的名称

CUDA_VISIBLE_DEVICES=0 python sparkvsr_inference_script.py \
    --input_dir test_input/smoke_test_input \
    --model_path $MODEL_PATH \
    --output_path results/smoke_test \
    --is_vae_st \
    --ref_mode no_ref \
    --ref_prompt_mode fixed \
    --ref_guidance_scale 1.0 \
    --eval_metrics psnr,ssim,lpips,dists,clipiqa \
    --upscale 4 \
    --chunk_len 49 \
    --tile_size_hw 720 960
