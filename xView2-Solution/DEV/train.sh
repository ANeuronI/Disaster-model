# Step 0
python convert_masks.py

# # Step 1
# python fit_predict.py --seed 330 -dd ./DATA -x fold0_resnet34_unet_v2 --model resnet34_unet_v2 --batch-size 32 --epochs 150 --learning-rate 0.0001 --criterion weighted_ce 1.0 -w 4 -a medium --fp16 --fold 0 --scheduler cos -wd 0.0001 --only-buildings --crops --post-transform

python predict_37_weighted.py --activation ensemble -w 4 -p dominantv2

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m streamlit run DEV/App_Inference.py