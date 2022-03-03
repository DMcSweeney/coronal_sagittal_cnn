echo "Training midline finder with Quantize-aware training"

python ../find_midline_qat.py --root_dir /data/PAB_data/midline_data/ \
                        --output_dir /data/PAB_data/midline_data/q1/ \
                        --fold 1 \
                        --mode Training