echo 'Running cross-validation experiment for vertebral body segmenter'

# for f in {1,2,3,4}; do
f=1
echo "FOLD ${f} as test set..." 

python ../SEGMENTER.py --root_dir /data/PAB_data/vert_seg/ \
                    --output_dir /data/PAB_data/vert_seg/q${f}/ \
                    --fold ${f} \
                    --mode Training

python ../SEGMENTER.py --root_dir /data/PAB_data/vert_seg/ \
                    --output_dir /data/PAB_data/vert_seg/q${f}/ \
                    --fold ${f} \
                    --mode Inference
        
# done