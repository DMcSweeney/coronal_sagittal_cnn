echo 'Running cross-validation experiment for vertebral body segmenter'

for f in {1,2,3,4}; do
    echo "FOLD ${f} as test set..." 
    python ../SEGMENTER.py --root_dir /data/PAB_data/midline_data/ \
                        --output_dir /data/PAB_data/midline_data/q${f}/ \
                        --fold ${f} \
                        --mode Training
    
    python ../SEGMENTER.py --root_dir /data/PAB_data/midline_data/ \
                        --output_dir /data/PAB_data/midline_data/q${f}/ \
                        --fold ${f} \
                        --mode Inference
        
done