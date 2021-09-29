echo 'Running cross-validation experiment for vertebral body labelling'

for f in {1,2,3,4}; do
echo "FOLD ${f} as test set..."
python ../LABELLER.py --root_dir /data/PAB_data/vert_labelling/ \
                    --output_dir /data/PAB_data/vert_labelling/q${f}/ \
                    --fold ${f} \
                    --mode Training

python ../LABELLER.py --root_dir /data/PAB_data/vert_labelling/ \
                    --output_dir /data/PAB_data/vert_labelling/q${f}/ \
                    --fold ${f} \
                    --mode Inference
                    
done