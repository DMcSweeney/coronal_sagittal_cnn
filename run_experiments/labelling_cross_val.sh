echo 'Running cross-validation experiment for vertebral body labelling'

# for f in {2,3,4}; do
f=1
echo "FOLD ${f} as test set..."
python ../LABELLER.py --root_dir /data/PAB_data/vert_labelling/ \
                    --output_dir /data/PAB_data/vert_labelling/q${f}/ \
                    --fold ${f} \
                    --mode Training

# python ../LABELLER.py --root_dir /data/PAB_data/vert_labelling/ \
#                     --output_dir /data/PAB_data/vert_labelling/q${f}/ \
#                     --fold ${f} \
#                     --mode Inference
                    
#done
