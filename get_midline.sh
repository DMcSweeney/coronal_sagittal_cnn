
echo 'Extracting sagittal midlines for each fold'

#for f in {1,2,3,4}; do
f=1
echo "FOLD ${f} as test set..." 
mkdir /data/PAB_data/images_sagittal/sagittal_midline/q${f}/
python extract_midline.py --root_dir /data/PAB_data/midline_data/ \
                        --volume_dir /data/PAB_data/volume_folds/ \
                        --output_dir /data/PAB_data/images_sagittal/sagittal_midline/q${f}/ \
                        --fold ${f} 
#done