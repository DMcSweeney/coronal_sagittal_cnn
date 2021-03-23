15/03/2020 <br>

Latest spine models, post-estro reset. <br>
All previous models (shape + appearance & CNNs) are in `/home/donal/PhD/initial_spines/CT_models/` <br>
CNN attempts  = multi-class segmentation + regression with DSNT. <br>


<h2> Workflow </h2>
  1. Inputs need to be nifty

`python projections.py` should make MIP, STD and AVG projections - need to change outputs + lines 177-179 depending on coronal or sagittal. <br>

`prepapre_data.ipynb` - notebook for preparing annotations + images for training. (__Will need to be changed__) <br>

`python trainTestValSplit.py`- self-explanatory <br>

`python train.py` - doesn't work yet and will need to be rewritten for predicting a single level