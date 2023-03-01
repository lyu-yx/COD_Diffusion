1. [x] distribution train debug
   1. [x] train.py args modify 
   2. [x] dist_util.py modify
   3. [x] hpc try
   
2. [ ] inference debug
   1. [ ] dist infer(still figuring out)
   2. [x] multi infer results fusion (staple alg) 02.21
   3. [x] DPM solver 02.21

3.  [ ] auto-save and val during train
    1. [x] wandb save inference img
    2. [ ] integrate _val_single_img_ to training process(debuging)

4. [ ] arch for COD task
   1. [ ] arch constrcut
      1. [x] arch design done 02.28
      2. [x] edge deducing module
      3. [ ] cross domain feature fusion(CDFF)
      4. [ ] prior guided feature refinement(PGFR)
      5. [ ] cat with Unet
   2. [ ] success train 