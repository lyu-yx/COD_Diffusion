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
   1. [x] arch constrcut 03.05
      1. [x] arch design done 02.28
      2. [x] edge deducing module
      3. [x] cross domain feature fusion(CDFF) 03.01
      4. [x] prior guided feature refinement(PGFR)  03.01
      5. [x] cat with Unet 03.05
   2. [x] success train 03.07
   3. [ ] hyper parameter adjust
      1. [ ] training part
         1. [ ] learning rate
         2. [ ] meta parameter in loss
      
      2. [ ] sampling part
         1. [ ] staple number
         2. [ ] diffusion step
   4. [ ] validation arch effectiveness

train meta
| lr | loss meta |
| --- |--------- |
| 1e-4 | 0.002   |
| 1e-4 | 0.001   |
| 1e-4 | 0.0005  |
| 5e-5 | 0.002   |
| 5e-5 | 0.001   |
| 5e-5 | 0.0005  |

sample meta
| staple num | Step | dpmsolver |
|       ---  |----  |     ---   |
| 5          | 50   |           |
| 5          | 100  |           |
| 3          | 50   |           |
| 3          | 100  |           |
| 10         | 50   |           |
| 10         | 100  |           |