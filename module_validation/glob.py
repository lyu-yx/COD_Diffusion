from glob import glob
import os 

list1 = []
list2 = []
for pth in glob("G:\Code\COD_Diffusion\\results\COD10K_100_0.8_5_148000_LN_pgfr_out_pos\*"):
    list1.append(os.path.basename(pth))

for pth in glob("G:\Code\BUDG\dataset\TestDataset\COD10K\GT\*"):
    list2.append(os.path.basename(pth))



print(len(list1))
print(len(list2))   

for file in list2:
    if file not in list1:
        print(file)

