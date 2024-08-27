import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


# with open('metastasis_critical_cases.csv', 'r') as f:
with open('metastasis-short-long-axis.csv', 'r') as f:
    lines = f.readlines()
critical_cases = [line.strip().split(',') for line in lines]
# sort by patient id, lambda x[0]
critical_cases.sort(key=lambda x: x[0])

pre_ct = None
root_dir = '/mnt/889cdd89-1094-48ae-b221-146ffe543605/gwd/dataset/RecT500'
LABEL_DIR = os.path.join(root_dir, 'label')
all_cases = []
for case in tqdm(critical_cases):
    if pre_ct != case[0]:
        pre_ct = case[0]
        label = sitk.ReadImage(os.path.join(LABEL_DIR, f'{case[0]}.nii.gz'))
        lbl_np = sitk.GetArrayFromImage(label)
        z_inds, *_ = np.where((lbl_np==6)|(lbl_np==7))
        z_min = max(0, z_inds.min() - 10)
    case[3] = str(int(case[3])+z_min)
    case[6] = str(int(case[6])+z_min)
    all_cases.append(','.join(case))


print('Writing to file: metastasis_critical_cases_rectified.csv')
with open('metastasis_all_cases_rectified.csv', 'w') as f:
# with open('metastasis_critical_cases_rectified.csv', 'w') as f:
    f.write('\n'.join(all_cases))
