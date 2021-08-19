import torch
import os 

dir = "./"

path = "./model_best.pt"

weight = torch.load(path)

new_dict = {}

for k, v in weight.items():
    print(k)
    # new_name = str(k)[6:]
    if k.split('.')[0] != 'H2A2SR':
        new_dict[k] =  v

torch.save(new_dict, os.path.join(dir,'model_best_1.pt'))

# for k, v in new_dict.items():
#     print(k)