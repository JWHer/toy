import platform, torch
# /Users/jwher/miniconda3/envs/py39_native/bin/python
print(platform.platform())

CPU= False
device = "cpu" if CPU else torch.device("mps")
print("Device is : {}".format(device))
