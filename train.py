import torch 

if torch.cuda.is_available():
    print("Available",torch.cuda.device)
    print(torch.cuda.device_count)
else:
    print("Not Available")
