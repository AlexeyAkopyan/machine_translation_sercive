import torch
import torchtext

print(torch.__version__)
print(torchtext.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)