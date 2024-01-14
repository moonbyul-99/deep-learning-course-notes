import numpy as np
import torch 
import torch.nn as nn
import torch.functional as f

loss = nn.CrossEntropyLoss()

for i, batch in enumerate(dataloader):