import torch
import numpy as np
from torch.autograd import Variable
m = np.zeros([4,4])
m_match = torch.from_numpy(m) == 0
m_cost = torch.Tensor(m) == 1
Imatch = Variable(m_match)
Icost = Variable(m_cost)
print(Imatch)
print(Icost)
