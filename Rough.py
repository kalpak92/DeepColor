import glob

import torch
import numpy as np
from PIL import Image

a = np.array([
    [[[1, 2, 4],
      [2, 5, 4]]],
    [[[1, 2, 4],
      [2, 7, 4]]],
    [[[1, 2, 4],
      [2, 23, 4]]]
])
t_a = torch.from_numpy(a * 1.0)

b = np.array([
    [[[1, 2, 4],
      [2, 15, 4]]],
    [[[1, 2, 4],
      [2, 11, 4]]],
    [[[1, 2, 4],
      [2, 11, 4]]]
])
t_b = torch.from_numpy(b * 1.0)

# print("-----")
# print("t_a size: ", t_a.size())
# t_a_mean = t_a.mean(dim=(2, 3))
# print("t_a_mean_size: ", t_a_mean.size())
# print("t_a_mean: ", t_a_mean)
# print("------")
# print("t_b size: ", t_b.size())
# t_b_mean = t_b.mean(dim=(2, 3))
# print("t_b_mean_size: ", t_b_mean.size())
# print("t_b_mean: ", t_b_mean)
# print("-----")
# t_orig = torch.cat([t_a_mean, t_b_mean], dim=1)
# print("t_orig_size: ", t_orig.size())
# print("t_orig: ", t_orig)

# t_orig = torch.cat([t_a, t_b], dim=1)
# print(t_orig.size())
#
# print(t_orig)


titles = ["GrayScale Image", "Original Image", "Reconstructed Image"]
print(titles*3)


