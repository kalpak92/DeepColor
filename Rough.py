import torch
import numpy as np
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

print("-----")
print("t_a size: ", t_a.size())
# t_a_mean = t_a.mean(dim=(2, 3))
# print("t_a_mean_size: ", t_a_mean.size())
# print("t_a_mean: ", t_a_mean)
print("------")
print("t_b size: ", t_b.size())
# t_b_mean = t_b.mean(dim=(2, 3))
# print("t_b_mean_size: ", t_b_mean.size())
# print("t_b_mean: ", t_b_mean)
# print("-----")
# t_orig = torch.cat([t_a_mean, t_b_mean], dim=1)
# print("t_orig_size: ", t_orig.size())
# print("t_orig: ", t_orig)

t_orig = torch.cat([t_a, t_b], dim=1)
print(t_orig.size())

print(t_orig)



def print_util(augmented_dataset_batch):
    sample = next(iter(augmented_dataset_batch))
    l_channel, a_channel, b_channel = sample

    print("L channel shape: ", l_channel.shape)
    print("a_channel shape:", a_channel.shape)
    print("b_channel shape:", b_channel.shape)

    regressor = Regressor(in_channel=1, hidden_channel=3, out_dims=2,
                          train_mode="regressor")
    output_hat = regressor(l_channel)
    print(output_hat.size())

    print(output_hat)

    print("------")
    a_channel_mean = a_channel.mean(dim=(2, 3))
    # print("a_channel_mean_size: ", a_channel_mean.size())
    # print("a_channel_mean: ", a_channel_mean)
    # print("------")
    b_channel_mean = b_channel.mean(dim=(2, 3))
    # print("b_channel_mean_size: ", b_channel_mean.size())
    # print("b_channel_mean: ", b_channel_mean)
    # print("-----")
    a_b_orig = torch.cat([a_channel_mean, b_channel_mean], dim=1)
    print("t_orig_size: ", a_b_orig.size())
    print("t_orig: ", a_b_orig)


def print_util_1(augmented_dataset_batch, activation_function):
    sample = next(iter(augmented_dataset_batch))
    l_channel, a_channel, b_channel = sample

    print("L channel shape: ", l_channel.shape)
    print("a_channel shape:", a_channel.shape)
    print("b_channel shape:", b_channel.shape)

    colorizer = Colorizer(in_channel=3, hidden_channel=3,
                          out_channel=2,
                          activation_function=activation_function)
    output_hat = colorizer(l_channel)
    print(output_hat.size())
    print(output_hat)
    print(a_channel)
    # Utils().show_img(torchvision.utils.make_grid(l_channel))
    # Utils().show_img(torchvision.utils.make_grid(a_channel))
    # Utils().show_img(torchvision.utils.make_grid(b_channel))
