import torch
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images)

import logging
logging.basicConfig(level=logging.INFO)

model = BigGAN.from_pretrained('biggan-deep-128')
# model.load_state_dict(torch.load('fine_tuned_biggan_weights.pth'))

truncation = 0.4
class_vector = one_hot_from_names([input('please input a word: ')], batch_size=1)
noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)

noise_vector = torch.from_numpy(noise_vector)
class_vector = torch.from_numpy(class_vector)

noise_vector = noise_vector.to('cuda')
class_vector = class_vector.to('cuda')
model.to('cuda')

with torch.no_grad():
    output = model(noise_vector, class_vector, truncation)

output = output.to('cpu')

save_as_images(output)