import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from pytorch_pretrained_biggan import BigGAN

torch.manual_seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 1
batch_size = 8
learning_rate = 0.0002

data_dir = "cifar-10-batches-py"

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if image.ndim == 1:
            image = np.expand_dims(image, axis=0)  
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

train_data = []
train_labels = []
for i in range(1, 2):
    file_path = f'{data_dir}/data_batch_{i}'
    data = unpickle(file_path)
    train_data.append(data[b'data'])
    train_labels.append(data[b'labels'])

train_data = np.concatenate(train_data, axis=0)
train_labels = np.concatenate(train_labels, axis=0)

test_file_path = f'{data_dir}/test_batch'
data = unpickle(test_file_path)
test_data = data[b'data']
test_labels = data[b'labels']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  
])

trainset = CustomDataset(train_data, train_labels, transform=transform)
testset = CustomDataset(test_data, test_labels, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

biggan = BigGAN.from_pretrained('biggan-deep-128').to(device)

generator_optimizer = torch.optim.Adam(biggan.generator.parameters(), lr=learning_rate)

def generator_loss(fake_output):
    target = torch.ones_like(fake_output)
    target = target.expand_as(fake_output)
    return torch.nn.functional.mse_loss(fake_output, target)

def train_step(images):
    images = images.to(device)

    noise = torch.randn(images.size(0), 256, device=device)
    fake_images = biggan.generator(noise, truncation=0.4)

    gen_loss = generator_loss(fake_images)

    generator_optimizer.zero_grad()
    gen_loss.backward()
    generator_optimizer.step()

for epoch in range(epochs):
    for i, data in enumerate(trainloader, 0):
        print(f'epoch: {epoch} batch: {i}')
        inputs, _ = data
        train_step(inputs)

torch.save(biggan.state_dict(), 'fine_tuned_biggan_weights.pth')