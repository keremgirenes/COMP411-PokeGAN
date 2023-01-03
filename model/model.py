import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

## DATASET

class PokemonDataset(Dataset):
  def __init__(self, root_dir, transform=None):
     # Navigate to the parent directory
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    # Access the "datasets" folder
    self.root_dir = os.path.join(parent_dir, "datasets", root_dir)
    self.filenames = os.listdir(root_dir)

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    # Load the image
    image = Image.open(os.path.join(self.root_dir, self.filenames[idx]))

    # Apply the transform, if specified
    if self.transform:
      image = self.transform(image)

    # Return the image
    return image

## MODEL COMPONENTS

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=64, num_blocks=6):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels*2, hidden_channels*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels*4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels*4, hidden_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels*2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_channels=64, num_blocks=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        for _ in range(num_blocks-1):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(hidden_channels*2))
            layers.append(nn.LeakyReLU(0.01))
            hidden_channels *= 2
        layers.append(nn.Conv2d(hidden_channels, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


## TRAINING

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the hyperparameters
batch_size = 64
lr = 0.0002
num_epochs = 200

# Create the generator and discriminator
generator = Generator(input_channels=3, output_channels=3).to(device)
discriminator = Discriminator(input_channels=3).to(device)

# Create the optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Create the criterion
criterion = nn.BCEWithLogitsLoss()

# Set the fixed input for testing
fixed_input = torch.randn(batch_size, 3, 256, 256).to(device)

# Load the datasets
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training datasets
train_fire_dataset = PokemonDataset("dataset/trainFire", transform=transform)
train_electric_dataset = PokemonDataset("dataset/trainElectric", transform=transform)

# Load the test datasets
test_fire_dataset = PokemonDataset("dataset/testFire", transform=transform)
test_electric_dataset = PokemonDataset("dataset/testElectric", transform=transform)

# Create the dataloaders
train_fire_dataloader = DataLoader(train_fire_dataset, batch_size=batch_size, shuffle=True)
test_fire_dataloader = DataLoader(test_fire_dataset, batch_size=batch_size, shuffle=False)
train_electric_dataloader = DataLoader(train_electric_dataset, batch_size=batch_size, shuffle=True)
test_electric_dataloader = DataLoader(test_electric_dataset, batch_size=batch_size, shuffle=False)

# Train the model
for epoch in range(num_epochs):
  # Shuffle the datasets
  train_fire_dataset.shuffle()
  train_electric_dataset.shuffle()

  # Concatenate the datasets
  train_dataset = torch.utils.data.ConcatDataset([train_fire_dataset, train_electric_dataset])

  # Create a dataloader for the concatenated dataset
  dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  
  for i, (real_images, _) in enumerate(dataloader):
    # Move the data to the device
    real_images = real_images.to(device)

    # Train the discriminator
    d_optimizer.zero_grad()

    # Compute the loss for real images
    real_logits = discriminator(real_images)
    real_loss = criterion(real_logits, torch.ones_like(real_logits))
    real_loss.backward()

    # Generate fake images
    z = torch.randn(batch_size, 3, 256, 256).to(device)
    fake_images = generator(z)

    # Compute the loss for fake images
    fake_logits = discriminator(fake_images)
    fake_loss = criterion(fake_logits, torch.zeros_like(fake_logits))
    fake_loss.backward()

    # Update the discriminator
    d_optimizer.step()

    # Train the generator
    g_optimizer.zero_grad()

    # Generate fake images
    z = torch.randn(batch_size, 3, 256, 256).to(device)
    fake_images = generator(z)

    # Compute the loss for the generator
    fake_logits = discriminator(fake_images)
    g_loss = criterion(fake_logits, torch.ones_like(fake_logits))

    # Compute the cycle consistency loss
    reconstructed_images = generator(fake_images)
    cycle_loss = torch.mean(torch.abs(real_images - reconstructed_images))

    # Total loss
    total_loss = g_loss + cycle_loss
    total_loss.backward()

    # Update the generator
    g_optimizer.step()

    # Print the losses
    print(f"Epoch: {epoch}, D: {real_loss + fake_loss:.4f}, G: {g_loss:.4f}, Cycle: {cycle_loss:.4f}")

    # Generate some samples
    if epoch % 10 == 0:
        with torch.no_grad():
            samples = generator(fixed_input)
        tv.utils.save_image(samples, f"samples/{epoch}.png", nrow=8, normalize=True)


## TESTING

# Set the model to evaluation mode
generator.eval()
discriminator.eval()

# Create a dataloader for the testFire data
test_fire_dataloader = DataLoader(test_fire_dataset, batch_size=batch_size, shuffle=False)

# Test the model on the testFire data
with torch.no_grad():
  for i, (real_images, _) in enumerate(test_fire_dataloader):
    # Move the data to the device
    real_images = real_images.to(device)

    # Generate fake images
    z = torch.randn(batch_size, 3, 256, 256).to(device)
    fake_images = generator(z)

    # Compute the losses
    real_logits = discriminator(real_images)
    fake_logits = discriminator(fake_images)
    real_loss = criterion(real_logits, torch.ones_like(real_logits))
    fake_loss = criterion(fake_logits, torch.zeros_like(fake_logits))

    # Print the losses
    print(f"Real loss (testFire): {real_loss:.4f}, Fake loss (testFire): {fake_loss:.4f}")

# Create a dataloader for the testElectric data
test_electric_dataloader = DataLoader(test_electric_dataset, batch_size=batch_size, shuffle=False)

# Test the model on the testElectric data
with torch.no_grad():
  for i, (real_images, _) in enumerate(test_electric_dataloader):
    # Move the data to the device
    real_images = real_images.to(device)

    # Generate fake images
    z = torch.randn(batch_size, 3, 256, 256).to(device)
    fake_images = generator(z)

    # Compute the losses
    real_logits = discriminator(real_images)
    fake_logits = discriminator(fake_images)
    real_loss = criterion(real_logits, torch.ones_like(real_logits))
    fake_loss = criterion(fake_logits, torch.zeros_like(fake_logits))

    # Print the losses
    print(f"Real loss (testElectric): {real_loss:.4f}, Fake loss (testElectric): {fake_loss:.4f}")