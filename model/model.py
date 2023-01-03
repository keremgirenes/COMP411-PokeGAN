import torch
import torch.nn as nn

class Generator(nn.Module):
  def __init__(self, input_size, output_channels):
    super(Generator, self).__init__()
    # Define the encoder part of the generator
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(128)
    self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
    self.bn3 = nn.BatchNorm2d(256)
    # Define the transformation part of the generator
    self.res1 = ResidualBlock(256)
    self.res2 = ResidualBlock(256)
    self.res3 = ResidualBlock(256)
    self.res4 = ResidualBlock(256)
    self.res5 = ResidualBlock(256)
    self.res6 = ResidualBlock(256)
    self.res7 = ResidualBlock(256)
    self.res8 = ResidualBlock(256)
    self.res9 = ResidualBlock(256)
    # Define the decoder part of the generator
    self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
    self.bn4 = nn.BatchNorm2d(128)
    self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
    self.bn5 = nn.BatchNorm2d(64)
    self.deconv3 = nn.ConvTranspose2d(64, output_channels, kernel_size=7, stride=1, padding=3, bias=False)

  def forward(self, x):
    # Apply the encoder part of the generator
    x = self.conv1(x)
    x = self.bn1(x)
    x = nn.ReLU(inplace=True)(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = nn.ReLU(inplace=True)(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = nn.ReLU(inplace=True)(x)
    # Apply the transformation part of the generator
    x = self.res1(x)
    x = self.res2(x)
    x = self.res3(x)
    x = self.res4(x)
    x = self.res5(x)
    x = self.res6(x)
    x = self.res7(x)
    x = self.res8(x)
    x = self.res9(x)
    # Apply the decoder part of the generator
    x = self.deconv1(x)
    x = self.bn4(x)
    x = nn.ReLU(inplace=True)(x)
    x = self.deconv2(x)
    x = self.bn5(x)
    x = nn.ReLU(inplace=True)(x)
    x = self.deconv3(x)
    return x


class Discriminator(nn.Module):
  def __init__(self, input_channels):
    super(Discriminator, self).__init__()
    # Define the input layer
    self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1, bias=False)
    # Define the intermediate layers
    self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(128)
    self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
    self.bn3 = nn.BatchNorm2d(256)
    self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False)
    self.bn4 = nn.BatchNorm2d(512)
    # Define the output layer
    self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=False)

  def forward(self, x):
    # Apply the input layer
    x = self.conv1(x)
    x = nn.LeakyReLU(0.01, inplace=True)(x)
    # Apply the intermediate layers
    x = self.conv2(x)
    x = self.bn2(x)
    x = nn.LeakyReLU(0.01, inplace=True)(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = nn.LeakyReLU(0.01, inplace=True)(x)
    x = self.conv4(x)
    x = self.bn4(x)
    x = nn.LeakyReLU(0.01, inplace=True)(x)
    # Apply the output layer
    x = self.conv5(x)
    return x


class ResidualBlock(nn.Module):
  def __init__(self, channels):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(channels)
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(channels)

  def forward(self, x):
    identity = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = nn.ReLU(inplace=True)(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x += identity
    x = nn.ReLU(inplace=True)(x)
    return x

# Define the generators
G1 = Generator(input_size=100, hidden_size=128, output_size=784)
G2 = Generator(input_size=100, hidden_size=128, output_size=784)

# Define the discriminators
D1 = Discriminator(input_size=784, hidden_size=128, output_size=1)
D2 = Discriminator(input_size=784, hidden_size=128, output_size=1)