import torch
import torch.nn as nn
import torch.fft as fft


class FourierConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, use_bn=True, activation=nn.ReLU()):
        super(FourierConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Fourier convolution layers
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        # Batch normalization
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)

        # Activation function
        self.activation = activation

    def forward(self, x):
        real_part = self.conv_real(x)
        imag_part = self.conv_imag(x)
        complex_result = torch.complex(real_part, imag_part)
        fourier_transform = fft.fftn(complex_result, dim=[2, 3])
        magnitude = torch.abs(fourier_transform)

        # Batch normalization
        if self.use_bn:
            magnitude = self.bn(magnitude)

        # Activation function
        magnitude = self.activation(magnitude)

        return magnitude


class FourierCNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FourierCNN, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Fourier convolution blocks
        self.block1 = FourierConvBlock(input_channels, output_channels, kernel_size=3, padding=1)
        self.block2 = FourierConvBlock(output_channels, output_channels * 2, kernel_size=3, padding=1)

        # Residual connections
        self.residual1 = nn.Conv2d(output_channels * 2, output_channels * 2, kernel_size=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc = nn.Linear(output_channels * 2 * 7 * 7, 10)  # Assuming input size is 28x28

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Fourier convolution blocks
        x = self.block1(x)
        x = self.block2(x)

        # Residual connections
        residual = self.residual1(x)
        x += residual
        x = self.relu(x)

        # Pooling
        x = self.pool(x)

        # Flatten
        x = x.view(-1, self.output_channels * 2 * 7 * 7)

        # Fully connected layer
        x = self.fc(x)

        return x


# Example usage:
input_channels = 3
output_channels = 16
model = FourierCNN(input_channels, output_channels)

# Assuming input tensor has shape [batch_size, input_channels, height, width]
input_tensor = torch.randn(1, input_channels, 28, 28)
output_tensor = model(input_tensor)

print("Output shape:", output_tensor.shape)
