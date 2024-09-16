import torch

class ConvBlock(torch.nn.Module):
    """
    A convolutional block for neural networks, supporting different activation functions.

    This block performs the following operations:
        1. Convolution: Applies a 2D convolutional layer with specified parameters.
        2. Batch Normalization: Normalizes the output of the convolution.
        3. Activation: Applies a chosen activation function (PReLU or LeakyReLU)
                       for non-linearity.

    Args:
        in_channels (int): Number of input channels for the convolution.
        out_channels (int): Number of output channels for the convolution.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the convolution.
        activation_type (str): Type of activation function to use ("prelu" or "lrelu").
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 activation_type: str):
        super(ConvBlock, self).__init__()

        self.activation = None  # Initialize activation layer as None

        # Set the activation function based on the provided type
        if activation_type == "prelu":
            self.activation = torch.nn.PReLU()
        elif activation_type == "lrelu":
            self.activation = torch.nn.LeakyReLU(negative_slope=0.2)

        # Define the convolutional layer
        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    stride=stride,
                                    kernel_size=kernel_size, padding=padding)

        # Define the batch normalization layer
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolutional block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying convolution,
                           batch normalization, and activation.
        """

        # Apply the convolution
        x = self.conv(x)

        # Apply batch normalization for improved training stability
        x = self.bn(x)

        # Apply the chosen activation function for non-linearity
        if self.activation is not None:
            x = self.activation(x)

        return x
