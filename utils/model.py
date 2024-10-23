import torch
import torch.nn as nn

class SoliCNNTaskA(nn.Module):
    """
    SoliCNNTaksA model with three convolutional layers and three fully connected layers
    """

    def __init__(self, num_classes=11):
        """
        :param num_classes (int): Number of classification classes
        """
        super(SoliCNNTaskA, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128*26*26, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Builds the neural network.

        :param x (torch.Tensor): Input tensor of shape (batch_size, 4, 32, 32, Number of channels)

        Returns:
        - x (torch.Tensor): Output
        """
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))

        x = x.view(-1, 128*26*26)
        x = nn.ReLU()(self.fc1(x)) 
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    
class DepthwiseConv(nn.Module):
    """
    Depthwise convoutional layer 
    """
    def __init__(self, in_channels, kernel_size=2, stride=2, padding=0):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)

        with torch.no_grad():
            self.depthwise.weight.fill_(1.0 / (kernel_size * kernel_size))

        self.depthwise.weight.requires_grad = False

    def forward(self, x):
        return self.depthwise(x)

class SoliCNNTaskB1(nn.Module):
    """
    SoliCNNTaksB1 model with three convolutional layers, three depthwise layers and two fully connected layers
    """
    def __init__(self, num_classes=11):
        """
        :param num_classes (int): Number of classification classes
        """
        super(SoliCNNTaskB1, self).__init__()
        # Convolutional and Depth-wise layers
        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.depth_layer1 = DepthwiseConv(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.depth_layer2 = DepthwiseConv(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.depth_layer3 = DepthwiseConv(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 16) 
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        """
        Builds the neural network.

        :param x (torch.Tensor): Input tensor of shape (batch_size, 4, 32, 32, Number of channels)

        Returns:
        - x (torch.Tensor): Output
        """
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.depth_layer1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.depth_layer2(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.depth_layer3(x)
        
        x = x.view(-1, 16 * 4 * 4)  # Flatten for fully connected layer
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)

        return x
    
class SoliCNNTaskB2(nn.Module):
    """
    SoliCNNTaksB2 model with three convolutional layers, three depthwise layers and two fully connected layers
    """
    def __init__(self, num_classes=11):
        """
        :param num_classes (int): Number of classification classes
        """
        super(SoliCNNTaskB2, self).__init__()
        # Quantization placeholder
        self.quant = torch.quantization.QuantStub()  
        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.depth_layer1 = DepthwiseConv(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.depth_layer2 = DepthwiseConv(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.depth_layer3 = DepthwiseConv(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 16)  
        self.fc2 = nn.Linear(16, num_classes)
        # Dequantization placeholder
        self.dequant = torch.quantization.DeQuantStub()  

    def forward(self, x):
        """
        Builds the neural network.

        :param x (torch.Tensor): Input tensor of shape (batch_size, 4, 32, 32, Number of channels)

        Returns:
        - x (torch.Tensor): Output
        """
        # Quantize the input
        x = self.quant(x)
        
        # Forward through the network
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.depth_layer1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.depth_layer2(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.depth_layer3(x)
        
        x = x.reshape(-1, 16 * 4 * 4)  # Flatten for fully connected layer
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        
        # Dequantize the output
        x = self.dequant(x)
        return x
