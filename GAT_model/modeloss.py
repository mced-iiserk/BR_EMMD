import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from sklearn.neighbors import KernelDensity
from torchvision import models
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

# Deep fully-connected neural network
class DeepFCnn(nn.Module):
    def __init__(self, feature_dim, targets_dim):
        super(DeepFCnn, self).__init__()
        self.feature_dim = feature_dim
        self.targets_dim = targets_dim

        # Fully connected layers
        self.fc0 = nn.Linear(feature_dim[0]*feature_dim[1]*feature_dim[2],32)
        self.fc1 = nn.Linear(32, 64)
        #self.fc2 = nn.Linear(512, 1024)
        #self.fc3 = nn.Linear(1024, 512)
        #self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(64, targets_dim[0]*targets_dim[1])

    def forward(self, data):
        x = data.view(-1, self.feature_dim[0]*self.feature_dim[1]*self.feature_dim[2])
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = x.view(-1, self.targets_dim[0], self.targets_dim[1])
        return x

# Graph Attention Network to Fully Connected to Matrix
class GraphAtt2Mesh(nn.Module):
    def __init__(self, node_feature_dim, conv_output_size=(10, 10), heads=8):
        super(GraphAtt2Mesh, self).__init__()

        # Graph Attention layers (multi-head attention)
        self.conv1 = GATConv(node_feature_dim, 64 // heads, heads=heads, concat=True)
        self.conv2 = GATConv(64, 128 // heads, heads=heads, concat=True)
        #self.conv3 = GATConv(128, 128 // heads, heads=heads, concat=True)

        # Fully connected layers to process pooled graph representation
        self.fc1 = nn.Linear(128, 256)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(512, conv_output_size[0] * conv_output_size[1])

        # Convolutional layer to process into 2D output
        self.conv_output_size = conv_output_size

    def forward(self, data):
        # data contains x (node features), edge_index (graph structure), and batch (batch info)
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply graph attentions (with multi-head)
        x, attn_weights_1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.relu(x)
        x, attn_weights_2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = F.relu(x)
        #x, attn_weights_3 = self.conv3(x, edge_index, return_attention_weights=True)
        #x = F.relu(x)

        # Global pooling to get a fixed-size representation
        x = global_mean_pool(x, batch)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        #x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        # Reshape to 2D matrix
        x = x.view(-1, self.conv_output_size[0], self.conv_output_size[1])

        return x#, [attn_weights_1, attn_weights_2]  # return attention weights as well


# Graph Conv to Fully Connected to Matrix
class Graph2Mesh(nn.Module):
    def __init__(self, node_feature_dim, conv_output_size=(10, 10)):
        super(Graph2Mesh, self).__init__()
        
        # Graph convolutional layers
        self.conv1 = GCNConv(node_feature_dim, 64)
        self.conv2 = GCNConv(64, 128)
        #self.conv3 = GCNConv(128, 128)
        
        # Fully connected layers to process pooled graph representation
        self.fc1 = nn.Linear(128, 256)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(512, conv_output_size[0] * conv_output_size[1])
        
        # Convolutional layer to process into 2D output
        self.conv_output_size = conv_output_size

    def forward(self, data):
        # data contains x (node features), edge_index (graph structure), and batch (batch info)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        #x = F.relu(self.conv3(x, edge_index))
        # Global pooling to get a fixed-size representation
        x = global_mean_pool(x, batch)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        #x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        # Reshape to 2D matrix
        x = x.view(-1, self.conv_output_size[0], self.conv_output_size[1])
        
        return x

class Graph2Latent(nn.Module):
    def __init__(self, node_feature_dim, latent_size=20):
        super(Graph2Latent, self).__init__()

        # Graph convolutional layers
        self.conv1 = GCNConv(node_feature_dim, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)

        # Fully connected layers to process pooled graph representation
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(64, latent_size)

        # Convolutional layer to process into 2D output
        #self.latent_size = latent_size

    def forward(self, data):
        # data contains x (node features), edge_index (graph structure), and batch (batch info)
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        # Global pooling to get a fixed-size representation
        x = global_mean_pool(x, batch)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        #x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout2(x)
        x = self.fc3(x)

        # Reshape to 2D matrix
        #x = x.view(-1, 1, self.conv_output_size[0], self.conv_output_size[1])

        return x

# Modify ResNet-50 to fit the input and output dimensions
class pretrained_ResNet50(nn.Module):
    def __init__(self, output_channels):
        super(pretrained_ResNet50, self).__init__()
        # Load pre-trained ResNet-50
        self.resnet = models.resnet50(pretrained=True)

        # Modify the first conv layer to accept 6 channels instead of 3
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the final fully connected layer to match output size
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, output_channels * 625),  # Flattened output size
            nn.Unflatten(1, (output_channels, 25, 25))  # Reshape to (output_channels, 10, 10)
        )

    def forward(self, x):
        return self.resnet(x)

class pretrained_ResNet18(nn.Module):
    def __init__(self, output_channels):
        super(pretrained_ResNet18, self).__init__()
        # Load pre-trained ResNet-50
        self.resnet = models.resnet18(pretrained=True)

        # Modify the first conv layer to accept 6 channels instead of 3
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the final fully connected layer to match output size
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, output_channels * 625),  # Flattened output size
            nn.Unflatten(1, (output_channels, 25, 25))  # Reshape to (output_channels, 10, 10)
        )

    def forward(self, x):
        return self.resnet(x)

class SmallCNN(nn.Module):
    def __init__(self, output_channels):
        super(SmallCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # input: (6, 30, 30), output: (32, 30, 30)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # input: (32, 30, 30), output: (64, 30, 30)
        self.bn2 = nn.BatchNorm2d(64)

        # Max Pooling layer to downsample
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # input: (64, 30, 30), output: (64, 15, 15)
        
        # Additional conv layers to match output size
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)   # input: (64, 15, 15), output: (128, 15, 15)
        self.bn3 = nn.BatchNorm2d(64)
        #self.dropout3 = nn.Dropout(p=0.3)                          # Drop 20% of neurons
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)   # input: (128, 15, 15), output: (32, 15, 15)
        self.bn4 = nn.BatchNorm2d(32)
        #self.dropout4 = nn.Dropout(p=0.3)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=3, padding=1)   # input: (128, 15, 15), output: (32, 15, 15)
        self.bn5 = nn.BatchNorm2d(1)
        #self.dropout5 = nn.Dropout(p=0.3)
        # A final conv layer to get the desired output channels (1 or 2)
        #self.conv_out = nn.Conv2d(8, output_channels, kernel_size=3, padding=1, stride=1)  # input: (32, 15, 15), output: (1 or 2, 15, 15)
        #self.final_upsample = nn.ConvTranspose2d(8, output_channels, kernel_size=3, stride=1, padding=1, output_padding=1)  # output: (output_channels, 25, 25)
        
        # A layer to resize the spatial dimensions to (10, 10)
        #self.resize = nn.AdaptiveAvgPool2d((6, 6))  # input: (1 or 2, 15, 15), output: (1 or 2, 10, 10)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        #x = self.pool(x)
        
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        #x = self.dropout3(x)
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        #x = self.dropout4(x)
        x = nn.functional.relu(self.bn5(self.conv5(x)))
        #x = self.dropout5(x)
        x = nn.functional.interpolate(x, scale_factor=2.5, mode='bilinear', align_corners=True)
        #x = self.final_upsample(x)
        #x = self.conv_out(x)
        #x = self.resize(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (identity)
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip_connection(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Skip connection
        out = F.relu(out)
        return out

class CustomResNetRN11(nn.Module):
    def __init__(self):
        super(CustomResNetRN11, self).__init__()

        # Initial Conv layer for 15x15x2 input
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Residual Blocks (24 total, grouped into several stages)
        self.layer1 = self._make_layer(ResidualBlock, 16, 64, 3)  # 3 residual blocks in this stage
        #self.dropout1 = nn.Dropout(p=0.1)
        self.layer2 = self._make_layer(ResidualBlock, 64, 128, 3)  # Next stage
        #self.dropout2 = nn.Dropout(p=0.4)
        self.layer3 = self._make_layer(ResidualBlock, 128, 256, 3)
        #self.dropout3 = nn.Dropout(p=0.5)
        self.layer4 = self._make_layer(ResidualBlock, 256, 128, 3)
        #self.dropout4 = nn.Dropout(p=0.5)
        self.layer5 = self._make_layer(ResidualBlock, 128, 64, 3)
        #self.dropout5 = nn.Dropout(p=0.4)
        self.layer6 = self._make_layer(ResidualBlock, 64, 8, 3)
        #self.dropout6 = nn.Dropout(p=0.3)

        # Final Conv layers for 16x16x3 output
        self.conv_final = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)  # Output 16x16x3

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual Blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        # Final output
        x = self.conv_final(x)
        x = F.interpolate(x, size=(25, 25), mode='bilinear', align_corners=False)

        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # Squeeze: Global average pooling
        y = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        # Excitation: Fully connected layers with ReLU and sigmoid
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        # Reshape and scale
        y = y.view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)

class SimpleAttentionCNN(nn.Module):
    def __init__(self, reduction=16):
        super(SimpleAttentionCNN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)  # Maintain spatial resolution
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Maintain spatial resolution
        self.bn2 = nn.BatchNorm2d(64)
        self.attention = SEBlock(64, reduction)  # Apply attention after the second conv layer

        # Additional convolutional layers to reduce channels and match the output shape
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # Maintain spatial resolution
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, padding=1)  # Maintain spatial resolution
        #self.upsample = nn.Upsample(size=(6, 6), mode='bilinear', align_corners=False)

    def forward(self, x):
        # Convolutional layers with attention
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Apply attention
        x = self.attention(x)

        # Additional convolutional layers to match output dimensions
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)  # Output shape should be (batch_size, 1, 10, 10)
        #x = self.upsample(x)
        x = F.interpolate(x, size=(46, 46), mode='bilinear', align_corners=False)

        return x

def passfunc(x):
    return x

#MSELoss = nn.MSELoss
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, logits, label, weights=None):
        #weights=None
        loss = torch.abs(label - logits).pow(2)
        #if type(weights) != type(None):
        #    loss = loss*weights
        #    mask = (weights != 0).float()
        #    meanloss = loss.sum()/mask.sum()
        #else:
        #    meanloss = loss.mean()
        meanloss = loss.mean()
        return meanloss

class FocalMSELoss(nn.Module):
    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 ):
        super(FocalMSELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, label, weights=None):
        L1 = torch.abs(label - logits)
        loss = torch.sigmoid(torch.abs(self.alpha*L1)).pow(self.gamma)*L1
        if type(weights) != type(None):
            loss = loss*weights
        return loss.mean()

