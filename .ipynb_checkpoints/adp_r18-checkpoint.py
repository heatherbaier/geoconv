import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperNet(nn.Module):
    def __init__(self, input_features, output_features, normalize = False, use_means = False):     
        super(HyperNet, self).__init__()
    
        self.normalize = normalize
        self.use_means = use_means
                    
        self.fc1 = nn.Linear(2, 128)
        
        if self.use_means:
            self.fc2 = nn.Linear(input_features, 128)
            self.fc4 = nn.Linear(128 *2, output_features)
        else:
            self.fc4 = nn.Linear(128, output_features)
                
    def forward(self, x, channel_means):

        if self.normalize:
            x = torch.nn.functional.normalize(x)
        
        x = F.relu(self.fc1(x))
                        
        if x.shape[0] != 1:
            x = x.unsqueeze(0)
            
        if self.use_means:
            channel_means = F.relu(self.fc2(channel_means.unsqueeze(0)))
            x = torch.cat((x, channel_means), dim = 1)

        x = torch.sigmoid(self.fc4(x))
                
        x = torch.nn.functional.normalize(x)

        return x


    
class AdaptiveConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, input_features=10, normalize = False, use_means = False):
        super(AdaptiveConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.adaptive_filter_elements = out_channels * in_channels * kernel_size * kernel_size

        self.hypernet = HyperNet(input_features, self.adaptive_filter_elements + (out_channels if bias else 0), normalize = normalize, use_means = use_means)
        
    def forward(self, x, coords):

        global_feature = coords
        channel_means = x.mean([0,2,3])
                
        # Generate adaptive elements (weights and optionally biases)
        adaptive_elements = self.hypernet(global_feature, channel_means)
                
        # Separate adaptive weights and biases if bias is True
        if self.bias:
            adaptive_weights = adaptive_elements[:, :-self.out_channels]
            adaptive_biases = adaptive_elements[:, -self.out_channels:]
        else:
            adaptive_weights = adaptive_elements
            adaptive_biases = None
        
        adaptive_weights = adaptive_weights.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        # Apply the adaptive convolution using the generated weights (and biases)
        x = F.conv2d(x, adaptive_weights, bias=adaptive_biases, stride=self.stride, padding=self.padding)
        
        return x

    
class CustomSequential(nn.Module):
    def __init__(self, *args):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x, *args):
        for module in self.modules_list:
            if 'coords' in module.forward.__code__.co_varnames:
                x = module(x, *args)
            else:
                x = module(x)
        return x



    
class BasicBlock(nn.Module):
    
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_adaptive_conv=False, normalize=False, use_means=False):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        
        self.use_adaptive_conv = use_adaptive_conv

        if use_adaptive_conv:
            self.adaptive_layer = AdaptiveConvLayer(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False,
                                                    input_features=out_channels, normalize=normalize, use_means=use_means)
        else:
            self.adaptive_layer = nn.Identity()

        self.downsample = downsample

    def forward(self, x, coords):
        
        identity = x
        if self.downsample:
            identity = self.downsample(x, coords)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Apply adaptive layer before adding to the shortcut
        if self.use_adaptive_conv:
            out = self.adaptive_layer(out, coords)
        else:
            out = self.adaptive_layer(out)

        out += identity
        out = F.relu(out)
        return out  
    
    
    
    
    




class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, normalize = False, use_means = False):
        super(ResNet, self).__init__()
        
        self.in_channels = 64        
        self.conv1 = AdaptiveConvLayer(3, 64, kernel_size=7, stride=2, padding=1, bias=False, input_features = 3, normalize = normalize, use_means = use_means)
        
        # Initial convolution
        self.bn1 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers with adaptive convolution potentially enabled in 2 layers
        self.layer1 = self._make_layer(block, 64, layers[0], use_adaptive_conv=False, normalize=normalize, use_means=use_means)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_adaptive_conv=False, normalize=normalize, use_means=use_means)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_adaptive_conv=False, normalize=normalize, use_means=use_means)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_adaptive_conv=True, normalize=normalize, use_means=use_means)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1, use_adaptive_conv=False, normalize=False, use_means=False):
        
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            if use_adaptive_conv:
                # Use the adaptive convolution layer
                conv_layer = AdaptiveConvLayer(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False, input_features=self.in_channels, normalize=normalize, use_means=use_means)
            else:
                # Use a regular convolution layer
                conv_layer = nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False)

            downsample = CustomSequential(
                conv_layer,
                nn.InstanceNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, use_adaptive_conv=use_adaptive_conv, normalize=normalize, use_means=use_means))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, use_adaptive_conv=use_adaptive_conv, normalize=normalize, use_means=use_means))

        return CustomSequential(*layers)



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, coords):
        x = self.conv1(x, coords)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.layer1(x, coords)
        x = self.layer2(x, coords)
        x = self.layer3(x, coords)     
        x = self.layer4(x, coords)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



    