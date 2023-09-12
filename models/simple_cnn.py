import torch
import torch.nn as nn

class SimpleCNN(torch.nn.Module):

    def __init__(
            self,
            input_channels: int,
            hidden_channels: int,
            num_hidden_layers: int,
            use_batchnormalization: bool,
            num_classes: int,
            kernel_size: int = 3,
            activation_function: torch.nn.Module = torch.nn.ReLU()
    ):
        super().__init__()

        self.hidden_channels = hidden_channels 
        self.activation_function = activation_function
        padding = (kernel_size - 1) // 2

        hidden_layers = []

        for _ in range(num_hidden_layers):
            layer = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size, padding=padding)
            hidden_layers.append(layer)

            if use_batchnormalization:
                hidden_layers.append(nn.BatchNorm2d(hidden_channels))

            hidden_layers.append(activation_function)
            input_channels = hidden_channels

        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.output_layer = nn.Linear(hidden_channels, num_classes)

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        output = self.hidden_layers(input_images)
    
        # Adding a final convolutional layer to get a single-channel output
        final_conv_layer = nn.Conv2d(in_channels=self.hidden_channels, out_channels=1, kernel_size=1)
        output = final_conv_layer.to(input_images.device)(output)
        
        return output




# if __name__ == "__main__":
#     torch.random.manual_seed(0)
#     network = SimpleCNN(1, 32, 3, True, 10, activation_function=nn.ELU())
#     input = torch.randn(8, 1, 170, 227)
#     output = network(input)
#     print(output)
    