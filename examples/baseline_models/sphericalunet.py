import torch 
from deepsphere.models.spherical_unet.unet_model import SphericalUNet

class SUNet(torch.nn.Module):
    def __init__(self, shape, input_channels, output_channels, embed_size, depth, laplacian_type, kernel_size, ratio, pool_factor):
        super(SUNet, self).__init__()
        if isinstance(shape, int):
            shape = [shape, shape]
        N = shape[0] * shape[1]
        self.unet = SphericalUNet(pooling_class="equiangular", N=N, depth=depth, laplacian_type=laplacian_type, kernel_size=kernel_size, ratio=ratio, pool_factor=pool_factor, output_channels=output_channels, input_channels=input_channels, embed_size=embed_size)

    def forward(self, x):
        B, C, H, W = x.shape 
        x = x.view(B, C, H*W)
        x = x.permute(0, 2, 1)
        output = self.unet(x)
        return output.view(B, -1, H, W)
    
def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_info(model, model_name="Model"):
    """Print model information including parameter count."""
    total_params = count_parameters(model)
    print(f"\n=== {model_name} Information ===")
    print(f"Total trainable parameters: {total_params:,}")
    
    # Convert to millions for readability
    if total_params >= 1_000_000:
        print(f"Parameters in millions: {total_params/1_000_000:.2f}M")
    elif total_params >= 1_000:
        print(f"Parameters in thousands: {total_params/1_000:.2f}K")
    
    # Memory estimation (rough)
    param_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32 parameter
    print(f"Estimated memory (parameters only): {param_size_mb:.2f} MB")

if __name__ == "__main__":
    # Example configuration
    x = torch.randn(1, 1, 256, 256)
    model = SUNet(shape=(256, 256), input_channels=1, output_channels=1, embed_size=16, depth=3, laplacian_type="combinatorial", kernel_size=3, ratio=1, pool_factor=4)
    
    # Print model information
    print_model_info(model, "Spherical UNet")
    
    # Test forward pass
    print(f"\nInput shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    print(f"\n💡 Model successfully created and tested!")