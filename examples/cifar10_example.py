import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from fame import get_xai_fame
import time


class NormalizeLayerConv2d(nn.Module):
    """Normalizes 4D image tensors (N, C, H, W) using a 1x1 depthwise Conv2d layer."""
    def __init__(self, mean, std, device="cpu"):
        super().__init__()
        channels = len(mean)
        
        # 1x1 depthwise convolution (groups=channels prevents cross-channel mixing)
        self.conv = nn.Conv2d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=1, 
            groups=channels, 
            bias=True, 
            device=device
        )
        
        mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
        std_t = torch.tensor(std, dtype=torch.float32, device=device)
        
        # Override randomly initialized weights and biases with normalization parameters
        with torch.no_grad():
            # Weight shape for depthwise conv must be (channels, 1, 1, 1)
            self.conv.weight.copy_((1.0 / std_t).view(channels, 1, 1, 1))
            self.conv.bias.copy_(-mean_t / std_t)

    def forward(self, x):
        # Expects x to be of shape (Batch, Channels, Height, Width)
        return self.conv(x)

class ELFlatten(nn.Module):
    """Flatten module compatible with CNN-3 checkpoints from expressive losses."""
    def forward(self, x):
        return x.view(x.size(0), -1)
        
def cnn3(in_ch=3, in_dim=32, width=64, num_class=10):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 5, stride=2, padding=2),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 4, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        ELFlatten(), # Replaced standard nn.Flatten() with your custom ELFlatten
        nn.Linear((in_dim // 4) ** 2 * 2 * width, num_class),
    )
    return model

def get_model_and_data(eps, mean=None, std=None, clip_min=0.0, clip_max=1.0):
    if mean is None:
        mean = [0.0, 0.0, 0.0]
    if std is None:
        std = [1.0, 1.0, 1.0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Instantiate base model and load checkpoint
    base_model = cnn3(in_ch=3, in_dim=32, width=64, num_class=10).to(device)
    model_path = "./models/cnn3_cifar10.pt"
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        base_model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Warning: {model_path} not found. Initializing with random weights.")

    # 2. Wrap the model with the NormalizeLayer so it can accept [0,1] images directly
    
    model = nn.Sequential(
        NormalizeLayerConv2d(mean, std, device),
        base_model
    ).to(device)
    
    model.eval()
    
    #model = base_model

    # 3. Load unnormalized test images in [0, 1]
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True,
        transform=transforms.ToTensor()
    )

    images = torch.stack([test_dataset[i][0] for i in range(10)]).to(device)
    labels = torch.tensor([test_dataset[i][1] for i in range(10)]).to(device)

    # 4. Calculate verification/adversarial bounds
    lower_bound = torch.clamp(images - eps, min=clip_min, max=clip_max)
    upper_bound = torch.clamp(images + eps, min=clip_min, max=clip_max)
    
    # Calculate max normalized epsilon (useful for downstream bounding/XAI tools)
    std_tensor = torch.tensor(std, device=device).view(1, -1, 1, 1)
    eps_norm_max = eps / torch.min(std_tensor).item()

    return model, labels, images, lower_bound, upper_bound, eps_norm_max


if __name__ == "__main__":
    eps = 16. / 255
    
    # Retrieve the configured components
    model, labels, images, lower_bound, upper_bound, eps_norm_max = get_model_and_data(
        eps, 
        mean=(0.4914, 0.4822, 0.4465), 
        std=(0.2023, 0.1994, 0.2010)
    )

    device = next(model.parameters()).device
    input_shape = (3, 32, 32)
    
    # Display execution details
    print(f"Execution Device : {device}")
    print(f"Input Shape      : {input_shape}")
    print(f"Eps Norm Max     : {eps_norm_max:.4f}")
    
    # Quick sanity check evaluation
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

    print(f"Ground Truth     : {labels.tolist()}")
    print(f"Predictions      : {predictions.tolist()}")

    for i in range(10):

        import time
        start_time = time.time()

        image_np = images[i].detach().cpu().numpy()
        gt_label = int(labels[i].detach().cpu().numpy())

        xai_indices = []
        free_indices = []

        lower_bound_reshaped = np.maximum(0, image_np - eps)
        upper_bound_reshaped = np.minimum(1, image_np + eps)

        xai_indices, free_indices, remaining_indices = get_xai_fame(
            model=model,
            input_shape=input_shape,
            gt_label=gt_label,
            input_sample=image_np,    
            lower_bound_input=lower_bound_reshaped,
            upper_bound_input=upper_bound_reshaped,
            data_format="channels_first",
            eps=eps,
            n_class = 10,
            free_methods=["CROWN"],
            xai_methods=[],
            traversal_order="lirpa",
            verbose=0,
        )

        print("index", i, "C_x (xai_indices)", len(xai_indices), "R_x free_indices", len(free_indices), 'U_x', len(remaining_indices), "running time:", time.time() - start_time)