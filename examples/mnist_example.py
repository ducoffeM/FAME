import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from fame import get_xai_fame


def cnn3(in_ch=1, in_dim=28, width=64, num_class=10):
    model = nn.Sequential(
        nn.Conv2d(in_ch, width, 5, stride=2, padding=2),
        nn.BatchNorm2d(width),
        nn.ReLU(),
        nn.Conv2d(width, 2 * width, 4, stride=2, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim // 4) ** 2 * 2 * width, num_class),
    )
    return model


def get_model_and_data():
    # 2. Select execution device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Instantiate model architecture
    model = cnn3(in_ch=1, in_dim=28, width=64, num_class=10).to(device)

    # 4. Load full checkpoint file and extract the nested state_dict
    model_path = "./models/cnn3_mnist.pt"
    checkpoint = torch.load(model_path, map_location=device)

    # Extract state_dict key from the checkpoint dictionary
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle case where checkpoint keys might have "module." prefix (e.g., trained with DataParallel)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # 5. Load MNIST test dataset with standard normalization
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # 6. Extract the first 10 test samples
    images = torch.stack([test_dataset[i][0] for i in range(10)]).to(device)
    labels = torch.tensor([test_dataset[i][1] for i in range(10)]).to(device)

    # 7. Evaluate
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

    # 8. Display results
    print(f"Ground Truth Labels : {labels.tolist()}")
    print(f"Model Predictions   : {predictions.tolist()}")

    return model, images, labels




if __name__=="__main__":

    model, images, labels = get_model_and_data()
    device = next(model.parameters()).device
    input_shape = (1, 28, 28)
    eps=0.25

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
            xai_methods=["apgd"],
            traversal_order="lirpa",
            verbose=0,
        )

        print("index", i, "C_x (xai_indices)", len(xai_indices), "R_x free_indices", len(free_indices), 'U_x', len(remaining_indices), "running time:", time.time() - start_time)

