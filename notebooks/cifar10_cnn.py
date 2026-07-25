import os
from argparse import ArgumentParser, Namespace

os.environ["KERAS_BACKEND"] = "torch"
from shared_gpu import configure_shared_gpu_from_argv


configure_shared_gpu_from_argv()

from keras.models import load_model
from keras import Model
import numpy as np
from tqdm import tqdm
from typing import Any, cast

# check robustness because of numerical approximation error between solvers
from fame.abstract_domain.utils import check_is_robust
from fame.batch_free.free_l2 import check_is_robust_l2
from fame.experiments import exp_A_1, exp_A_2

import random

from configs.cifar10_configs import get_dataset, dataset_to_numpy, means_np, stddevs_np

random.seed(42)

def main(args: Namespace):
    train_dataset, val_dataset = get_dataset(augment=False, get_train=True, get_val=True)
    test_dataset = get_dataset(augment=False, get_train=False, get_val=False)

    DATASET = "CIFAR10"
    MODEL = "cnn"
    norm = args.norm
    eps = args.eps

    means_avg = float(np.mean(means_np))
    std_avg = float(np.mean(stddevs_np))
    print("eps:", eps)
    channel = 3
    data_format = "channels_last"
    n_class = 10

    """
    Download and process CIFAR10 data.
    """
    x_train, y_train = dataset_to_numpy(cast(Any, train_dataset), means_np, stddevs_np)
    x_valid, y_valid = dataset_to_numpy(cast(Any, val_dataset), means_np, stddevs_np)
    x_test, y_test = dataset_to_numpy(cast(Any, test_dataset), means_np, stddevs_np)
    del x_train, y_train, x_valid, y_valid

    x_test_flattened = np.reshape(x_test, (-1, 3072))
    print(f"x_test shape (normalized, NHWC): {x_test.shape}")
    print(f"x_test dtype: {x_test.dtype}")

    k_model = cast(Model, load_model("./models/resnet_2b_ported.keras"))

    def get_predicted_label(input_sample: np.ndarray) -> int:
        prediction = k_model.predict(np.asarray(input_sample)[None])
        return int(np.argmax(prediction[0]))

    def is_robust(j: int) -> bool:
        if norm == "linf":
            return check_is_robust(
                model=k_model,
                input_sample=x_test_flattened[j],
                eps=eps,
                channel=channel,
                data_format=data_format,
                n_class=n_class,
            )
        if norm == "l2":
            return check_is_robust_l2(
                model=k_model,
                input_sample=x_test_flattened[j],
                gt_label=get_predicted_label(x_test_flattened[j]),
                eps=eps,
                channel=channel,
                data_format=data_format,
                n_class=n_class,
            )
        raise ValueError(f"Unknown norm: {norm}")

    indices = list(range(len(x_test_flattened)))
    random.shuffle(indices)
    indices = indices[: args.n_samples]
    indices = [i for i in tqdm(indices, desc="Checking robustness") if not is_robust(i)]
    print("len(indices): ", len(indices))
    print("indices: ", indices)

    dataframe_repository = "./results"
    filename = "{}_{}_{}_norm_{}_eps_{}".format(
        DATASET,
        MODEL,
        args.exp,
        norm,
        str(eps).replace("0.", ""),
    )

    if args.exp == "A1":
        exp_A_1(
            model=k_model,
            x_test=x_test_flattened,
            y_test=y_test,
            indices=indices,
            eps=eps,
            dataframe_repository=dataframe_repository,
            dataframe_filename=filename,
            channel=channel,
            data_format=data_format,
            n_class=n_class,
            verbose=1,
            norm=norm,
            means=means_avg,
            stddev=std_avg,
        )
    elif args.exp == "A2":
        exp_A_2(
            model=k_model,
            x_test=x_test_flattened,
            y_test=y_test,
            indices=indices,
            eps=eps,
            dataframe_repository=dataframe_repository,
            dataframe_filename=filename,
            channel=channel,
            data_format=data_format,
            n_class=n_class,
            verbose=1,
            norm=norm,
            means=means_avg,
            stddev=std_avg,
        )
    else:
        raise ValueError(f"Unknown experiment: {args.exp}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--norm", type=str, default="l2", choices=["linf", "l2"])
    parser.add_argument("--eps", type=float, default=0.005)
    parser.add_argument("--exp", type=str, default="A1", choices=["A1", "A2"])
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument(
        "--shared_gpus",
        action="store_true",
        help="Select the GPU with the most free memory and mask visibility to that GPU only.",
    )
    args = parser.parse_args()

    main(args)
