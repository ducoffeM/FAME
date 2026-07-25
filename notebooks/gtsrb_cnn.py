import os
from argparse import ArgumentParser, Namespace

os.environ["KERAS_BACKEND"] = "torch"
from shared_gpu import configure_shared_gpu_from_argv
configure_shared_gpu_from_argv()

from keras.models import load_model
import numpy as np
from tqdm import tqdm

# check robustness because of numerical approximation error between solvers
from fame.abstract_domain.utils import check_is_robust 
from fame.batch_free.free_l2 import check_is_robust_l2
from fame.experiments import exp_A_1, exp_A_2

import pickle
import random

random.seed(42)

def main(args: Namespace):
    filename = "gtsrb.pickle"
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    DATASET = "GTSRB"
    MODEL = "cnn"
    norm = args.norm
    eps = args.eps

    channel = 3
    data_format = "channels_last"
    n_class = 10

    """
    download and process GTSRB data.
    """
    x_test, y_test = data['x_test'], data['y_test']
    x_test = x_test.astype('float32') / 255
    x_test = np.reshape(x_test, (-1, 3072))

    k_model = load_model("./models/xairobas_gtsrb-cnn.keras")

    def get_predicted_label(input_sample: np.ndarray,) -> int:
        prediction = k_model.predict(np.asarray(input_sample)[None], verbose=0)
        return int(np.argmax(prediction[0]))

    def is_robust(j: int) -> bool:
        if norm == "linf":
            return check_is_robust(
                model=k_model,
                input_sample=x_test[j],
                eps=eps,
                channel=channel,
                data_format=data_format,
                n_class=n_class,
            )
        if norm == "l2":
            return check_is_robust_l2(
                model=k_model,
                input_sample=x_test[j],
                gt_label=get_predicted_label(x_test[j]),
                eps=eps,
                channel=channel,
                data_format=data_format,
                n_class=n_class,
            )
        raise ValueError(f"Unknown norm: {norm}")
    indices = list(range(len(x_test)))
    random.shuffle(indices)
    indices = indices[:100]
    indices = [i for i in tqdm(indices, desc="Checking robustness") if not is_robust(i)]
    print("len(indices): ", len(indices))
    print("indices: ", indices)

    dataframe_repository = "./results"

    filename = "{}_{}_{}_norm_{}_eps_{}".format(DATASET, MODEL, args.exp, norm, str(eps).replace("0.", ""))
    if args.exp == "A1":
        exp_A_1(
                model=k_model,
                x_test=x_test,
                y_test=y_test,
                indices=indices,
                eps=eps,
                dataframe_repository=dataframe_repository,
                dataframe_filename=filename,
                channel=channel,
                data_format=data_format,
                n_class=n_class,
                verbose=1,
                norm=norm
            )
    elif args.exp == "A2":
        exp_A_2(
                model=k_model,
                x_test=x_test,
                y_test=y_test,
                indices=indices,
                eps=eps,
                dataframe_repository=dataframe_repository,
                dataframe_filename=filename,
                channel=channel,
                data_format=data_format,
                n_class=n_class,
                verbose=1,
                norm=norm
            )
    else:
        raise ValueError(f"Unknown experiment: {args.exp}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--norm", type=str, default="l2", choices=["linf", "l2"])
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--exp", type=str, default="A1", choices=["A1", "A2"])
    parser.add_argument(
        "--shared_gpus",
        action="store_true",
        help="Select the GPU with the most free memory and mask visibility to that GPU only.",
    )
    args = parser.parse_args()

    main(args)