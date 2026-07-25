import os
os.environ["KERAS_BACKEND"] = "torch"

from keras.models import load_model
from argparse import ArgumentParser, Namespace
import numpy as np

# check robustness because of numerical approximation error between solvers
from fame.abstract_domain.utils import check_is_robust 
from fame.experiments import exp_A_1, exp_A_2, exp_A_2_no_overwrite, exp_B_no_overwrite

import pickle


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

    indices = [i for i in range(100) if not i in [0,\
    1,2,4, 5, 8, 11, 16, 20, 22, 24, 31, 38, 42, 43, 44, 46, 49, 50, 52, 53, 55, 56,57, 58, 60, 62, 64, 67, 72, 74, 77, 78, 79, 81, 83, 88, 91, 92, 93]]

    def is_robust(j):
        return check_is_robust(model=k_model, 
                        input_sample=x_test[j], 
                        eps=eps, 
                        channel=channel, 
                        data_format=data_format, 
                        n_class=n_class)

    indices = [i for i in indices if not is_robust(i)]

    print("len(indices): ", len(indices))

    dataframe_repository = "./results"

    EXP = "A_1"
    filename = "{}_{}_{}_norm_{}".format(DATASET, MODEL, EXP, norm)
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
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--norm", type=str, default="linf", choices=["linf", "l2"])
    parser.add_argument("--eps", type=float, default=0.01)
    args = parser.parse_args()

    main(args)