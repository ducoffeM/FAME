import os
import sys
from argparse import ArgumentParser


def configure_shared_gpu_from_argv() -> None:
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--shared_gpus", action="store_true")
    early_args, _ = parser.parse_known_args(sys.argv[1:])

    if not early_args.shared_gpus:
        return

    try:
        import GPUtil
    except ImportError as exc:
        raise RuntimeError(
            "GPUtil is required when --shared_gpus is enabled. Install it with `pip install gputil`."
        ) from exc

    gpus = GPUtil.getGPUs()
    if not gpus:
        raise RuntimeError("No GPU detected by GPUtil while --shared_gpus is enabled.")

    selected_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu.id)
    print(
        "[shared_gpus] Selected GPU {} (free memory: {:.0f} MB)".format(
            selected_gpu.id,
            selected_gpu.memoryFree,
        )
    )