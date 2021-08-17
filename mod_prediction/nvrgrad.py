import sys
import json
import os

import nevergrad as ng
import names

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import mod_prediction  # noqa: F401

    __package__ = "mod_prediction"

# Custom imports
from mod_prediction.train import main

# Load config
with open("mod_prediction/configs/max.json", "r") as f:
    common_args = json.load(f)


def training(encoder_size, decoder_size):
    common_args["encoder_size"] = encoder_size
    common_args["decoder_size"] = decoder_size
    common_args["model_name"] = names.get_first_name()
    common_args["debug"] = False
    common_args["use_cuda"] = bool(common_args["gpu"])
    common_args["online_layer"] = 0
    common_args["grid_size"] = [13, 3]
    common_args["vis"] = False

    print(
        "{} running with encoder_size of {} and {} decoder_size decay.".format(
            common_args["model_name"],
            common_args["encoder_size"],
            common_args["decoder_size"],
        )
    )

    nll = main(common_args, verbose=True)

    print("Finished with NLL of {0:.2f}".format(nll))

    return nll


parametrization = ng.p.Instrumentation(
    decoder_size=ng.p.Scalar(lower=10, upper=100).set_integer_casting(),
    encoder_size=ng.p.Scalar(lower=10, upper=100).set_integer_casting(),
)

optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=20)
recommendation = optimizer.minimize(training)

print(recommendation.kwargs)
