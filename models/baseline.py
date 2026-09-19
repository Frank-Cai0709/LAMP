from .lamp import LAMP


class Baseline(LAMP):
    def __init__(self, config):
        config = dict(config)
        model = dict(config["model"])
        model["spb"] = {**model["spb"], "enabled": False}
        model["erab"] = {**model["erab"], "enabled": False}
        model["afrb"] = {**model["afrb"], "enabled": False}
        config["model"] = model
        super().__init__(config)
