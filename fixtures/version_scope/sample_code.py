class Model:
    def run(self, params):
        if "v1" in params.version:
            return 1
        if "v10." in params.version:
            return 2
        if "v3.8" not in params.version:
            return 3
        if params.version == "legacy":
            return 4
        if params.mode == "train":
            return 5
        if params.version.startswith("v9"):
            return 6
        if "a" in ["a", "b"]:
            return 7
