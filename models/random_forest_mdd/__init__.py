from .model import RandomForestMDD


def build_model() -> RandomForestMDD:
    return RandomForestMDD.train()


__all__ = ["RandomForestMDD", "build_model"]
