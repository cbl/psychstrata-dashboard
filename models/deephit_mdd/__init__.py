from .model import DeepHitMDD


def build_model() -> DeepHitMDD:
    return DeepHitMDD.build()


__all__ = ["DeepHitMDD", "build_model"]
