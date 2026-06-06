from typing import Dict, Type

_REGISTRY: Dict[str, Type] = {}


def register_recommender(name: str = None):
    def _decorator(cls):
        key = name or cls.__name__
        key = key.lower()
        _REGISTRY[key] = cls
        return cls

    return _decorator


def get_registry() -> Dict[str, Type]:
    return dict(_REGISTRY)
