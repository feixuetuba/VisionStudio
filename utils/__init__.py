import logging
import os.path
import importlib

def load_instance(pkg:str, *args,**kwargs):
    pkg, cls = os.path.splitext(pkg)
    cls = cls[1:]
    cls = load_attr(pkg, cls)
    if hasattr(cls, "parse_args"):
        opts = getattr(cls,"parse_args")(args, kwargs)
        return cls(opts)
    return cls(*args, **kwargs)


def load_attr(pkg:str, cls:str):
    pkg = importlib.import_module(pkg)
    cls = getattr(pkg, cls)
    return cls

