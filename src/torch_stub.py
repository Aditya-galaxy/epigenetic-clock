"""Stand in for torch so biolearn imports.

Only AltumAge needs torch and we do not use it, so a 2 GB install to read CSV coefficients would
be wasteful. scipy probes for torch.Tensor during import, so the stub carries one, and biolearn
imports torch.nn.functional, so that has to exist as a real module entry.
"""
import contextlib
import sys
import types


def _module(name):
    mod = types.ModuleType(name)
    mod.__getattr__ = lambda attr: (_ for _ in ()).throw(AttributeError(attr))
    sys.modules[name] = mod
    return mod


torch = _module("torch")
torch.Tensor = type("Tensor", (), {})
torch.load = lambda *a, **k: None
torch.tensor = lambda *a, **k: None
torch.from_numpy = lambda *a, **k: None
torch.float32 = "float32"
torch.no_grad = lambda: contextlib.nullcontext()
torch.__version__ = "0.0.0-stub"

nn = _module("torch.nn")
nn.Module = type("Module", (), {"__init__": lambda self, *a, **k: None})
nn.Linear = type("Linear", (), {"__init__": lambda self, *a, **k: None})
torch.nn = nn

functional = _module("torch.nn.functional")
nn.functional = functional
