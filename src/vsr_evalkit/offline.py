"""Local artifact resolution plus a network-denying evaluation context."""

import hashlib
import json
import socket
import sys
from contextlib import ExitStack, contextmanager
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch
from urllib.parse import unquote, urlparse

from .errors import EvaluationError
from .registry import METRICS


def catalog():
    return json.loads(
        files("vsr_evalkit.resources")
        .joinpath("weights.json")
        .read_text(encoding="utf-8")
    )


def required_artifacts(metrics):
    return sorted(
        {artifact for name in metrics for artifact in METRICS[name].artifacts}
    )


def check_artifacts(metrics, paths):
    records, problems = {}, []
    known = catalog()
    for name in required_artifacts(metrics):
        path = paths.get(name)
        if path is None or not path.is_file() or path.stat().st_size == 0:
            problems.append(
                f"{name}: set weights.{name}; source: {known[name]['source']}"
            )
        else:
            records[name] = {"path": str(path), "size_bytes": path.stat().st_size}
    if problems:
        raise EvaluationError(
            "Missing local artifacts (nothing was downloaded):\n" + "\n".join(problems)
        )
    return records


def fingerprint(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def deny_network(*args, **kwargs):
    raise EvaluationError(
        "Network access is disabled during evaluation; configure every artifact locally"
    )


@contextmanager
def network_disabled():
    with ExitStack() as stack:
        for attr in ("connect", "connect_ex", "sendto"):
            stack.enter_context(patch.object(socket.socket, attr, deny_network))
        stack.enter_context(patch("socket.create_connection", deny_network))
        yield


class LocalResolver:
    def __init__(self, paths):
        self.paths = paths
        self.by_filename = {}
        for name, item in catalog().items():
            if name in paths:
                for filename in item["filenames"]:
                    self.by_filename[filename] = paths[name]

    def file(self, url, *args, **kwargs):
        filename = unquote(Path(urlparse(str(url)).path).name)
        path = self.by_filename.get(filename)
        if path is None or not path.is_file():
            raise EvaluationError(
                f"Unmapped implicit artifact requested: {url}. Add its official local file; downloads are disabled."
            )
        return str(path)

    def state_dict(self, url, *args, **kwargs):
        import torch

        return torch.load(
            self.file(url),
            map_location=kwargs.get("map_location", "cpu"),
            weights_only=False,
        )

    @contextmanager
    def activate(self):
        # Patch imported aliases as well as originals; the socket guard remains the final barrier.
        with ExitStack() as stack:
            for module in list(sys.modules.values()):
                name = getattr(module, "__name__", "")
                if not name.startswith(("torch", "torchvision", "pyiqa")):
                    continue
                for attr, replacement in (
                    ("load_file_from_url", self.file),
                    ("load_state_dict_from_url", self.state_dict),
                ):
                    if attr in vars(module):
                        stack.enter_context(patch.object(module, attr, replacement))
                if name == "pyiqa.archs.clip_model":
                    stack.enter_context(patch.object(module, "_download", self.file))
            with network_disabled():
                yield
