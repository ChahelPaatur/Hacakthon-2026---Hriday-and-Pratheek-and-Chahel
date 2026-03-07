"""
NeuroLang Python SDK
====================

Write neural networks in plain English from Python.
Works via the NeuroLang CLI (subprocess) or HTTP API (if server is running).

Installation:
    pip install neurolang   # (or copy this file into your project)
    npm install -g neurolang  # ensure the CLI is available

Usage:
    from neurolang import nl

    # One-liner
    result = nl("Predict species from iris")
    print(result.accuracy)   # 0.967

    # With your own data (pandas DataFrame or list of dicts)
    import pandas as pd
    df = pd.read_csv("customers.csv")
    result = nl("Predict churned with age & income", data=df)

    # Interpolate variables
    result = nl("Predict {target} with {features} from {dataset}",
                target="price", features=["size", "bedrooms"], dataset="housing")

    # Just compile (no training)
    compiled = nl.compile("Predict species from iris")
    print(compiled.code)      # TensorFlow.js code
    print(compiled.ir)        # intermediate representation

    # Generate code for other frameworks
    print(nl.pytorch("Predict species from iris"))
    print(nl.keras("Predict species from iris"))
    print(nl.jax("Predict species from iris"))

    # Decorator pattern — define models as classes
    @nl.model("Predict species from iris epochs 30")
    class IrisClassifier:
        pass

    print(IrisClassifier.accuracy)
    print(IrisClassifier.predictions)

    # Jupyter notebook magic
    # In a cell: %load_ext neurolang
    # Then:      %%neurolang
    #            Predict species from iris
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

__all__ = ["nl", "NeurolangResult", "NeurolangCompileResult"]


@dataclass
class NeurolangResult:
    """Result of compiling and training a NeuroLang program."""
    accuracy: Optional[float] = None
    mse: Optional[float] = None
    final_loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    predictions: List[Dict[str, float]] = field(default_factory=list)
    epochs: int = 0
    early_stop_epoch: Optional[int] = None
    code: str = ""
    target: str = "tensorflow"
    compile_time_ms: float = 0.0
    train_time_ms: float = 0.0
    ir: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NeurolangCompileResult:
    """Result of compiling (without training) a NeuroLang program."""
    code: str = ""
    target: str = "tensorflow"
    ir: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


def _find_cli() -> str:
    """Locate the neurolang CLI binary."""
    # Check if neurolang is in PATH
    which = shutil.which("neurolang")
    if which:
        return which

    # Check common locations
    for candidate in [
        "npx neurolang",
        "npx tsx src/cli.ts",
    ]:
        return candidate

    raise FileNotFoundError(
        "NeuroLang CLI not found. Install with: npm install -g neurolang"
    )


def _data_to_csv(data: Any) -> str:
    """Convert pandas DataFrame or list of dicts to a temp CSV file path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    )

    # Handle pandas DataFrame
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            data.to_csv(tmp.name, index=False)
            return tmp.name
    except ImportError:
        pass

    # Handle list of dicts
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        keys = list(data[0].keys())
        writer = csv.DictWriter(tmp, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
        tmp.close()
        return tmp.name

    # Handle numpy array
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            headers = [f"col_{i}" for i in range(data.shape[1])]
            writer = csv.writer(tmp)
            writer.writerow(headers)
            for row in data:
                writer.writerow(row)
            tmp.close()
            return tmp.name
    except ImportError:
        pass

    raise TypeError(f"Unsupported data type: {type(data)}. Use a pandas DataFrame, list of dicts, or numpy array.")


def _to_list_of_dicts(data: Any) -> List[Dict[str, float]]:
    """Convert various data types to a list of dicts for JSON payload."""
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient="records")
    except ImportError:
        pass

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return data

    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            headers = [f"col_{i}" for i in range(data.shape[1])]
            return [dict(zip(headers, row)) for row in data]
    except ImportError:
        pass

    raise TypeError(f"Unsupported data type: {type(data)}")


def _interpolate_source(source: str, **kwargs: Any) -> str:
    """Interpolate Python variables into NeuroLang source."""
    for key, value in kwargs.items():
        placeholder = "{" + key + "}"
        if placeholder not in source:
            continue

        if isinstance(value, list):
            if all(isinstance(v, str) for v in value):
                source = source.replace(placeholder, " & ".join(value))
            else:
                source = source.replace(placeholder, " ".join(str(v) for v in value))
        else:
            source = source.replace(placeholder, str(value))

    return source


def _call_cli(
    source: str,
    *,
    run: bool = False,
    target: str = "tensorflow",
    data_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the NeuroLang CLI and return parsed JSON output."""
    # Build the source — if we have an external data file, replace the dataset reference
    if data_path:
        # If source doesn't have a dataset reference, append it
        if "dataset" not in source.lower() and "from" not in source.lower():
            source += f"\ndataset {data_path}"

    # Write source to temp file
    src_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".nl", delete=False
    )
    src_file.write(source)
    src_file.close()

    try:
        cmd_parts = ["npx", "tsx", "src/cli.ts", src_file.name, "--target", target]
        if run:
            cmd_parts.append("--run")

        # Find the neurolang project directory
        pkg_dir = _find_project_dir()

        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            cwd=pkg_dir,
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"NeuroLang compilation failed:\n{result.stderr or result.stdout}"
            )

        return {"stdout": result.stdout, "stderr": result.stderr}

    finally:
        os.unlink(src_file.name)


def _call_http(
    endpoint: str,
    payload: Dict[str, Any],
    *,
    base_url: str = "http://localhost:3000",
) -> Dict[str, Any]:
    """Call the NeuroLang HTTP API."""
    import urllib.request
    import urllib.error

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            err_data = json.loads(body)
            raise RuntimeError(f"NeuroLang API error: {err_data.get('error', body)}")
        except (json.JSONDecodeError, RuntimeError):
            raise RuntimeError(f"NeuroLang API error ({e.code}): {body}")
    except (urllib.error.URLError, ConnectionRefusedError, OSError):
        raise ConnectionError(
            f"NeuroLang server not running at {base_url}. "
            "Start it with: neurolang --serve"
        )


def _find_project_dir() -> str:
    """Find the NeuroLang project directory."""
    # Walk up from this file
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "package.json").exists():
            try:
                pkg = json.loads((current / "package.json").read_text())
                if pkg.get("name") == "neurolang":
                    return str(current)
            except (json.JSONDecodeError, KeyError):
                pass
        current = current.parent

    # Try common locations
    for candidate in [
        Path.home() / "neurolang",
        Path.cwd() / "neurolang",
        Path.cwd(),
    ]:
        if (candidate / "package.json").exists():
            return str(candidate)

    raise FileNotFoundError(
        "NeuroLang project directory not found. "
        "Set NEUROLANG_DIR environment variable."
    )


class _NL:
    """Main NeuroLang interface for Python."""

    def __init__(self):
        self._server_url: Optional[str] = os.environ.get("NEUROLANG_URL")
        self._project_dir: Optional[str] = os.environ.get("NEUROLANG_DIR")

    def __call__(
        self,
        source: str,
        *,
        data: Any = None,
        target: str = "tensorflow",
        epochs: Optional[int] = None,
        **kwargs: Any,
    ) -> NeurolangResult:
        """
        Compile and train a NeuroLang program.

        Args:
            source: NeuroLang source code (keyword or natural language syntax)
            data: Training data — pandas DataFrame, list of dicts, or numpy array
            target: Code generation target (tensorflow, pytorch, keras, jax)
            epochs: Override number of training epochs
            **kwargs: Variables to interpolate into the source

        Returns:
            NeurolangResult with accuracy, predictions, generated code, etc.
        """
        source = _interpolate_source(source, **kwargs)

        if epochs is not None:
            source += f"\nepochs {epochs}"

        # Try HTTP API first (handles inline data natively via JSON payload)
        if self._server_url:
            try:
                payload: Dict[str, Any] = {"source": source, "target": target}
                if data is not None:
                    # Convert pandas/numpy to list of dicts for JSON
                    payload["data"] = _to_list_of_dicts(data)
                raw = _call_http("run", payload, base_url=self._server_url)
                return self._parse_run_result(raw)
            except ConnectionError:
                pass

        # Fall back to CLI (needs CSV file for data)
        csv_path = None
        try:
            if data is not None:
                csv_path = _data_to_csv(data)
                source += f"\ndataset {csv_path}"

            raw = _call_cli(source, run=True, target=target)
            return self._parse_cli_result(raw)
        finally:
            if csv_path and os.path.exists(csv_path):
                os.unlink(csv_path)

    def compile(
        self,
        source: str,
        *,
        target: str = "tensorflow",
        **kwargs: Any,
    ) -> NeurolangCompileResult:
        """Compile without training — returns IR and generated code."""
        source = _interpolate_source(source, **kwargs)

        if self._server_url:
            try:
                raw = _call_http(
                    "compile",
                    {"source": source, "target": target},
                    base_url=self._server_url,
                )
                return NeurolangCompileResult(
                    code=raw.get("code", ""),
                    target=raw.get("target", target),
                    ir=raw.get("ir", {}),
                    raw=raw,
                )
            except ConnectionError:
                pass

        raw = _call_cli(source, target=target)
        return NeurolangCompileResult(
            code=raw.get("stdout", ""),
            target=target,
            raw=raw,
        )

    def pytorch(self, source: str, **kwargs: Any) -> str:
        """Generate PyTorch code from NeuroLang source."""
        return self.compile(source, target="pytorch", **kwargs).code

    def keras(self, source: str, **kwargs: Any) -> str:
        """Generate Keras code from NeuroLang source."""
        return self.compile(source, target="keras", **kwargs).code

    def jax(self, source: str, **kwargs: Any) -> str:
        """Generate JAX/Flax code from NeuroLang source."""
        return self.compile(source, target="jax", **kwargs).code

    def model(self, source: str, **kwargs: Any) -> Callable:
        """
        Decorator that trains a model at class definition time.

        @nl.model("Predict species from iris epochs 30")
        class IrisClassifier:
            pass

        print(IrisClassifier.accuracy)
        print(IrisClassifier.predictions)
        """
        def decorator(cls: type) -> type:
            result = self(source, **kwargs)
            cls.result = result
            cls.accuracy = result.accuracy
            cls.mse = result.mse
            cls.predictions = result.predictions
            cls.loss_history = result.loss_history
            cls.epochs = result.epochs
            cls.code = result.code
            cls.ir = result.ir
            cls.final_loss = result.final_loss

            original_init = cls.__init__ if hasattr(cls, "__init__") else None

            def __init__(self_instance, *args, **kw):
                self_instance.result = result
                self_instance.accuracy = result.accuracy
                self_instance.mse = result.mse
                self_instance.predictions = result.predictions
                if original_init and original_init is not object.__init__:
                    original_init(self_instance, *args, **kw)

            cls.__init__ = __init__
            return cls
        return decorator

    def _parse_run_result(self, raw: Dict[str, Any]) -> NeurolangResult:
        return NeurolangResult(
            accuracy=raw.get("accuracy"),
            mse=raw.get("mse"),
            final_loss=raw.get("finalLoss", 0),
            loss_history=raw.get("lossHistory", []),
            predictions=raw.get("predictions", []),
            epochs=raw.get("epochs", 0),
            early_stop_epoch=raw.get("earlyStopEpoch"),
            code=raw.get("code", ""),
            target=raw.get("target", "tensorflow"),
            ir=raw.get("ir", {}),
            raw=raw,
        )

    def _parse_cli_result(self, raw: Dict[str, Any]) -> NeurolangResult:
        stdout = raw.get("stdout", "")

        accuracy = None
        mse = None
        for line in stdout.split("\n"):
            if "accuracy" in line.lower():
                try:
                    accuracy = float(line.split(":")[-1].strip().rstrip("%")) / 100
                except (ValueError, IndexError):
                    pass
            if "mse" in line.lower():
                try:
                    mse = float(line.split(":")[-1].strip())
                except (ValueError, IndexError):
                    pass

        return NeurolangResult(
            accuracy=accuracy,
            mse=mse,
            code=stdout,
            raw=raw,
        )


# Singleton instance
nl = _NL()


# ─── Jupyter Magic ─────────────────────────────────────────────

def load_ipython_extension(ipython: Any) -> None:
    """Register %%neurolang magic in Jupyter/IPython."""
    from IPython.core.magic import register_cell_magic

    @register_cell_magic
    def neurolang(line: str, cell: str) -> None:
        """
        Jupyter cell magic for NeuroLang.

        Usage:
            %%neurolang
            Predict species from iris epochs 30

            %%neurolang --target pytorch
            Predict species from iris
        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--target", default="tensorflow")
        parser.add_argument("--compile-only", action="store_true")
        args = parser.parse_args(line.split() if line else [])

        if args.compile_only:
            result = nl.compile(cell.strip(), target=args.target)
            print(result.code)
        else:
            result = nl(cell.strip(), target=args.target)
            if result.accuracy is not None:
                print(f"Accuracy: {result.accuracy:.4f}")
            if result.mse is not None:
                print(f"MSE: {result.mse:.6f}")
            if result.final_loss:
                print(f"Final Loss: {result.final_loss:.6f}")
            if result.predictions:
                print(f"\nPredictions (first {len(result.predictions)}):")
                for p in result.predictions:
                    print(f"  actual={p.get('actual', '?')}, predicted={p.get('predicted', '?')}")

    print("NeuroLang magic loaded. Use %%neurolang in cells.")
