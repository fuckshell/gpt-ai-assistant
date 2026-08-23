"""Symbolic discovery wrapper with dimensional-consistency filtering."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from modules.json_util import json_safe

try:
    from pysr import PySRRegressor
except ImportError:  # pragma: no cover - depends on local environment
    PySRRegressor = None


@dataclass(frozen=True)
class Dimension:
    meter: float = 0.0
    second: float = 0.0
    free: bool = False

    def __mul__(self, other: "Dimension") -> "Dimension":
        # Free coefficients absorb unknown units until constrained by addition/target.
        if self.free or other.free:
            return Dimension(free=True)
        return Dimension(self.meter + other.meter, self.second + other.second)

    def __truediv__(self, other: "Dimension") -> "Dimension":
        if self.free or other.free:
            return Dimension(free=True)
        return Dimension(self.meter - other.meter, self.second - other.second)

    def __pow__(self, exponent: float) -> "Dimension":
        if self.free:
            return Dimension(free=True)
        return Dimension(self.meter * exponent, self.second * exponent)

    def compatible(self, other: "Dimension") -> bool:
        return self.free or other.free or (
            self.meter == other.meter and self.second == other.second
        )

    def unify(self, other: "Dimension") -> "Dimension":
        if self.free:
            return other
        if other.free:
            return self
        if not self.compatible(other):
            raise DimensionError("Incompatible dimensions")
        return self


DIMENSIONLESS = Dimension()
POSITION_DIM = Dimension(meter=1.0, second=0.0)
TIME_DIM = Dimension(meter=0.0, second=1.0)
FREE = Dimension(free=True)


@dataclass
class DiscoveryCandidate:
    equation: str
    loss: float
    complexity: float
    dimensionally_valid: bool
    adjusted_loss: float

    def to_dict(self) -> dict:
        return json_safe(asdict(self))


class DimensionError(ValueError):
    pass


def _func_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    raise DimensionError("Unsupported call target in equation")


def _parse_dimension(node: ast.AST, variable_dims: Dict[str, Dimension]) -> Dimension:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            # Numeric literals are free-dimension coefficients/parameters.
            return FREE
        raise DimensionError("Non-numeric constants are unsupported")

    if isinstance(node, ast.Name):
        if node.id not in variable_dims:
            raise DimensionError(f"Unknown variable '{node.id}'")
        return variable_dims[node.id]

    if isinstance(node, ast.UnaryOp):
        return _parse_dimension(node.operand, variable_dims)

    if isinstance(node, ast.BinOp):
        left = _parse_dimension(node.left, variable_dims)
        right = _parse_dimension(node.right, variable_dims)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return left.unify(right)
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant):
                raise DimensionError("Exponent must be a numeric constant")
            exponent = float(node.right.value)
            return left**exponent
        raise DimensionError("Unsupported binary operator")

    if isinstance(node, ast.Call):
        fn = _func_name(node)
        if fn == "pow":
            if len(node.args) != 2:
                raise DimensionError("pow expects 2 arguments")
            base_dim = _parse_dimension(node.args[0], variable_dims)
            if not isinstance(node.args[1], ast.Constant):
                raise DimensionError("pow exponent must be dimensionless constant")
            return base_dim ** float(node.args[1].value)

        if len(node.args) != 1:
            raise DimensionError(f"Function '{fn}' must have 1 argument")
        arg_dim = _parse_dimension(node.args[0], variable_dims)

        if fn in {"sin", "cos", "tan", "exp", "log"}:
            # Require dimensionless or free (coefficients absorbed into free constants).
            if not (arg_dim.free or arg_dim == DIMENSIONLESS):
                raise DimensionError(f"{fn} argument must be dimensionless")
            return FREE if fn in {"exp", "log"} else FREE
        if fn == "sqrt":
            return arg_dim**0.5
        raise DimensionError(f"Unsupported function '{fn}'")

    raise DimensionError("Unsupported expression node")


def default_variable_dims() -> Dict[str, Dimension]:
    # PySR names the single time feature x0 by default; also accept t.
    return {"x0": TIME_DIM, "t": TIME_DIM}


def is_dimensionally_consistent(
    equation: str,
    variable_dims: Dict[str, Dimension] | None = None,
    target_dim: Dimension = POSITION_DIM,
) -> bool:
    variable_dims = variable_dims or default_variable_dims()
    normalized = equation.replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
        dim = _parse_dimension(tree.body, variable_dims)
        return dim.compatible(target_dim)
    except (SyntaxError, DimensionError):
        return False


def _make_model(niterations: int, dimensional_constraint_penalty: float) -> "PySRRegressor":
    if PySRRegressor is None:  # pragma: no cover - depends on local environment
        raise ImportError(
            "PySR is not installed. Install optional discovery deps with "
            "`pip install -r requirements-newton-discovery.txt`."
        )

    kwargs = {
        "niterations": niterations,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["sin", "cos", "exp"],
        "model_selection": "best",
        "maxsize": 30,
        "progress": False,
    }
    try:
        return PySRRegressor(
            dimensional_constraint_penalty=dimensional_constraint_penalty,
            **kwargs,
        )
    except TypeError:
        return PySRRegressor(**kwargs)


def discover_equation(
    t_data: Iterable[float],
    x_data: Iterable[float],
    niterations: int = 200,
    dimensional_constraint_penalty: float = 10.0,
    variable_dims: Dict[str, Dimension] | None = None,
    target_dim: Dimension = POSITION_DIM,
) -> dict:
    """Run symbolic search and rank formulas with dimensional penalty."""
    t_arr = np.asarray(list(t_data), dtype=float)
    x_arr = np.asarray(list(x_data), dtype=float)
    if t_arr.ndim != 1 or x_arr.ndim != 1:
        raise ValueError("t_data and x_data must be 1D")
    if t_arr.shape != x_arr.shape:
        raise ValueError("t_data and x_data must have the same shape")
    if t_arr.size < 5:
        raise ValueError("Need at least 5 points for symbolic discovery")
    if not np.all(np.isfinite(t_arr)) or not np.all(np.isfinite(x_arr)):
        raise ValueError("Discovery inputs contain NaN/Inf")

    model = _make_model(
        niterations=niterations,
        dimensional_constraint_penalty=dimensional_constraint_penalty,
    )
    X = t_arr.reshape(-1, 1)
    model.fit(X, x_arr)

    variable_dims = variable_dims or default_variable_dims()
    equations_df = model.equations_
    candidates: List[DiscoveryCandidate] = []
    for _, row in equations_df.iterrows():
        equation = str(row.get("equation", ""))
        loss = float(row.get("loss", np.inf))
        complexity = float(row.get("complexity", np.inf))
        valid = is_dimensionally_consistent(
            equation=equation,
            variable_dims=variable_dims,
            target_dim=target_dim,
        )
        adjusted_loss = loss if valid else loss + dimensional_constraint_penalty
        candidates.append(
            DiscoveryCandidate(
                equation=equation,
                loss=loss,
                complexity=complexity,
                dimensionally_valid=valid,
                adjusted_loss=adjusted_loss,
            )
        )

    candidates.sort(key=lambda item: (item.adjusted_loss, item.complexity))
    valid_candidates = [item for item in candidates if item.dimensionally_valid]
    best: Optional[DiscoveryCandidate] = valid_candidates[0] if valid_candidates else None

    return json_safe(
        {
            "best": best.to_dict() if best else None,
            "candidates": [item.to_dict() for item in candidates],
            "valid_count": len(valid_candidates),
        }
    )
