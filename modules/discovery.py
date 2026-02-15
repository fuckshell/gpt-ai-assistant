"""Symbolic discovery wrapper with dimensional-consistency filtering."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

import numpy as np

try:
    from pysr import PySRRegressor
except ImportError:  # pragma: no cover - depends on local environment
    PySRRegressor = None


@dataclass(frozen=True)
class Dimension:
    meter: float = 0.0
    second: float = 0.0

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(self.meter + other.meter, self.second + other.second)

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(self.meter - other.meter, self.second - other.second)

    def __pow__(self, exponent: float) -> "Dimension":
        return Dimension(self.meter * exponent, self.second * exponent)


DIMENSIONLESS = Dimension()
POSITION_DIM = Dimension(meter=1.0, second=0.0)
TIME_DIM = Dimension(meter=0.0, second=1.0)


@dataclass
class DiscoveryCandidate:
    equation: str
    loss: float
    complexity: float
    dimensionally_valid: bool
    adjusted_loss: float

    def to_dict(self) -> dict:
        return asdict(self)


class DimensionError(ValueError):
    pass


def _func_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    raise DimensionError("Unsupported call target in equation")


def _parse_dimension(node: ast.AST, variable_dims: Dict[str, Dimension]) -> Dimension:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return DIMENSIONLESS
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
            if left != right:
                raise DimensionError("Addition/Subtraction requires equal dimensions")
            return left
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            if right != DIMENSIONLESS or not isinstance(node.right, ast.Constant):
                raise DimensionError("Exponent must be a dimensionless numeric constant")
            exponent = float(node.right.value)
            return left**exponent
        raise DimensionError("Unsupported binary operator")

    if isinstance(node, ast.Call):
        fn = _func_name(node)
        if fn == "pow":
            if len(node.args) != 2:
                raise DimensionError("pow expects 2 arguments")
            base_dim = _parse_dimension(node.args[0], variable_dims)
            exponent_dim = _parse_dimension(node.args[1], variable_dims)
            if exponent_dim != DIMENSIONLESS or not isinstance(node.args[1], ast.Constant):
                raise DimensionError("pow exponent must be dimensionless constant")
            return base_dim ** float(node.args[1].value)

        if len(node.args) != 1:
            raise DimensionError(f"Function '{fn}' must have 1 argument")
        arg_dim = _parse_dimension(node.args[0], variable_dims)

        if fn in {"sin", "cos", "tan", "exp", "log"}:
            if arg_dim != DIMENSIONLESS:
                raise DimensionError(f"{fn} argument must be dimensionless")
            return DIMENSIONLESS
        if fn == "sqrt":
            return arg_dim**0.5
        raise DimensionError(f"Unsupported function '{fn}'")

    raise DimensionError("Unsupported expression node")


def is_dimensionally_consistent(
    equation: str,
    variable_dims: Dict[str, Dimension] | None = None,
    target_dim: Dimension = POSITION_DIM,
) -> bool:
    variable_dims = variable_dims or {"x0": TIME_DIM, "t": TIME_DIM}
    normalized = equation.replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
        dim = _parse_dimension(tree.body, variable_dims)
        return dim == target_dim
    except (SyntaxError, DimensionError):
        return False


def _make_model(niterations: int, dimensional_constraint_penalty: float) -> "PySRRegressor":
    if PySRRegressor is None:  # pragma: no cover - depends on local environment
        raise ImportError("PySR is not installed. Install with `pip install pysr`.")

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
    if t_arr.shape != x_arr.shape:
        raise ValueError("t_data and x_data must have the same shape")
    if t_arr.size < 5:
        raise ValueError("Need at least 5 points for symbolic discovery")

    model = _make_model(
        niterations=niterations,
        dimensional_constraint_penalty=dimensional_constraint_penalty,
    )
    X = t_arr.reshape(-1, 1)
    model.fit(X, x_arr)

    variable_dims = variable_dims or {"x0": TIME_DIM, "t": TIME_DIM}
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
    best = candidates[0] if candidates else None

    return {
        "best": best.to_dict() if best else None,
        "candidates": [item.to_dict() for item in candidates],
    }

