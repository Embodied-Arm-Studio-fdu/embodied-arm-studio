"""Versioned, non-executable student reward expressions."""

import ast
import hashlib
import math
from typing import Mapping


REWARD_EXTENSION_SCHEMA = "reward-expression/v1"
TASK_TERMS = {
    "s1_push": (
        "precontact_progress", "wrong_side_or_alignment_term",
        "directional_push_progress", "lateral_push_penalty",
        "contact_sustain_bonus", "action_penalty",
    ),
    "s1_reach": ("progress_reward", "action_penalty"),
    "s1_place": ("xy_progress", "target_zone", "height_stability", "grasp_stability", "action_failure"),
}
PROTECTED_TERMS = ("success_bonus", "joint_limit_penalty")


def starter_source(task_id):
    names = TASK_TERMS[task_id]
    return "def combine_reward(terms):\n    return " + " + ".join(
        'terms["%s"]' % name for name in names
    ) + "\n"


class RewardExpression:
    def __init__(self, task_id, source):
        if task_id not in TASK_TERMS:
            raise ValueError("Reward editing is supported only for Push, Reach and Place")
        if not isinstance(source, str) or len(source.encode("utf-8")) > 8192:
            raise ValueError("Reward source must be text of at most 8 KiB")
        try:
            module = ast.parse(source)
        except SyntaxError as exc:
            raise ValueError("Syntax error on line %s: %s" % (exc.lineno, exc.msg)) from exc
        except (RecursionError, MemoryError) as exc:
            raise ValueError("Reward source is too complex") from exc
        if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
            raise ValueError("Only the combine_reward function is allowed")
        function = module.body[0]
        arguments = function.args
        if (
            function.name != "combine_reward" or function.decorator_list
            or function.returns is not None or len(arguments.args) != 1
            or arguments.args[0].arg != "terms" or arguments.args[0].annotation
            or arguments.posonlyargs or arguments.kwonlyargs
            or arguments.vararg or arguments.kwarg or arguments.defaults
            or len(function.body) != 1 or not isinstance(function.body[0], ast.Return)
        ):
            raise ValueError("Keep the signature combine_reward(terms) and one return expression")
        self.task_id = task_id
        self.source = source
        self.digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        self.expression = function.body[0].value
        nodes = list(ast.walk(self.expression))
        if len(nodes) > 128:
            raise ValueError("Reward expression is too complex")
        self._validate(self.expression)

    def _validate(self, node):
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float) or abs(node.value) > 10 or not math.isfinite(node.value):
                raise ValueError("Coefficients must be finite numbers between -10 and 10")
        elif isinstance(node, ast.Subscript):
            key = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
            if (
                not isinstance(node.value, ast.Name) or node.value.id != "terms"
                or not isinstance(key, ast.Constant) or key.value not in TASK_TERMS[self.task_id]
            ):
                raise ValueError("Only the task's editable reward terms may be referenced")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            self._validate(node.left)
            self._validate(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            self._validate(node.operand)
        else:
            raise ValueError("Only reward terms, coefficients and + - * / are allowed")

    def evaluate(self, terms: Mapping[str, float]) -> float:
        def calculate(node):
            if isinstance(node, ast.Constant):
                return float(node.value)
            if isinstance(node, ast.Subscript):
                key = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
                return float(terms[key.value])
            if isinstance(node, ast.UnaryOp):
                value = calculate(node.operand)
                return -value if isinstance(node.op, ast.USub) else value
            left, right = calculate(node.left), calculate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if abs(right) < 1e-8:
                raise ValueError("Reward division denominator is zero or too small")
            return left / right

        try:
            value = calculate(self.expression)
            if not math.isfinite(value) or abs(value) > 1000:
                raise ValueError("Reward must be finite and within [-1000, 1000]")
            protected = sum(float(terms[name]) for name in PROTECTED_TERMS)
            total = value + protected
            if not math.isfinite(total):
                raise ValueError("Protected reward terms must be finite")
            return total
        except (KeyError, OverflowError, ZeroDivisionError) as exc:
            raise ValueError("Reward expression could not be evaluated: %s" % exc) from exc


def validate_extension(task_id, extension):
    if not isinstance(extension, dict) or set(extension) != {"schema", "source", "sha256"}:
        raise ValueError("Invalid reward extension fields")
    if extension["schema"] != REWARD_EXTENSION_SCHEMA:
        raise ValueError("Unsupported reward extension schema")
    expression = RewardExpression(task_id, extension["source"])
    if extension["sha256"] != expression.digest:
        raise ValueError("Reward source digest mismatch")
    return expression