import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import arg_confidence, outcome  # noqa: E402
from score import canon, correct  # noqa: E402

REF = json.dumps([{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}])


def test_exact_and_optional():
    assert correct([{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}], REF)
    assert correct([{"name": "calculate_triangle_area", "arguments": {"base": 10.0, "height": 5, "unit": "Units"}}], REF)


def test_wrong_value_name_or_extra_arg():
    assert not correct([{"name": "calculate_triangle_area", "arguments": {"base": 15, "height": 5}}], REF)
    assert not correct([{"name": "triangle_area", "arguments": {"base": 10, "height": 5}}], REF)
    assert not correct([{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5, "colour": "red"}}], REF)
    assert not correct([{"name": "calculate_triangle_area", "arguments": {"base": 10}}], REF)


def test_irrelevance_and_unparseable():
    assert correct([], "[]")
    assert not correct([{"name": "x", "arguments": {}}], "[]")
    assert not correct(None, REF)


def test_canon_ignores_order_case_and_int_float():
    a = [{"name": "f", "arguments": {"city": "New York", "n": 2.0}}]
    b = [{"name": "f", "arguments": {"n": 2, "city": "new york"}}]
    assert canon(a) == canon(b)
    assert canon(a) != canon([{"name": "f", "arguments": {"n": 3, "city": "new york"}}])


def test_arg_confidence_uses_argument_words():
    words = [{"w": "base", "p": 0.99}, {"w": "of", "p": 0.4}, {"w": "15", "p": 0.55}, {"w": "units.", "p": 0.98}]
    calls = [{"name": "f", "arguments": {"base": 15}}]
    assert arg_confidence(calls, words) == 0.55  # "of" is low but not in the call


def test_confirmation_outcome():
    rows = [
        {"acts": True, "top_ok": False, "oracle_ok": True, "oracle_acts": True},   # misheard, caught if confirmed
        {"acts": True, "top_ok": True, "oracle_ok": True, "oracle_acts": True},    # fine
        {"acts": False, "top_ok": False, "oracle_ok": True, "oracle_acts": True},  # no action: nothing to confirm
    ]
    w, c = outcome(rows, [False, False, False])
    assert w.tolist() == [1, 0, 0] and c.sum() == 0
    w, c = outcome(rows, [True, True, True], "hearing")
    assert w.tolist() == [0, 0, 0] and c.tolist() == [1, 1, 0]
