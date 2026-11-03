"""Test design notes for solver-output and GSM8K answer parsing."""

import pytest

from self_play_judge.parsing import ParsedResponse, ParseMathResponse


def test_from_json_parses_shared_response_fields() -> None:
    response = ParsedResponse.from_json(
        '{"reasoning": "calculation", "answer": "1,000"}'
    )

    assert response.reasoning == "calculation"
    assert response.answer == "1,000"


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "[]",
        '{"reasoning": "work"}',
        '{"reasoning": 3, "answer": "3"}',
    ],
)
def test_from_json_rejects_invalid_response_shapes(response: str) -> None:
    with pytest.raises(ValueError):
        ParsedResponse.from_json(response)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("  1,000  ", "1000"),
        ("-0012", "-12"),
        ("+12", "12"),
        ("12.500", "25/2"),
        ("0.0", "0"),
        ("3/4", "3/4"),
        ("0.75", "3/4"),
        ("1e3", "1000"),
        ("1.2e2 / 3e1", "4"),
        ("1e3/2", "500"),
        ("1,000/2", "500"),
        ("-0/3", "0"),
    ],
)
def test_math_response_normalizes_numeric_formatting(
    answer: str, expected: str
) -> None:
    response = ParseMathResponse(reasoning="work", answer=answer)

    assert response.answer == expected


@pytest.mark.parametrize(
    "answer",
    [
        "not a number",
        "3/0",
        "1/2/3",
        "1,00",
        "1e1001",
        "1e3/2e-1001",
        "1" * 257,
    ],
)
def test_math_response_rejects_invalid_or_excessive_answers(answer: str) -> None:
    with pytest.raises(ValueError):
        ParseMathResponse(reasoning="work", answer=answer)


def test_math_response_from_json_applies_math_normalization() -> None:
    response = ParseMathResponse.from_json(
        '{"reasoning": "calculation", "answer": "1,000.00"}'
    )

    assert isinstance(response, ParseMathResponse)
    assert response.answer == "1000"
