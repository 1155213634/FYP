"""Parse model outputs and normalize final answers for exact-match checks."""

import json
import re


# Base parser (assume able to output json dict for now)
class ParsedResponse:
    """A structured representation of a model's response, including reasoning and final answer."""

    def __init__(self, reasoning: str, answer: str) -> None:
        self.reasoning = reasoning
        self.answer = answer

    @classmethod
    def from_json(cls, response: str) -> "ParsedResponse":
        """Parse a JSON object containing string-valued reasoning and answer fields."""
        try:
            data = json.loads(response)
        except json.JSONDecodeError as error:
            raise ValueError("Response must be valid JSON.") from error

        if not isinstance(data, dict):
            raise ValueError("Response JSON must be an object.")

        reasoning = data.get("reasoning")
        answer = data.get("answer")
        if not isinstance(reasoning, str) or not isinstance(answer, str):
            raise ValueError(
                "Response JSON must contain string-valued 'reasoning' and 'answer' fields."
            )

        return cls(reasoning=reasoning, answer=answer)


# Math stuff
from decimal import Decimal, InvalidOperation
from fractions import Fraction

_DIGITS = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
_MANTISSA = rf"(?:{_DIGITS}(?:\.\d*)?|\.\d+)"
_SCIENTIFIC_NUMBER = rf"[+-]?{_MANTISSA}(?:[eE][+-]?\d+)?"
_NUMERIC_ANSWER = re.compile(rf"^{_SCIENTIFIC_NUMBER}(?:\s*/\s*{_SCIENTIFIC_NUMBER})?$")
_EXPONENT = re.compile(r"[eE]([+-]?\d+)")
_MAX_ANSWER_LENGTH = 256
_MAX_ABS_EXPONENT = 1000


class ParseMathResponse(ParsedResponse):
    """A parsed response whose final answer is normalized for math exact-match checks."""

    def __init__(self, reasoning: str, answer: str) -> None:
        super().__init__(reasoning, answer)
        self.answer = self.normalize_math_answer(self.answer)

    @staticmethod
    def normalize_math_answer(answer: str) -> str:
        """Normalize a numeric scalar or ratio using exact rational arithmetic."""
        normalized = " ".join(answer.strip().split()).lower()
        if len(normalized) > _MAX_ANSWER_LENGTH:
            raise ValueError("Numeric answer is too long.")
        if not _NUMERIC_ANSWER.fullmatch(normalized):
            raise ValueError("Response does not contain a valid numeric answer.")

        parts = re.split(r"\s*/\s*", normalized)
        if any(
            abs(int(match.group(1))) > _MAX_ABS_EXPONENT
            for part in parts
            if (match := _EXPONENT.search(part)) is not None
        ):
            raise ValueError("Scientific exponent is outside the supported range.")

        try:
            values = [Fraction(Decimal(part.replace(",", ""))) for part in parts]
        except (InvalidOperation, ValueError) as error:
            raise ValueError(
                "Response does not contain a valid numeric answer."
            ) from error

        if len(values) == 2:
            if values[1] == 0:
                raise ValueError("A fraction denominator cannot be zero.")
            value = values[0] / values[1]
        else:
            value = values[0]

        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"


class GSM8KAnswerParser(ParseMathResponse):
    """
    Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72
    """

    def __init__(self, GSM8K_answer: str):
        reasoning, answer = self.GSM8K_answer_parse(GSM8K_answer)
        super().__init__(reasoning, answer)

    @staticmethod
    def GSM8K_answer_parse(GSM8K_answer: str):
        # "####" separates reasoning with answer
        return GSM8K_answer.split("####")
