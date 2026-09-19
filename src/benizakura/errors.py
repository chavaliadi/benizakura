from __future__ import annotations


class BenizakuraJudgeError(Exception):
    """Base exception for all judge-related failures in Benizakura."""
    pass


class JudgeParseError(BenizakuraJudgeError):
    """Raised when raw model output cannot be parsed as valid JSON."""
    pass


class JudgeValidationError(BenizakuraJudgeError):
    """Raised when parsed JSON violates the structured evaluation contract or domain schema."""
    pass


class JudgeProviderError(BenizakuraJudgeError):
    """Raised when the underlying model provider fails to generate a response."""
    pass
