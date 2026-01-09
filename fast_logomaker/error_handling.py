"""
Error handling utilities for Fast Logomaker.
"""

import warnings


class LogomakerError(Exception):
    """
    Custom exception class for Fast Logomaker errors.

    Parameters
    ----------
    message: str
        The error message.
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


def check(condition: bool, message: str, warn: bool = False):
    """
    Checks a condition; raises a LogomakerError with message if condition
    evaluates to False.

    Parameters
    ----------
    condition: bool
        A condition that, if false, halts execution and raises a
        clean error to user.
    message: str
        The string to show user if condition is False.
    warn: bool
        If True, warn instead of raising error. Default is False.

    Returns
    -------
    None
    """
    if not condition:
        if warn:
            warnings.warn(message)
        else:
            raise LogomakerError(message)
