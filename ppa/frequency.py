"""
Copyright 2025 John D. Reynolds, All Rights Reserved.

This software and associated documentation files (the “Software”) are provided for reference or
archival purposes only. No license is granted to any person to copy, modify, publish, distribute,
sublicense, or sell copies of the Software, or to use it for any commercial or non-commercial
purpose, without the express prior written permission of John D. Reynolds.

This module contains the Frequency Enum and its associated methods.
"""

# Pyton imports
import calendar
import datetime as dt
from enum import Enum

# Project imports
import ppa.errors as errs


class Frequency(Enum):
    """
    An enumeration of the time-period frequencies.

    Args:
        Enum (_type_): enum
    """

    AS_OFTEN_AS_POSSIBLE = "Periodic"  # As often as possible based on the frequency of the data.
    MONTHLY = "Monthly"  # Currently only supports calendar month-end, not business month-end.
    QUARTERLY = "Quarterly"  # Calendar quarters.
    YEARLY = "Yearly"  # Calendar years.


def date_matches_frequency(date: dt.date, frequency: Frequency) -> bool:
    """
    Determines if the date matches the frequency.

    Args:
        date (dt.date): The date.
        frequency (Frequency): The frequency.

    Returns:
        bool: True if the date matches the frequency, otherwise False.
    """
    match frequency:
        case Frequency.AS_OFTEN_AS_POSSIBLE:
            return True
        case Frequency.MONTHLY:
            return _is_calendar_month_end(date)
        case Frequency.QUARTERLY:
            return date.month in (3, 6, 9, 12) and _is_calendar_month_end(date)
        case Frequency.YEARLY:
            return date.month == 12 and _is_calendar_month_end(date)


def _is_calendar_month_end(date: dt.date) -> bool:
    """
    Determines if the date parameter is a calendar month-end date.

    Args:
        date (dt.date): The date.

    Returns:
        bool: True if the date is a calendar month-end date.  Otherwise False.
    """
    if date.day < 28:
        return False
    return date.day == calendar.monthrange(date.year, date.month)[1]


def periods_per_year(frequency: Frequency) -> int:
    """
    Calculates the periods per year.

    Args:
        frequency (Frequency): The freuency.

    Returns:
        int: The periods per year.
    """
    match frequency:
        case Frequency.MONTHLY:
            return 12
        case Frequency.QUARTERLY:
            return 4
        case Frequency.YEARLY:
            return 1
        case _:  # frequncy.AS_OFTEN_AS_POSSIBLE
            raise errs.PpaError(f"{errs.ERROR_999_UNEXPECTED}Unhandled Frequency {frequency}")
