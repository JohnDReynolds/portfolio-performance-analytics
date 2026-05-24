"""
This module contains numbered errors and the PpaError class
"""

ERRORS = {
    # Performance Class Error Messages
    102: "Error 102: Ending dates are not unique ",
    103: "Error 103: No performance rows found ",
    104: "Error 104: There are missing values ",
    105: "Error 105: Beginning dates not less than ending dates ",
    106: "Error 106: There are discontinuous time periods ",
    107: "Error 107: The return columns (.ret) are not equal to the weight columns (.wgt) ",
    108: "Error 108: The weights do not sum to 1.0 ",
    109: "Error 109: There are no return columns (.ret) or weight columns (.wgt) ",
    110: "Error 110: Invalid Performance data format ",
    111: "Error 111: Beginning Date cannot be after Ending Date: ",
    112: "Error 112: Duplicate rows for the same period and identifier ",
    # Attribution Class Error Messages
    202: "Error 202: There are no common reportable dates found ",
    203: "Error 203: A return less than zero is undefined. ",
    204: "Error 204: Too many rows to produce 'great_table' html: ",
    # Analytics Class Error Messages
    252: "Error 252: Must specify classification_name",
    # Classification Class Error Messages
    302: "Error 302: The Classification DataFrame must contain at least 2 columns.",
    # Mapping Class Error Messages
    353: "Error 353: The Mapping DataFrame must contain at least 2 columns.",
    # Ex-Post Risk Statistics Class Error Messages
    402: "Error 402: Invalid frequency for ex-Post risk statistics: ",
    403: "Error 403: Insufficient quantity of returns for ex-Post risk statistics: ",
    404: "Error 404: The qty of portfolio returns <> the qty of the benchmark returns: ",
    405: "Error 405: The portfolio returns or benchmark returns have NaN values.",
    # Axys Errors
    502: "Error 502: Missing required column(s): ",
    503: "Error 503: Could not derive weights for secperf: ",
    504: "Error 504: Bad specifications file: ",
    505: "Error 505: Portperf and secperf have no common periods.",
    # General Error Messages
    802: "Error 802: File path does not exist: ",
    803: "Error 803: Cannot convert to a date. ",
    804: "Error 804: Missing data source.",
    # Unexpected Logic Error Message
    999: "Error 999: Unexpected Logic error: ",
}


class PpaError(Exception):
    """Custom Portfolio Analytics error class.

    Attributes:
        message: Human-readable error message.
        code: Optional integer error code for programmatic handling.
    """

    def __init__(self, message: str, code: int | None) -> None:
        """Initialize the error.

        Args:
            message: Description of the error.
            code: Optional integer error code.
        """
        if code is not None:
            message = f"{ERRORS[code]}{message}"
        super().__init__(message)
