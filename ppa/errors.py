"""
This module contains numbered errors and the PpaError Class
"""

# Performance Class Error Messages
ERROR_102_ENDING_DATES_ARE_NOT_UNIQUE = "Error 102: Ending dates are not unique "
ERROR_103_NO_PERFORMANCE_ROWS = "Error 103: No performance rows found "
ERROR_104_MISSING_VALUES = "Error 104: There are missing values "
ERROR_105_BEGINNING_DATES_GREATER_THAN_ENDING_DATES = (
    "Error 105: Beginning dates not less than ending dates "
)
ERROR_106_DISCONTINUOS_TIME_PERIODS = "Error 106: There are discontinuous time periods "
ERROR_107_RETURN_COLUMNS_NOT_EQUAL_TO_WEIGHT_COLUMNS = (
    "Error 107: The return columns (.ret) are not equal to the weight columns (.wgt) "
)
ERROR_108_WEIGHTS_DO_NOT_SUM_TO_1 = "Error 108: The weights do not sum to 1.0 "
ERROR_109_NO_RETURNS_OR_WEIGHTS = (
    "Error 109: There are no return columns (.ret) or weight columns (.wgt) "
)
ERROR_110_INVALID_PERFORMANCE_DATA_FORMAT = "Error 110: Invalid Performance data format "
ERROR_111_INVALID_DATES = "Error 111: Beginning Date cannot be after Ending Date: "


# Attribution Class Error Messages
ERROR_202_NO_REPORTABLE_DATES = "Error 202: There are no common reportable dates found "
ERROR_203_UNDEFINED_RETURN = "Error 203: A return less than zero is undefined.  "
ERROR_204_TOO_MANY_HTML_ROWS = "Error 204: Too many rows to produce 'great_table' html: "

# Analytics Class Error Messages
ERROR_252_MUST_SPECIFY_CLASSIFICATION_NAME = "Error 252: Must specify classification_name"

# Classification Class Error Messages
ERROR_302_CLASSIFICATION_MUST_CONTAIN_2_COLUMNS = (
    "Error 302: The Classification DataFrame must contain at least 2 columns."
)

# Mapping Class Error Messages
ERROR_353_MAPPING_MUST_CONTAIN_2_COLUMNS = (
    "Error 353: The Mapping DataFrame must contain at least 2 columns."
)

# Ex-Post Risk Statistics Class Error Messages
ERROR_402_INVALID_FREQUENCY = "Error 402: Invalid frequency for ex-Post risk statistics: "
ERROR_403_INSUFFICIENT_QUANTITY_OF_RETURNS = (
    "Error 403: Insufficient quantity of returns for ex-Post risk statistics: "
)
ERROR_404_PORTFOLIO_BENCHMARK_RETURNS_QTY_NOT_EQUAL = (
    "Error 404: The qty of portfolio returns is not equal to the qty of the benchmark returns: "
)
ERROR_405_NAN_VALUES = "Error 405: The portfolio returns or benchmark returns have NaN values."

# General Error Messages
ERROR_802_FILE_PATH_DOES_NOT_EXIST = "Error 802: File path does not exist: "
ERROR_803_CANNOT_CONVERT_TO_A_DATE = "Error 803: Cannot convert to a date.  "
ERROR_804_MISSING_DATA_SOURCE = "Error 804: Missing data source."

# Unexpected Logic Error Message
ERROR_999_UNEXPECTED = "Error 999: Unexpected Logic error: "


class PpaError(Exception):
    """
    The custom "Portfolio Analytics" error class.

    Args:
        Exception (_type_): exception
    """

    def __init__(self, value: str):
        self.value = value

    def __str__(self) -> str:
        return repr(self.value)
