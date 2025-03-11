"""
This module contains column names, column groupings and column functions.
"""

# All Column Names
ACTIVE_CONTRIB_SMOOTHED = "Active_Contribution_Smoothed"
ACTIVE_CONTRIB_SIMPLE = "Active_Contribution_Simple"
ACTIVE_RETURN = "Active_Return"
ACTIVE_WEIGHT = "Active_Weight"
ALLOCATION_EFFECT_SMOOTHED = "Allocation_Effect_Smoothed"
ALLOCATION_EFFECT_SIMPLE = "Allocation_Effect_Simple"
BEGINNING_DATE = "beginning_date"
BENCHMARK_CONTRIB_SMOOTHED = "Benchmark_Contribution_Smoothed"
BENCHMARK_CONTRIB_SIMPLE = "Benchmark_Contribution_Simple"
BENCHMARK_RETURN = "Benchmark_Return"
BENCHMARK_WEIGHT = "Benchmark_Weight"
CLASSIFICATION_IDENTIFIER = "Classification_Identifier"
CLASSIFICATION_NAME = "Classification_Name"
CUMULATIVE_ACTIVE_CONTRIB = "Cumulative_Active_Contribution"
CUMULATIVE_ACTIVE_RETURN = "Cumulative_Active_Return"
CUMULATIVE_ALLOCATION_EFFECT = "Cumulative_Allocation_Effect"
CUMULATIVE_TOTAL_EFFECT = "Cumulative_Total_Effect"
CUMULATIVE_BENCHMARK_CONTRIB = "Cumulative_Benchmark_Contribution"
CUMULATIVE_BENCHMARK_RETURN = "Cumulative_Benchmark_Return"
CUMULATIVE_PORTFOLIO_CONTRIB = "Cumulative_Portfolio_Contribution"
CUMULATIVE_PORTFOLIO_RETURN = "Cumulative_Portfolio_Return"
CUMULATIVE_SELECTION_EFFECT = "Cumulative_Selection_Effect"
ENDING_DATE = "ending_date"
IDENTIFIER = "identifier"
NAME = "name"
PORTFOLIO_CONTRIB_SMOOTHED = "Portfolio_Contribution_Smoothed"
PORTFOLIO_CONTRIB_SIMPLE = "Portfolio_Contribution_Simple"
PORTFOLIO_RETURN = "Portfolio_Return"
PORTFOLIO_WEIGHT = "Portfolio_Weight"
QUANTITY_OF_DAYS = "Quantity_Of_Days"
RETURN = "return"
SELECTION_EFFECT_SMOOTHED = "Selection_Effect_Smoothed"
SELECTION_EFFECT_SIMPLE = "Selection_Effect_Simple"
TOTAL_EFFECT_SMOOTHED = "Total_Effect_Smoothed"
TOTAL_EFFECT_SIMPLE = "Total_Effect_Simple"
TOTAL_RETURN = "Total_Return"
WEIGHT = "weight"

# Column Groupings
ACTIVE_COLUMNS_SMOOTHED = [ACTIVE_WEIGHT, ACTIVE_RETURN, ACTIVE_CONTRIB_SMOOTHED]
ACTIVE_COLUMNS_SIMPLE = [ACTIVE_WEIGHT, ACTIVE_RETURN, ACTIVE_CONTRIB_SIMPLE]
ATTRIBUTION_COLUMNS_SMOOTHED = [
    ALLOCATION_EFFECT_SMOOTHED,
    SELECTION_EFFECT_SMOOTHED,
    TOTAL_EFFECT_SMOOTHED,
]
ATTRIBUTION_COLUMNS_SIMPLE = [
    ALLOCATION_EFFECT_SIMPLE,
    SELECTION_EFFECT_SIMPLE,
    TOTAL_EFFECT_SIMPLE,
]
BENCHMARK_COLUMNS_SMOOTHED = [BENCHMARK_WEIGHT, BENCHMARK_RETURN, BENCHMARK_CONTRIB_SMOOTHED]
BENCHMARK_COLUMNS_SIMPLE = [BENCHMARK_WEIGHT, BENCHMARK_RETURN, BENCHMARK_CONTRIB_SIMPLE]
CLASSIFICATION_COLUMNS = [CLASSIFICATION_IDENTIFIER, CLASSIFICATION_NAME]
CONTRIBUTION_COLUMNS_SMOOTHED = [
    PORTFOLIO_CONTRIB_SMOOTHED,
    BENCHMARK_CONTRIB_SMOOTHED,
    ACTIVE_CONTRIB_SMOOTHED,
]
CONTRIBUTION_COLUMNS_SIMPLE = [
    PORTFOLIO_CONTRIB_SIMPLE,
    BENCHMARK_CONTRIB_SIMPLE,
    ACTIVE_CONTRIB_SIMPLE,
]
CUMULATIVE_ATTRIBUTION_COLUMNS = [
    CUMULATIVE_ALLOCATION_EFFECT,
    CUMULATIVE_SELECTION_EFFECT,
    CUMULATIVE_TOTAL_EFFECT,
]
CUMULATIVE_CONTRIBUTION_COLUMNS = [
    CUMULATIVE_PORTFOLIO_CONTRIB,
    CUMULATIVE_BENCHMARK_CONTRIB,
    CUMULATIVE_ACTIVE_CONTRIB,
]
CUMULATIVE_RETURN_COLUMNS = [
    CUMULATIVE_PORTFOLIO_RETURN,
    CUMULATIVE_BENCHMARK_RETURN,
    CUMULATIVE_ACTIVE_RETURN,
]
DATE_COLUMNS = [BEGINNING_DATE, ENDING_DATE]
FROM_TO_COLUMNS = ["from", "to"]
PERFORMANCE_CLASSIFICATION_COLUMNS = [IDENTIFIER, NAME]
PORTFOLIO_COLUMNS_SMOOTHED = [PORTFOLIO_WEIGHT, PORTFOLIO_RETURN, PORTFOLIO_CONTRIB_SMOOTHED]
PORTFOLIO_COLUMNS_SIMPLE = [PORTFOLIO_WEIGHT, PORTFOLIO_RETURN, PORTFOLIO_CONTRIB_SIMPLE]
RETURN_COLUMNS = [PORTFOLIO_RETURN, BENCHMARK_RETURN, ACTIVE_RETURN]

# Aggregations of column groupings
ALL_CUMULATIVE_COLUMNS = (
    CUMULATIVE_ATTRIBUTION_COLUMNS + CUMULATIVE_CONTRIBUTION_COLUMNS + CUMULATIVE_RETURN_COLUMNS
)
ALL_SIMPLE_COLUMNS = ATTRIBUTION_COLUMNS_SIMPLE + CONTRIBUTION_COLUMNS_SIMPLE
ALL_SMOOTHED_COLUMNS = ATTRIBUTION_COLUMNS_SMOOTHED + CONTRIBUTION_COLUMNS_SMOOTHED

# Special-purpose combinations
PORTFOLIO_BENCHMARK_CONTRIBUTION_COLUMN_PAIRS = (
    (PORTFOLIO_WEIGHT, BENCHMARK_WEIGHT),
    (PORTFOLIO_RETURN, BENCHMARK_RETURN),
    (PORTFOLIO_CONTRIB_SMOOTHED, BENCHMARK_CONTRIB_SMOOTHED),
)

# View column names
VIEW_CUMULATIVE_ATTRIBUTION_COLUMNS = (
    RETURN_COLUMNS
    + CUMULATIVE_RETURN_COLUMNS
    + CONTRIBUTION_COLUMNS_SMOOTHED
    + CUMULATIVE_CONTRIBUTION_COLUMNS
    + ATTRIBUTION_COLUMNS_SMOOTHED
    + CUMULATIVE_ATTRIBUTION_COLUMNS
)
VIEW_OVERALL_ATTRIBUTION_COLUMNS = (
    PORTFOLIO_COLUMNS_SMOOTHED
    + BENCHMARK_COLUMNS_SMOOTHED
    + ACTIVE_COLUMNS_SMOOTHED
    + ATTRIBUTION_COLUMNS_SMOOTHED
)
VIEW_SUBPERIOD_ATTRIBUTION_COLUMNS = (
    PORTFOLIO_COLUMNS_SIMPLE
    + BENCHMARK_COLUMNS_SIMPLE
    + ACTIVE_COLUMNS_SIMPLE
    + ATTRIBUTION_COLUMNS_SIMPLE
)
VIEW_SUBPERIOD_SUMMARY_COLUMNS = (
    RETURN_COLUMNS + CONTRIBUTION_COLUMNS_SIMPLE + ATTRIBUTION_COLUMNS_SIMPLE
)

# Performance column suffixes
AEL = ".ael"  # Allocation Effect smoothed (log-linked)
AES = ".aes"  # Allocation Effect simple
BCL = ".bcl"  # Benchmark Contribution smoothed (log-linked)
BCS = ".bcs"  # Benchmark Contribution simple
CON = ".con"  # Contribution
PCL = ".pcl"  # Portfolio Contribution smoothed (log-linked)
PCS = ".pcs"  # Portfolio Contribution simple
RET = ".ret"  # Return
SEL = ".sel"  # Selection Effect smoothed (log-linked)
SES = ".ses"  # Selection Effect simple
WGT = ".wgt"  # Weight


def col_names(from_col_names: list[str], to_col_name_suffix: str) -> list[str]:
    """
    Gets the column names corresponding to to_col_name_suffix.

    Args:
        from_col_names (list[str]): A list of the column names to translate (with their suffixes).
        to_col_name_suffix (str): The new suffix.

    Returns:
        list[str]: The column names corresponding to to_col_name_suffix.
    """
    return [f"{from_name[:-4]}{to_col_name_suffix}" for from_name in from_col_names]


def short_column_name(full_column_name: str) -> str:
    """
    Remove extraneous technical words from column_name to yield a shorter friendly "engish" name.

    Args:
        column_name (str): The full column name.

    Returns:
        str: The short column name.
    """
    return (
        full_column_name.replace("Cumulative", "")
        .replace("Simple", "")
        .replace("Smoothed", "")
        .replace("_", " ")
        .strip()
    )
