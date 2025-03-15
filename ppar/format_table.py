"""
This module contains static methods for formatting "great_tables" for the views enumerated in
Attribution.View.
"""

# Third-Party Imports
import great_tables as gt

# Project Imports
import ppar.columns as cols
import ppar.utilities as util

# Constants
_DISPLAY_DECIMALS = 4

# html labels
_label_active = gt.html("Active")
_label_allocation = gt.html("Allocation")
_label_benchmark = gt.html("Benchmark")
_label_contrib = gt.html("Contrib")
_label_id = gt.html("ID")
_label_name = gt.html("Name")
_label_portfolio = gt.html("Portfolio")
_label_return = gt.html("Return")
_label_selection = gt.html("Selection")
_label_total = gt.html("Total")
_label_weight = gt.html("Weight")


def _align_dates(table: gt.GT) -> gt.GT:
    """Center-Align the dates."""
    return table.cols_label(
        beginning_date=gt.html('<p align="center">Beginning</p>'),
        ending_date=gt.html('<p align="center">Ending</p>'),
    )


def cumulative_attribution(table: gt.GT) -> gt.GT:
    """
    Formats a "great_table" version of View.CUMULATIVE_ATTRIBUTION.

    Args:
        table (gt.GT): The generic template table.

    Returns:
        table (gt.GT): The formatted "great_table" version of View.CUMULATIVE_ATTRIBUTION.
    """
    # This table is wide, so apply a narrow font style.
    table = table.tab_options(table_font_names="Arial Narrow")

    # Format the table.
    table = (
        table.fmt_number(
            columns=cols.VIEW_CUMULATIVE_ATTRIBUTION_COLUMNS,  # type: ignore
            decimals=_DISPLAY_DECIMALS,
        )
        .tab_spanner(
            label="Returns",
            columns=cols.RETURN_COLUMNS,  # type: ignore
        )
        .tab_spanner(
            label="Cumulative Returns",
            columns=cols.CUMULATIVE_RETURN_COLUMNS,  # type: ignore
        )
        .tab_spanner(
            label="Contributions",
            columns=cols.CONTRIBUTION_COLUMNS_SMOOTHED,  # type: ignore
        )
        .tab_spanner(
            label="Cumulative Contributions",
            columns=cols.CUMULATIVE_CONTRIBUTION_COLUMNS,  # type: ignore
        )
        .tab_spanner(
            label="Attribution Effects",
            columns=cols.ATTRIBUTION_COLUMNS_SMOOTHED,  # type: ignore
        )
        .tab_spanner(
            label="Cumulative Attribution Effects",
            columns=cols.CUMULATIVE_ATTRIBUTION_COLUMNS,  # type: ignore
        )
        # Change the column labels now that the spanners are in place.
        .cols_label(
            Portfolio_Return=_label_portfolio,
            Benchmark_Return=_label_benchmark,
            Active_Return=_label_active,
            Cumulative_Portfolio_Return=_label_portfolio,
            Cumulative_Benchmark_Return=_label_benchmark,
            Cumulative_Active_Return=_label_active,
            Portfolio_Contribution_Smoothed=_label_portfolio,
            Benchmark_Contribution_Smoothed=_label_benchmark,
            Active_Contribution_Smoothed=_label_active,
            Cumulative_Portfolio_Contribution=_label_portfolio,
            Cumulative_Benchmark_Contribution=_label_benchmark,
            Cumulative_Active_Contribution=_label_active,
            Allocation_Effect_Smoothed=_label_allocation,
            Selection_Effect_Smoothed=_label_selection,
            Total_Effect_Smoothed=_label_total,
            Cumulative_Allocation_Effect=_label_allocation,
            Cumulative_Selection_Effect=_label_selection,
            Cumulative_Total_Effect=_label_total,
        )
        # Standard style theme
        .opt_stylize(style=1)
    )

    # Align the dates.
    table = _align_dates(table)

    # Return the table.
    return table


def _display_classification_label(classification_label: str) -> str:
    """Return the classification_label for displaying."""
    return "" if util.is_empty(classification_label) else classification_label


def overall_attribution(table: gt.GT, classification_label: str) -> gt.GT:
    """
    Formats a "great_table" version of View.OVERALL_ATTRIBUTION.

    Args:
        table (gt.GT): The generic template table.
        classification_label (str): The classification label.

    Returns:
        table (gt.GT): The formatted "great_table" version of View.OVERALL_ATTRIBUTION
    """
    # Format the table.
    table = (
        table.fmt_number(
            columns=cols.VIEW_OVERALL_ATTRIBUTION_COLUMNS,  # type: ignore
            decimals=_DISPLAY_DECIMALS,
        )
        .tab_spanner(
            label=_display_classification_label(classification_label),
            columns=cols.CLASSIFICATION_COLUMNS,  # type: ignore
        )
        .tab_spanner(
            label="Portfolio",
            columns=cols.PORTFOLIO_COLUMNS_SMOOTHED,  # type: ignore
        )
        .tab_spanner(
            label="Benchmark",
            columns=cols.BENCHMARK_COLUMNS_SMOOTHED,  # type: ignore
        )
        .tab_spanner(
            label="Active",
            columns=cols.ACTIVE_COLUMNS_SMOOTHED,  # type: ignore
        )
        .tab_spanner(
            label="Attribution",
            columns=cols.ATTRIBUTION_COLUMNS_SMOOTHED,  # type: ignore
        )
        # Change the column labels now that the spanners are in place.
        .cols_label(
            Classification_Identifier=_label_id,
            Classification_Name=_label_name,
            Portfolio_Weight=_label_weight,
            Portfolio_Return=_label_return,
            Portfolio_Contribution_Smoothed=_label_contrib,
            Benchmark_Weight=_label_weight,
            Benchmark_Return=_label_return,
            Benchmark_Contribution_Smoothed=_label_contrib,
            Active_Weight=_label_weight,
            Active_Return=_label_return,
            Active_Contribution_Smoothed=_label_contrib,
            Allocation_Effect_Smoothed=_label_allocation,
            Selection_Effect_Smoothed=_label_selection,
            Total_Effect_Smoothed=_label_total,
        )
        # Standard style theme
        .opt_stylize(style=1)
    )

    # Return the table.
    return table


def subperiod_attribution(table: gt.GT, classification_label: str) -> gt.GT:
    """
    Formats a "great_table" version of View.SUBPERIOD_ATTRIBUTION.

    Args:
        table (gt.GT): The generic template table.
        classification_label (str): The classification label.

    Returns:
        table (gt.GT): The formatted "great_table" version of View.SUBPERIOD_ATTRIBUTION
    """
    # Format the table.
    table = (
        table.fmt_number(
            columns=cols.VIEW_SUBPERIOD_ATTRIBUTION_COLUMNS,  # type: ignore
            decimals=_DISPLAY_DECIMALS,
        )
        .tab_spanner(
            label=_display_classification_label(classification_label),
            columns=cols.CLASSIFICATION_COLUMNS,  # type: ignore
        )
        .tab_spanner(
            label="Portfolio",
            columns=cols.PORTFOLIO_COLUMNS_SIMPLE,  # type: ignore
        )
        .tab_spanner(
            label="Benchmark",
            columns=cols.BENCHMARK_COLUMNS_SIMPLE,  # type: ignore
        )
        .tab_spanner(
            label="Active",
            columns=cols.ACTIVE_COLUMNS_SIMPLE,  # type: ignore
        )
        .tab_spanner(
            label="Attribution",
            columns=cols.ATTRIBUTION_COLUMNS_SIMPLE,  # type: ignore
        )
        # Change the column labels now that the spanners are in place.
        .cols_label(
            Classification_Identifier=_label_id,
            Classification_Name=_label_name,
            Portfolio_Weight=_label_weight,
            Portfolio_Return=_label_return,
            Portfolio_Contribution_Simple=_label_contrib,
            Benchmark_Weight=_label_weight,
            Benchmark_Return=_label_return,
            Benchmark_Contribution_Simple=_label_contrib,
            Active_Weight=_label_weight,
            Active_Return=_label_return,
            Active_Contribution_Simple=_label_contrib,
            Allocation_Effect_Simple=_label_allocation,
            Selection_Effect_Simple=_label_selection,
            Total_Effect_Simple=_label_total,
        )
        # Standard style theme
        .opt_stylize(style=1)
    )

    # Align the dates.
    table = _align_dates(table)

    # Return the table.
    return table


def subperiod_summary(table: gt.GT) -> gt.GT:
    """
    Formats a "great_table" version of View.SUBPERIOD_SUMMARY.

    Args:
        table (gt.GT): The generic template table.

    Returns:
        table (gt.GT): The formatted "great_table" version of View.SUBPERIOD_SUMMARY
    """
    # Format the table.
    table = (
        table.fmt_number(
            columns=cols.VIEW_SUBPERIOD_SUMMARY_COLUMNS,  # type: ignore
            decimals=_DISPLAY_DECIMALS,
        )
        .tab_spanner(
            label="Returns",
            columns=cols.RETURN_COLUMNS,  # type: ignore
        )
        .tab_spanner(
            label="Contributions",
            columns=cols.CONTRIBUTION_COLUMNS_SIMPLE,  # type: ignore
        )
        .tab_spanner(
            label="Attribution Effects",
            columns=cols.ATTRIBUTION_COLUMNS_SIMPLE,  # type: ignore
        )
        # Change the column labels now that the spanners are in place.
        .cols_label(
            Portfolio_Return=_label_portfolio,
            Benchmark_Return=_label_benchmark,
            Active_Return=_label_active,
            Portfolio_Contribution_Simple=_label_portfolio,
            Benchmark_Contribution_Simple=_label_benchmark,
            Active_Contribution_Simple=_label_active,
            Allocation_Effect_Simple=_label_allocation,
            Selection_Effect_Simple=_label_selection,
            Total_Effect_Simple=_label_total,
        )
        # Standard style theme
        .opt_stylize(style=1)
    )

    # Align the dates.
    table = _align_dates(table)

    # Return the table.
    return table
