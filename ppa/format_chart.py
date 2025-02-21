"""
This module contains static methods for formatting the charts enumerated in Attribution.Chart.
"""

## Override for pylint and pylance.  All of the plt methods are "type partially unknown".
# pyright: reportUnknownMemberType=none

# Python Imports
import io
import math
import textwrap

# Third-Party Imports
from matplotlib import ticker
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# Project Imports
import ppa.columns as cols
import ppa.utilities as util

# Reasonable chart sizing constraints, just so they don't get too tiny or huge.
_DEFAULT_FIGSIZE = (16, 7)  # width, height
_MAXIMUM_FIGURE_HEIGHT = 3 * _DEFAULT_FIGSIZE[1]  # in inches
_MAXIMUM_FIGURE_WIDTH = 3 * _DEFAULT_FIGSIZE[0]  # in inches
_MAXIMUM_LABEL_LENGTH = 45  # characters
_MINIMUM_FIGURE_HEIGHT = 0.9 * _DEFAULT_FIGSIZE[1]  # in inches
_MINIMUM_FIGURE_WIDTH = 0.9 * _DEFAULT_FIGSIZE[0]  # in inches

# Chart colors
# 0 = portfolio, 1 = benchmark, 2 = active
# 0 = allocation effect, 1 = selection effect, 2 = total effect
_COLORS = ("green", "blue", "orange")

# The _TOP_MARGIN will allow room for the suptitle at the top.
_TOP_MARGIN = 0.99


def cumulative_lines(
    df: pl.DataFrame,
    column_names: list[str],
    title_lines: tuple[str, str],
    y_axis_label: str,
) -> bytes:
    """
    Formats the CUMULATIVE charts.

    Args:
        df (pl.DataFrame): The View.CUMULATIVE_ATTRIBUTION DataFrame.
        column_names (list[str]): A list of the 3 cumulative column names to chart.
        title_lines (tuple[str, str]): A tuple of the title and subtitle lines.
        y_axis_label (str): The label for the y-axis.

    Returns:
        bytes: An in-memory png of the matplotlib chart.
    """
    # Create figure
    fig = plt.figure(figsize=_figsize(_DEFAULT_FIGSIZE))

    # Add axes at position (6% from left, 13% from bottom, 92% wide, 79% tall)
    ax = fig.add_axes((0.06, 0.13, 0.92, 0.79))

    # Set the title lines
    plt.suptitle(f"{title_lines[0]}\n{title_lines[1]}")

    # Set the dates
    dates = df[cols.ENDING_DATE]

    # Plot the lines
    for idx, column_name in enumerate(column_names):
        ax.plot(
            dates,
            df[column_name],
            label=cols.short_column_name(column_name),
            color=_COLORS[idx],
        )

    # Set the y-axis labels.
    ax.set_ylabel(y_axis_label)

    # Add horizontal grid line at y == 0
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)

    # Set x-axis labels.
    # This will set the maximum qty of x-ticks to 23.  Once it hits 24, then it will only show a
    # label for every-other date, at 36 every third date, etc.
    use_dates = dates if len(dates) <= 12 else dates[:: len(dates) // 12]
    ax.set_xticks(use_dates)

    # Rotate x-axis labels for better readability
    ax.set_xticklabels(use_dates, rotation=45, ha="right")

    # Show the legend.
    ax.legend()

    # Return the png.
    return _to_png(fig)


def _figsize(figsize: tuple[float, float]) -> tuple[float, float]:
    """
    Constrain the figsize by minimum and maximum sizes.

    Args:
        figsize (tuple[float, float]): The desired figsize (width, height).

    Returns:
        tuple[float, float]: The constrained figsize.
    """
    return (
        max(min(figsize[0], _MAXIMUM_FIGURE_WIDTH), _MINIMUM_FIGURE_WIDTH),
        max(min(figsize[1], _MAXIMUM_FIGURE_HEIGHT), _MINIMUM_FIGURE_HEIGHT),
    )


def heatmap(
    df: pl.DataFrame,
    column_name: str,
    title_lines: tuple[str, str],
) -> bytes:
    """
    Formats the HEATMAP charts.

    Args:
        df (pl.DataFrame): The View.SUBPERIOD_SUMMARY DataFrame.
        column_name (str): The column name.
        title_lines (tuple[str, str]): A tuple of the title and subtitle lines.

    Returns:
        bytes: An in-memory png of the matplotlib chart.
    """
    # Convert the date column to a string label with the format "yyyy-mm-dd"
    df = df.with_columns(
        pl.col(cols.ENDING_DATE).dt.strftime(util.DATE_FORMAT_STRING).alias("date_label")
    )

    # Word-wrap the classification labels
    df = df.with_columns(
        pl.Series("classification_label", _word_wrap(df[cols.CLASSIFICATION_NAME]))
    )

    # Set the figure width and height
    fig_width = len(set(df["date_label"])) * 0.7
    fig_height = len(set(df["classification_label"])) * 0.6

    # Create the figure
    fig = plt.figure(figsize=_figsize((fig_width, fig_height)))

    # Set the overall figure title.
    plt.suptitle(f"{title_lines[0]}\n{title_lines[1]}")

    # If it is a "portfolio-only" heatmap, then get rid of the cells where the portfolio weight is
    # zero.  They are there because the benchmark weight is not 0.0.
    if column_name in (cols.PORTFOLIO_CONTRIB_SIMPLE, cols.PORTFOLIO_RETURN):
        df = df.filter(pl.col(cols.PORTFOLIO_WEIGHT) != 0)

    # Select just the needed columns.
    df = df[["date_label", "classification_label", column_name]]

    # Pivot the data for the heatmap.  Cannot get it to work for polars, so convert to pandas.
    heatmap_data = df.to_pandas().pivot(
        index="classification_label",
        columns="date_label",
        values=column_name,
    )

    # Create the cmap: 0 = green, 120 = red, 100=saturation, 50=lightness
    cmap = sns.diverging_palette(0, 120, s=100, l=50, as_cmap=True)

    # Create the heatmap.
    ax = sns.heatmap(
        heatmap_data, cmap=cmap, center=0, annot=True, fmt=".4f", linewidths=0.5, cbar=False
    )

    # Remove the axes labels.
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # tight_layout() is a "best pratice" that does some automatic spacing between subplots
    # rect=[left, bottom, right, top], where (0, 0, 1, 1) means the entire figure.
    # The bottom_margin will allow room for the rotated dates at the bottom.
    bottom_margin = 0.02
    fig.tight_layout(rect=(0, bottom_margin, 1, _TOP_MARGIN))

    # Return the png.
    return _to_png(fig)


def overall_attribution(
    df: pl.DataFrame,
    title_lines: tuple[str, str],
) -> bytes:
    """
    Formats the chart: OVERALL_ATTRIBUTION

    Args:
        df (pl.DataFrame): The View.OVERALL_ATTRIBUTION DataFrame.
        title_lines (tuple[str, str]): A tuple of the title and subtitle lines.

    Returns:
        bytes: An in-memory png of the matplotlib chart.
    """
    # Sort the dataframe by TOTAL_EFFECT_SMOOTHED, descending.
    df = df.sort(cols.TOTAL_EFFECT_SMOOTHED, descending=True)

    # Set the labels, data series names and data series values.
    labels = _word_wrap(df[cols.CLASSIFICATION_NAME])
    series_names = [cols.short_column_name(col) for col in cols.ATTRIBUTION_COLUMNS_SMOOTHED]
    series_values = [df[col] for col in cols.ATTRIBUTION_COLUMNS_SMOOTHED]

    # Concatenate all series into a single Polars DataFrame column and find min/max.
    combined_series = pl.concat(series_values)
    overall_min = math.floor(combined_series.min() * 100) / 100  # type: ignore
    overall_max = math.ceil(combined_series.max() * 100) / 100  # type: ignore

    # Get the vertical chart measurements.
    bar_height, _, _, fig_height = _vertical_chart_measurements(len(labels))

    # _vertical_chart_measurements gives a bar_height for double bars, so make the bar_height
    # larger since this chart only has single bars.
    bar_height = bar_height * 1.75

    # Create the overall figure with 3 subplots and a shared y axis.
    fig, axes = plt.subplots(
        1, 3, figsize=_figsize((_DEFAULT_FIGSIZE[0], fig_height)), sharey=True
    )

    # Ensure unique y positions in case the labels are not unique.
    y_positions = range(len(labels))

    for ax, series_values, title in zip(axes, series_values, series_names):
        # Create the subplot.
        ax.set_title(title)
        colors = ["green" if val >= 0 else "red" for val in series_values]
        ax.barh(labels, series_values, height=bar_height, color=colors)

        # Set the y-axis.
        ax.set_yticks(y_positions)  # Must be done because labels might not be unique
        ax.invert_yaxis()
        ax.set_yticklabels(labels)

        # Set the x-axis ticks min/max, and format them to 2 decimals.
        ax.set_xticks(np.linspace(overall_min, overall_max, num=7))  # type: ignore
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        # Add vertical grid line at x == 0
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)

    # Set the titles
    plt.suptitle(f"{title_lines[0]}\n{title_lines[1]}")

    # Automatically adjust the spacing between subplots.
    fig.tight_layout(rect=(0, 0, 1, _TOP_MARGIN))

    # Return the png.
    return _to_png(fig)


def overall_contribution(
    df: pl.DataFrame,
    title_lines: tuple[str, str],
    portfolio_name: str,
    benchmark_name: str,
) -> bytes:
    """
    Formats the chart: OVERALL_CONTRIBUTION

    Args:
        df (pl.DataFrame): The View.OVERALL_ATTRIBUTION DataFrame.
        title_lines (tuple[str, str]): A tuple of the title and subtitle lines.
        portfolio_name (str): The portfolio name.
        benchmark_name (str): The benchmark name.

    Returns:
        bytes: An in-memory png of the matplotlib chart.
    """
    # Sort the dataframe by PORTFOLIO_CONTRIB_SMOOTHED, descending.
    df = df.sort(cols.PORTFOLIO_CONTRIB_SMOOTHED, descending=True)

    # Get the series names.
    series_names = ("Weight", "Return", "Contribution")

    # Get the series values in 3 groups of 2:
    #   0 = Weight, 1 = Return, 2 = Contribution
    #     0 = Portfolio, 1 = Benchmark
    series_values = [
        ((df[col[0]], df[col[1]])) for col in cols.PORTFOLIO_BENCHMARK_CONTRIBUTION_COLUMN_PAIRS
    ]

    # Get the labels
    labels = _word_wrap(df[cols.CLASSIFICATION_NAME])

    # Get the vertival chart measurements.
    bar_height, bottom_margin, delta, fig_height = _vertical_chart_measurements(len(labels))

    # Create the overall figure with 3 subplots and a shared y axis.
    fig, axes = plt.subplots(
        1, 3, figsize=_figsize((_DEFAULT_FIGSIZE[0], fig_height)), sharey=True
    )

    # Set the overall figure title.
    plt.suptitle(f"{title_lines[0]}\n{title_lines[1]}")

    # Loop through to create the 3 sub-plots.
    for ax, values, name in zip(axes, series_values, series_names):
        # Set the sub-plot title to be the series name.
        ax.set_title(name)

        # Set the y-axis ticks to be at the group centers.
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)

        # Invert the y-axis so the first group will be at the top.
        ax.invert_yaxis()

        # Set x-axis ticks to 2 decimals.
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        # Add vertical grid line at x == 0
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)

        # Plot the bars.
        for i in range(len(labels)):
            # Plot the green portfolio bar at y = i - delta
            ax.barh(i - delta, values[0][i], height=bar_height, color="green")
            # Plot the blue benchmark bar at y = i + delta
            ax.barh(i + delta, values[1][i], height=bar_height, color="blue")

    # tight_layout() is a "best pratice" that does some automatic spacing between subplots
    # rect=[left, bottom, right, top], where (0, 0, 1, 1) means the entire figure.
    # The bottom_margin will allow room for the legend at the bottom.
    fig.tight_layout(rect=(0, bottom_margin, 1, _TOP_MARGIN))

    # Create a legend for the portfolio and benchmark.
    portfolio_patch = mpatches.Patch(color="green", label=portfolio_name)
    benchmark_patch = mpatches.Patch(color="blue", label=benchmark_name)
    fig.legend(
        handles=[portfolio_patch, benchmark_patch],
        loc="lower center",
        ncol=2,  # This makes them horizontal instead of vertical.
        fontsize=12,
    )

    # Return the png.
    return _to_png(fig)


def _to_png(figure: Figure) -> bytes:
    """
    Convert figure to png bytes.

    Args:
        figure (Figure): The figure to convert.

    Returns:
        bytes: The resulting png bytes.
    """
    buf = io.BytesIO()
    figure.savefig(buf, format="png")
    return buf.getvalue()


def vertical_bars(
    df: pl.DataFrame,
    column_names: list[str],
    title_lines: tuple[str, str],
    y_axis_label: str,
) -> bytes:
    """
    Formats the charts with vertical bars.

    Args:
        df (pl.DataFrame): The View.SUBPERIOD_SUMMARY DataFrame.
        column_names (list[str]): A list of the 3 cumulative column names to chart.
        title_lines (tuple[str, str]): A tuple of the title and subtitle lines.
        y_axis_label (str): The label for the y-axis.

    Returns:
        bytes: An in-memory png of the matplotlib chart.
    """
    # Set the dates
    dates = df[cols.ENDING_DATE]

    # Define the bar width
    bar_width = 0.2
    indices = np.arange(len(dates))

    # Adjust the figure width based on the quantity of dates
    fig_width = len(dates) * bar_width * len(column_names)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=_figsize((fig_width, _DEFAULT_FIGSIZE[1])))

    # Set the overall figure title.
    plt.suptitle(f"{title_lines[0]}\n{title_lines[1]}")

    # Plot the bars
    for idx, column_name in enumerate(column_names):
        location = indices
        if idx == 0:
            location = indices - bar_width
        elif idx == 1:
            location = indices
        elif idx == 2:
            location = indices + bar_width
        ax.bar(location, df[column_name], width=bar_width, color=_COLORS[idx])

    # Set x-axis labels and formatting.
    # ha="right" will align the rotated dates so they are positioned at the x-axis tick.
    ax.set_xticks(indices)
    ax.set_xticklabels(dates, rotation=45, ha="right")

    # Set y-axis labels to 2 decimals, and add a horizontal grid line at y == 0
    ax.set_ylabel(y_axis_label)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)

    # tight_layout() is a "best pratice" that does some automatic spacing between subplots
    # rect=[left, bottom, right, top], where (0, 0, 1, 1) means the entire figure.
    # The bottom_margin will allow room for the legend at the bottom.
    bottom_margin = 0.06
    fig.tight_layout(rect=(0, bottom_margin, 1, 1))

    # Create a legend for the portfolio and benchmark.
    patches = [
        mpatches.Patch(color=_COLORS[idx], label=cols.short_column_name(column_name))
        for idx, column_name in enumerate(column_names)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches), fontsize=12)

    # Return the png.
    return _to_png(fig)


def _vertical_chart_measurements(qty_of_y_ticks: int) -> tuple[float, float, float, float]:
    """
    Get the vertical chart measurements.

    Args:
        qty_of_y_ticks (int): The quantity of y-axis ticks.

    Returns:
        tuple[float, float, float]: (bar_height, bottom_margin, delta, fig_height)
    """
    # Set the overall figure height (in inches) based on qty_of_y_ticks.
    # The height_factor of 0.4 seems to work the best.  If you decrease it, then the labels
    # can start overlapping, especially when they are word-wrapped to 3 lines.  If you increase it,
    # then you have too much extra space.
    height_factor = 0.4
    fig_height = max(
        _MINIMUM_FIGURE_HEIGHT, min(qty_of_y_ticks * height_factor, _MAXIMUM_FIGURE_HEIGHT)
    )

    # The bottom_margin will allow for a legend.
    bottom_margin = 0.07 * _MINIMUM_FIGURE_HEIGHT / fig_height

    # Use a grouped bar approach.  Let the “group index” be on the y-axis.  For each group i,
    # draw two bars:
    #   Portfolio bar at y = i - delta
    #   Benchmark bar at y = i + delta
    # The tick label for group i will be at y = i.
    bar_height = 0.3
    delta = (bar_height / 2) + 0.01  # vertical offset of each bar from the group center

    return bar_height, bottom_margin, delta, fig_height


def _word_wrap(phrases: pl.Series) -> list[str]:
    """
    Word-wrap each phrase at a maximum_length.

    Args:
        phrases (pl.Series): A series of phrases, each phrase containing multiple words.

    Returns:
        list[str]: The word-wrapped list of phrases.
    """
    return [textwrap.fill(phrase, width=_MAXIMUM_LABEL_LENGTH) for phrase in phrases]
