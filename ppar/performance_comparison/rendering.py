"""Render shared Markdown and HTML fragments for performance comparison reports."""

from __future__ import annotations

# Python imports
from collections.abc import Callable, Iterable, Mapping, Sequence
import datetime as dt
import html as html_lib

# Third-party imports
import polars as pl

RowIdCallback = Callable[[Mapping[str, object], str | None, dict[str, int]], str]


def markdown_table(
    table: pl.DataFrame,
    columns: Sequence[str],
    *,
    empty_message: str = "No rows.",
) -> str:
    """Return a compact Markdown pipe table for selected columns.

    Args:
        table: Source table.
        columns: Columns to show, in display order.
        empty_message: Message to render when no requested columns have rows.

    Returns:
        Markdown table text, or an italic empty-state message.
    """
    if table.is_empty():
        return f"_{escape_markdown_text(empty_message)}_"

    available_columns = [column for column in columns if column in table.columns]
    if not available_columns:
        return f"_{escape_markdown_text(empty_message)}_"

    header = "| " + " | ".join(display_header(column) for column in available_columns) + " |"
    separator = "| " + " | ".join("---" for _ in available_columns) + " |"
    body = [
        "| "
        + " | ".join(format_markdown_cell(row[column]) for column in available_columns)
        + " |"
        for row in table.select(available_columns).iter_rows(named=True)
    ]
    return "\n".join([header, separator, *body])


def html_section(title: str, content: str) -> str:
    """Return one titled HTML report section.

    Args:
        title: Visible section title.
        content: Already-rendered HTML content for the section body.

    Returns:
        HTML section fragment.
    """
    section_id = html_section_id(title)
    return "\n".join(
        [
            f'<section class="pc-section" id="{section_id}">',
            f"<h2>{escape_html(title)}</h2>",
            content,
            "</section>",
        ]
    )


def html_summary_card(label: str, value: object) -> str:
    """Return one compact summary card.

    Args:
        label: Card label.
        value: Card value.

    Returns:
        HTML card fragment.
    """
    return "\n".join(
        [
            '<div class="pc-card">',
            f"<span>{escape_html(label)}</span>",
            f"<strong>{escape_html(format_value(value))}</strong>",
            "</div>",
        ]
    )


def html_paragraph(value: object) -> str:
    """Return one escaped HTML paragraph.

    Args:
        value: Paragraph text.

    Returns:
        HTML paragraph fragment.
    """
    return f"<p>{escape_html(value)}</p>"


def html_table(
    table: pl.DataFrame,
    columns: Sequence[str],
    *,
    empty_message: str = "No rows.",
    row_id_prefix: str | None = None,
    row_id_callback: RowIdCallback | None = None,
) -> str:
    """Return an HTML table for selected columns.

    Args:
        table: Source table.
        columns: Columns to show, in display order.
        empty_message: Message to render when no requested columns have rows.
        row_id_prefix: Optional prefix for generated row identifiers.
        row_id_callback: Optional callback that can produce stable row IDs from
            table rows. The report module supplies portfolio-period specific IDs.

    Returns:
        HTML table fragment, or an empty-state paragraph.
    """
    if table.is_empty():
        return html_empty(empty_message)

    available_columns = [column for column in columns if column in table.columns]
    if not available_columns:
        return html_empty(empty_message)

    header_cells = [
        f'<th scope="col">{escape_html(display_header(column))}</th>'
        for column in available_columns
    ]
    body_rows = []
    row_id_counts: dict[str, int] = {}
    for row in table.select(available_columns).iter_rows(named=True):
        cells = [html_table_cell(row[column], column) for column in available_columns]
        row_id = ""
        if row_id_callback is not None:
            row_id = row_id_callback(row, row_id_prefix, row_id_counts)
        row_id_attribute = f' id="{row_id}"' if row_id else ""
        body_rows.append(f"<tr{row_id_attribute}>" + "".join(cells) + "</tr>")
    return "\n".join(
        [
            '<div class="pc-table-wrap">',
            f'<p class="pc-table-meta">Rows: {escape_html(table.height)}</p>',
            '<table class="pc-table">',
            f"<caption>{html_table_caption(table, available_columns)}</caption>",
            "<thead>",
            "<tr>" + "".join(header_cells) + "</tr>",
            "</thead>",
            "<tbody>",
            *body_rows,
            "</tbody>",
            "</table>",
            "</div>",
        ]
    )


def html_review_key_row_id(section_id: str, review_key: str) -> str:
    """Return a stable HTML row ID for a section/review-key pair.

    Args:
        section_id: Section identifier.
        review_key: Review key from report data.

    Returns:
        HTML-safe row identifier.
    """
    return f"{section_id}--{html_id_token(review_key)}"


def html_id_token(value: str) -> str:
    """Return a conservative HTML ID token.

    Args:
        value: Arbitrary text value.

    Returns:
        Lowercase token containing alphanumeric runs separated by hyphens.
    """
    token = "".join(
        character.lower() if character.isalnum() else "-"
        for character in value
    ).strip("-")
    return token or "row"


def html_table_caption(table: pl.DataFrame, columns: Sequence[str]) -> str:
    """Return an accessible compact caption for an HTML review table.

    Args:
        table: Source table.
        columns: Displayed columns.

    Returns:
        Escaped caption text.
    """
    row_count = format_value(table.height)
    column_count = format_value(len(columns))
    caption = f"Review table with {row_count} row(s) and {column_count} column(s)."
    return escape_html(caption)


def html_table_cell(value: object, column: str) -> str:
    """Return one escaped HTML table cell.

    Args:
        value: Cell value.
        column: Source column name.

    Returns:
        HTML table-cell fragment.
    """
    classes = " ".join(
        [
            html_cell_alignment(value),
            html_column_class(column),
            *html_value_classes(column, value),
        ]
    )
    return f'<td class="{classes}">{escape_html(format_value(value))}</td>'


def html_cell_alignment(value: object) -> str:
    """Return a CSS alignment class for an HTML table value.

    Args:
        value: Cell value.

    Returns:
        Alignment CSS class.
    """
    if isinstance(value, bool):
        return "pc-center"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "pc-right"
    return "pc-left"


def html_column_class(column: str) -> str:
    """Return a stable CSS class for a report table column.

    Args:
        column: Source column name.

    Returns:
        CSS class token for the column.
    """
    normalized = column.replace("_", "-")
    return f"pc-col-{normalized}"


def html_value_classes(column: str, value: object) -> list[str]:
    """Return CSS classes derived from stable report status values.

    Args:
        column: Source column name.
        value: Cell value.

    Returns:
        Additional CSS classes for status styling.
    """
    if column == "review_status":
        return [f"pc-status-{css_token(format_value(value))}"]
    if column == "residual_status" and _is_residual_withheld_status(value):
        return ["pc-status-withheld"]
    return []


def css_token(value: str) -> str:
    """Return a simple CSS token for controlled status strings.

    Args:
        value: Status text.

    Returns:
        Lowercase CSS token.
    """
    return value.replace("_", "-").lower()


def html_empty(message: str) -> str:
    """Return a styled empty-state paragraph.

    Args:
        message: Empty-state text.

    Returns:
        HTML paragraph fragment.
    """
    return f'<p class="pc-empty">{escape_html(message)}</p>'


def html_list(items: Sequence[str]) -> str:
    """Return an escaped unordered HTML list.

    Args:
        items: List item text values.

    Returns:
        HTML unordered-list fragment.
    """
    list_items = [f"<li>{escape_html(item)}</li>" for item in items]
    return "\n".join(["<ul>", *list_items, "</ul>"])


def html_section_id(title: str) -> str:
    """Return a deterministic HTML section ID.

    Args:
        title: Section title.

    Returns:
        Lowercase hyphenated section ID.
    """
    return title.lower().replace(" ", "-")


def display_header(column: str) -> str:
    """Return a report-friendly column label.

    Args:
        column: Source column name.

    Returns:
        Title-cased display label.
    """
    return column.replace("_", " ").title()


def format_markdown_cell(value: object) -> str:
    """Return one escaped Markdown table cell.

    Args:
        value: Cell value.

    Returns:
        Escaped Markdown table cell text.
    """
    return escape_markdown_text(format_value(value))


def format_value(value: object) -> str:
    """Return a compact display value for report cells.

    Args:
        value: Arbitrary report value.

    Returns:
        Human-readable text. Null values are rendered blank.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return str(value)


def comma_separated(values: Sequence[str]) -> str:
    """Return a readable comma-separated list.

    Args:
        values: Display values.

    Returns:
        Comma-separated text.
    """
    return ", ".join(values)


def unique_nonblank_values(values: Iterable[object]) -> list[str]:
    """Return sorted unique display values, omitting blanks and nulls.

    Args:
        values: Values to normalize.

    Returns:
        Sorted unique nonblank display values.
    """
    unique_values = {
        format_value(value)
        for value in values
        if value is not None and format_value(value)
    }
    return sorted(unique_values)


def escape_markdown_text(value: object) -> str:
    """Escape Markdown table delimiters and normalize whitespace.

    Args:
        value: Text value.

    Returns:
        Escaped Markdown text.
    """
    text = " ".join(str(value).split())
    return text.replace("|", "\\|")


def escape_html(value: object) -> str:
    """Escape text for HTML element content.

    Args:
        value: Text value.

    Returns:
        Escaped HTML text.
    """
    text = " ".join(str(value).split())
    return html_lib.escape(text, quote=True)


def html_style_block() -> str:
    """Return CSS for the standalone performance comparison HTML report.

    Returns:
        HTML ``style`` element.
    """
    return """
<style>
:root {
  color-scheme: light;
  --pc-bg: #ecefed;
  --pc-panel: #ffffff;
  --pc-border: #aeb7ba;
  --pc-border-light: #d8dddd;
  --pc-border-strong: #526165;
  --pc-text: #1f2527;
  --pc-muted: #596365;
  --pc-accent: #24596a;
  --pc-table-stripe: #f6f7f6;
  --pc-table-head: #dfe6e7;
  --pc-title-rule: #314247;
  --pc-status-review: #8a3f10;
  --pc-status-monitor: #51610f;
  --pc-status-clear: #24613d;
}
body {
  margin: 0;
  background: var(--pc-bg);
  color: var(--pc-text);
  font-family: Arial, Helvetica, sans-serif;
  font-size: 13px;
  line-height: 1.35;
}
.pc-report {
  max-width: 1360px;
  margin: 0 auto;
  padding: 18px 22px 28px;
}
.pc-header,
.pc-section {
  background: var(--pc-panel);
  border: 1px solid var(--pc-border);
  border-radius: 0;
  box-shadow: 0 1px 2px rgb(0 0 0 / 6%);
  margin: 0 0 12px;
  padding: 12px 14px;
}
.pc-header {
  border-top: 5px solid var(--pc-title-rule);
}
.pc-header h1 {
  border-bottom: 1px solid var(--pc-border-strong);
  font-size: 24px;
  font-weight: 700;
  margin: 0 0 8px;
  padding-bottom: 6px;
}
.pc-section h2 {
  border-bottom: 2px solid var(--pc-border-strong);
  font-size: 17px;
  font-weight: 700;
  margin: 0 0 9px;
  padding-bottom: 4px;
}
.pc-section h3 {
  color: var(--pc-title-rule);
  font-size: 13px;
  font-weight: 700;
  margin: 12px 0 6px;
  text-transform: uppercase;
}
.pc-header p,
.pc-section p {
  margin: 5px 0;
}
.pc-review-basis {
  background: var(--pc-panel);
  border: 1px solid var(--pc-border);
  border-left: 5px solid var(--pc-title-rule);
  box-shadow: 0 1px 2px rgb(0 0 0 / 6%);
  display: grid;
  gap: 0;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  margin: 0 0 12px;
}
.pc-basis-item {
  border-right: 1px solid var(--pc-border-light);
  padding: 7px 10px;
}
.pc-basis-item span {
  color: var(--pc-muted);
  display: block;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
}
.pc-basis-item strong {
  display: block;
  font-size: 13px;
  margin-top: 2px;
}
.pc-section a {
  color: var(--pc-accent);
}
.pc-contents-list {
  column-gap: 28px;
  columns: 2;
  margin: 0;
  padding-left: 18px;
}
.pc-contents-list li {
  break-inside: avoid;
  margin: 0 0 3px;
}
.pc-card-row {
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  margin-bottom: 12px;
}
.pc-card {
  border: 1px solid var(--pc-border);
  border-left: 3px solid var(--pc-border-strong);
  border-radius: 0;
  padding: 7px 9px;
}
.pc-card span {
  color: var(--pc-muted);
  display: block;
  font-size: 12px;
}
.pc-card strong {
  display: block;
  font-size: 20px;
  margin-top: 2px;
}
.pc-triage-row .pc-card {
  border-left-color: var(--pc-accent);
}
.pc-dashboard-summary {
  font-weight: 700;
}
.pc-dashboard-filters {
  align-items: end;
  border: 1px solid var(--pc-border-light);
  display: grid;
  gap: 6px 8px;
  grid-template-columns: minmax(180px, 1fr) minmax(150px, 220px) auto auto;
  margin: 8px 0 10px;
  padding: 8px;
}
.pc-dashboard-filters label {
  color: var(--pc-muted);
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
}
.pc-dashboard-filters input[type="search"],
.pc-dashboard-filters select {
  border: 1px solid var(--pc-border);
  color: var(--pc-text);
  font: inherit;
  min-height: 28px;
  padding: 3px 6px;
}
.pc-dashboard-checkbox {
  align-items: center;
  display: flex;
  gap: 5px;
  min-height: 28px;
}
.pc-dashboard-filters button {
  background: var(--pc-panel);
  border: 1px solid var(--pc-border);
  color: var(--pc-accent);
  font: inherit;
  font-weight: 700;
  min-height: 28px;
  padding: 3px 8px;
}
.pc-dashboard-table-wrap {
  overflow-x: auto;
}
.pc-dashboard-table {
  font-size: 12px;
}
.pc-dashboard-table th,
.pc-dashboard-table td {
  padding: 4px 5px;
}
.pc-dashboard-table th button {
  background: transparent;
  border: 0;
  color: inherit;
  cursor: pointer;
  font: inherit;
  font-weight: 700;
  padding: 0;
  text-align: left;
}
.pc-dashboard-table th button::after {
  color: var(--pc-muted);
  content: " sort";
  font-size: 10px;
  font-weight: 400;
}
.pc-dashboard-row {
  border-left: 5px solid var(--pc-accent);
}
.pc-dashboard-needs-review {
  border-left-color: var(--pc-status-review);
}
.pc-dashboard-monitor {
  border-left-color: var(--pc-status-monitor);
}
.pc-dashboard-clear {
  border-left-color: var(--pc-status-clear);
}
.pc-problem-evidence-link {
  border: 1px solid var(--pc-border);
  display: inline-block;
  font-weight: 700;
  padding: 3px 6px;
  text-decoration: none;
  white-space: nowrap;
}
.pc-dashboard-no-results {
  border: 1px dashed var(--pc-border);
  color: var(--pc-muted);
  padding: 8px;
}
.pc-detail {
  border: 1px solid var(--pc-border-light);
  margin: 7px 0;
}
.pc-detail > summary {
  background: var(--pc-table-head);
  color: var(--pc-title-rule);
  cursor: pointer;
  font-weight: 700;
  padding: 7px 9px;
}
.pc-detail > .pc-section {
  border: 0;
  box-shadow: none;
  margin: 0;
}
.pc-note,
.pc-empty {
  color: var(--pc-muted);
}
.pc-table-wrap {
  overflow-x: auto;
  margin-top: 6px;
}
.pc-table-meta {
  color: var(--pc-muted);
  font-size: 11px;
  font-weight: 700;
  margin: 0 0 3px;
  text-transform: uppercase;
}
table {
  border-collapse: collapse;
  min-width: 100%;
  width: 100%;
}
caption {
  height: 1px;
  overflow: hidden;
  position: absolute;
  white-space: nowrap;
  width: 1px;
}
th,
td {
  border: 1px solid var(--pc-border);
  padding: 4px 6px;
  vertical-align: top;
}
th {
  background: var(--pc-table-head);
  border-bottom: 2px solid var(--pc-border-strong);
  border-top: 1px solid var(--pc-border-strong);
  color: #263033;
  font-size: 11px;
  font-weight: 700;
  text-align: left;
  white-space: nowrap;
}
td {
  border-color: var(--pc-border-light);
}
tbody tr:nth-child(even) {
  background: var(--pc-table-stripe);
}
tbody tr:hover {
  background: #edf2f3;
}
.pc-left {
  text-align: left;
}
.pc-center {
  text-align: center;
}
.pc-right {
  text-align: right;
  white-space: nowrap;
}
.pc-col-portfolio-return-delta,
.pc-col-estimated-return-impact,
.pc-col-estimated-return-impact-total,
.pc-col-transaction-impact-diagnostic-estimate,
.pc-col-delta-b-minus-a,
.pc-col-amount-delta,
.pc-col-quantity-delta,
.pc-col-price-delta {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
}
.pc-col-review-status,
.pc-status-withheld {
  font-weight: 700;
}
.pc-status-needs-review {
  color: var(--pc-status-review);
}
.pc-status-monitor {
  color: var(--pc-status-monitor);
}
.pc-status-clear {
  color: var(--pc-status-clear);
}
#needs-review-summary {
  border-left: 4px solid var(--pc-status-review);
}
#impact-coverage,
#context-evidence-summary,
#residual-status {
  border-left: 4px solid var(--pc-accent);
}
@media (max-width: 760px) {
  .pc-report {
    padding: 12px;
  }
  .pc-contents-list {
    columns: 1;
  }
  .pc-dashboard-filters {
    grid-template-columns: 1fr;
  }
}
@media print {
  body {
    background: #ffffff;
    font-size: 11px;
  }
  .pc-report {
    max-width: none;
    padding: 0;
  }
  .pc-header,
  .pc-review-basis,
  .pc-section {
    border-color: #888888;
    box-shadow: none;
    break-inside: avoid;
    page-break-inside: avoid;
  }
  .pc-section {
    margin-bottom: 10px;
  }
  .pc-table-wrap {
    overflow: visible;
  }
  th,
  td {
    padding: 3px 4px;
  }
  a {
    color: inherit;
    text-decoration: none;
  }
}
</style>""".strip()


def html_dashboard_script() -> str:
    """Return progressive dashboard filtering script.

    Returns:
        HTML ``script`` element.
    """
    return """
<script>
(() => {
  const filters = document.querySelector("[data-dashboard-filters]");
  if (!filters) {
    return;
  }
  const tableBody = document.querySelector(".pc-dashboard-table tbody");
  let rows = Array.from(document.querySelectorAll("[data-dashboard-row]"));
  const search = filters.querySelector("[data-dashboard-search]");
  const status = filters.querySelector("[data-dashboard-status]");
  const missingOnly = filters.querySelector("[data-dashboard-missing-only]");
  const noResults = document.querySelector(".pc-dashboard-no-results");
  let currentSort = {key: "", direction: "asc"};

  const applyFilters = () => {
    const query = (search?.value || "").trim().toLowerCase();
    const selectedStatus = status?.value || "";
    const requireMissing = Boolean(missingOnly?.checked);
    let visibleCount = 0;

    for (const row of rows) {
      const matchesSearch = !query || row.dataset.dashboardSearch.includes(query);
      const matchesStatus = !selectedStatus || row.dataset.reviewStatus === selectedStatus;
      const matchesMissing = !requireMissing || row.dataset.missingInputs === "true";
      const visible = matchesSearch && matchesStatus && matchesMissing;
      row.hidden = !visible;
      if (visible) {
        visibleCount += 1;
      }
    }
    if (noResults) {
      noResults.hidden = visibleCount !== 0;
    }
  };

  filters.addEventListener("input", applyFilters);
  filters.addEventListener("change", applyFilters);
  filters.addEventListener("reset", () => {
    window.setTimeout(applyFilters, 0);
  });
  for (const button of document.querySelectorAll("[data-dashboard-sort]")) {
    button.addEventListener("click", () => {
      const key = button.dataset.dashboardSort || "";
      const sameKey = currentSort.key === key;
      const direction = sameKey && currentSort.direction === "asc" ? "desc" : "asc";
      currentSort = {key, direction};
      rows = [...rows].sort((left, right) => {
        const leftValue = left.dataset[`sort${toDatasetSuffix(key)}`] || "";
        const rightValue = right.dataset[`sort${toDatasetSuffix(key)}`] || "";
        return compareValues(leftValue, rightValue, direction);
      });
      tableBody?.append(...rows);
      applyFilters();
    });
  }
  applyFilters();

  function toDatasetSuffix(key) {
    return key
      .split("-")
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join("");
  }

  function compareValues(leftValue, rightValue, direction) {
    const leftNumber = Number(leftValue);
    const rightNumber = Number(rightValue);
    const bothNumeric = Number.isFinite(leftNumber) && Number.isFinite(rightNumber);
    const comparison = bothNumeric
      ? leftNumber - rightNumber
      : leftValue.localeCompare(rightValue);
    return direction === "asc" ? comparison : -comparison;
  }
})();
</script>
""".strip()


def _is_residual_withheld_status(value: object) -> bool:
    """Return whether a residual status represents a withheld residual."""
    return isinstance(value, str) and value.startswith("withheld")
