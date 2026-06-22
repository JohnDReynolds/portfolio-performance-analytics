"""Demonstrate security-level performance comparison output."""

# Python imports
from importlib.resources import as_file, files
from pathlib import Path

# Project imports
from ppar.demos.performance_comparison_common import run_performance_comparison_demo


def main() -> None:
    """Run the security performance comparison demonstration."""
    with as_file(files("ppar.demos.data") / "axys") as axys_data_root:
        run_performance_comparison_demo(
            comparison_path=(
                axys_data_root / "ppar_performance_comparison_security_full_spec.yaml"
            ),
            bundle_path=Path.cwd() / "_demo_output" / "performance_comparison_security",
            title="Security Performance Comparison Demo",
        )


if __name__ == "__main__":
    main()
