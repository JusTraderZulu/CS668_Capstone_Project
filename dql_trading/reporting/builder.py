import os
from typing import Dict, Any, Optional, List

class TradingReportBuilder:
    def __init__(self, experiment_name: str, output_dir: str = "results") -> None:
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.pdf = TradingReportPDF(experiment_name)
        self.report_path = os.path.join(output_dir, f"{experiment_name}_report.pdf")

        # Track where each major section begins so we can build a dynamic TOC
        self._section_pages: Dict[str, int] = {}
        self._toc_page_no: int | None = None

    def add_title_page(self, title: str, subtitle: str | None = None, date: str | None = None):
        if date:
            self.pdf.set_font("Arial", "", 12)
            self.pdf.set_text_color(0, 0, 0)
            self.pdf.ln(10)
            self.pdf.cell(0, 10, f"Generated on: {date}", 0, 1, "C")

        # Immediately add a blank page that will later contain the TOC.  We remember its page number.
        self.pdf.add_page()
        self._toc_page_no = self.pdf.page_no()
        # No content yet; we'll fill it inside generate().

    def add_executive_summary(self, train_summary: Dict[str, Any], test_summary: Optional[Dict[str, Any]] = None):
        self.pdf.add_page()
        self._section_pages["Executive Summary"] = self.pdf.page_no()

    def add_training_metrics(self, metrics: Dict[str, Any], training_plots_path: str, learning_curves_path: str):
        self.pdf.add_page()
        self._section_pages["Training Performance Analysis"] = self.pdf.page_no()

    def add_testing_metrics(self, metrics: Dict[str, Any], performance_dashboard_path: str, trade_visualization_path: str):
        self.pdf.add_page()
        self._section_pages["Out-of-Sample Testing Results"] = self.pdf.page_no()

    def add_baseline_comparison(self, dql_metrics: Dict[str, Any], baseline_metrics: Dict[str, Dict[str, Any]], comparison_chart_path: str):
        self.pdf.add_page()
        self._section_pages["Comparison with Baseline Strategies"] = self.pdf.page_no()

    def add_hyperparameter_analysis(self, parameters: Dict[str, Dict[str, Any]]):
        self.pdf.add_page()
        self._section_pages["Model Configuration and Hyperparameters"] = self.pdf.page_no()

    def add_hyperparameter_tuning(self, best_params: dict, results_path: str | None = None):
        """Insert a brief hyper-parameter tuning summary.

        Parameters
        ----------
        best_params : dict
            Dictionary of the best parameters found.
        results_path : str | None
            Optional path to the full CSV of tuning results – will be referenced
            in the PDF so the reader can locate the data.
        """
        self.pdf.add_page()
        self._section_pages["Hyperparameter Tuning Summary"] = self.pdf.page_no()

        self.pdf.chapter_title("Hyperparameter Tuning Summary")

        # Best parameters table
        self.pdf.set_font("Arial", size=10)
        for k, v in best_params.items():
            self.pdf.cell(60, 6, str(k), 0, 0)
            self.pdf.cell(0, 6, str(v), 0, 1)

        if results_path:
            self.pdf.ln(4)
            self.pdf.set_font("Arial", "I", 9)
            self.pdf.multi_cell(0, 5, f"Full tuning results CSV: {results_path}")

    def add_feature_importance(self, feature_importance_data: Optional[Dict[str, float]] = None,
                               feature_importance_path: Optional[str] = None):
        self.pdf.add_page()
        self._section_pages["Feature Importance Analysis"] = self.pdf.page_no()

    def add_conclusion(self, key_findings: List[str], recommendations: List[str]):
        self.pdf.add_page()
        self._section_pages["Conclusion and Recommendations"] = self.pdf.page_no()

    def generate(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Fill in the previously blank TOC page with dynamic page numbers
        # ------------------------------------------------------------------
        if self._toc_page_no is not None:
            current_page = self.pdf.page_no()
            self.pdf.page = self._toc_page_no
            self.pdf.chapter_title("Table of Contents")

            # Use the recorded starting pages; sort by the order we expect
            ordered_keys = [
                "Executive Summary",
                "Training Performance Analysis",
                "Out-of-Sample Testing Results",
                "Comparison with Baseline Strategies",
                "Hyperparameter Tuning Summary",
                "Model Configuration and Hyperparameters",
                "Feature Importance Analysis",
                "Conclusion and Recommendations",
            ]
            for key in ordered_keys:
                page_no = self._section_pages.get(key, "-")
                self.pdf.cell(0, 10, key, 0, 0)
                self.pdf.cell(0, 10, str(page_no), 0, 1, "R")

            # Return to the last page so further operations are safe
            self.pdf.page = current_page

        self.pdf.output(self.report_path)
        return self.report_path 