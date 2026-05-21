from pathlib import Path
import re
import json

from .html.html_overview import create_html


def _extract_plot_data(html_content: str) -> dict | None:
    """Extract plotData.six and plotData.binary from a deprel index HTML page."""
    match = re.search(
        r'six:\s*(\[.+?\]),\s*\n\s*binary:\s*(\[.+?\])',
        html_content,
    )
    if not match:
        return None
    return {
        "six": json.loads(match.group(1)),
        "binary": json.loads(match.group(2)),
    }


def _safe_id(deprel: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', deprel)


def generate_html_overview_index(html_directory: str) -> None:
    """Generate a top-level index page with scatter plot thumbnails per dependency relation.

    Globs over all {deprel}/index.html pages inside html_directory, scrapes their
    embedded scatter-plot data, and produces an index.html at the root of that
    directory with a 3-column panel grid.

    Args:
        html_directory: Directory containing per-deprel subdirectories with index.html files.
    """
    html_directory = Path(html_directory)

    deprels: dict[str, dict] = {}
    for index_file in sorted(html_directory.glob("*/index.html")):
        deprel = index_file.parent.name
        data = _extract_plot_data(index_file.read_text(encoding="utf-8"))
        if data:
            deprels[deprel] = data

    if not deprels:
        raise ValueError(f"No deprel index pages with plot data found in {html_directory}")

    panels_html = "\n".join(
        f'            <div class="panel">\n'
        f'                <div id="plot-{_safe_id(deprel)}" class="mini-plot"'
        f' data-url="/multiblimp/{deprel}" data-deprel="{deprel}"></div>\n'
        f'                <div class="panel-label"><a href="/multiblimp/{deprel}">{deprel}</a></div>\n'
        f'            </div>'
        for deprel in deprels
    )

    all_data_json = json.dumps(deprels)

    html_content = create_html(panels_html, all_data_json)

    output_path = html_directory / "index.html"
    output_path.write_text(html_content, encoding="utf-8")
