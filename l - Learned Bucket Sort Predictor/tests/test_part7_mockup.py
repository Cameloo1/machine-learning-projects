from html.parser import HTMLParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOCKUP_PATH = PROJECT_ROOT / "mockup" / "index.html"


class ImageSourceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.image_sources = []
        self.section_ids = set()

    def handle_starttag(self, tag, attrs):
        attr_map = dict(attrs)
        if tag == "img" and "src" in attr_map:
            self.image_sources.append(attr_map["src"])
        if tag == "section" and "id" in attr_map:
            self.section_ids.add(attr_map["id"])


def test_mockup_exists_and_includes_current_project_methods():
    html = MOCKUP_PATH.read_text(encoding="utf-8")

    assert MOCKUP_PATH.exists()
    assert "analytic_baseline" in html
    assert "linear_cdf" in html
    assert "mlp_cdf" in html
    assert "optional Torch" in html
    assert "Speed win is not proven yet" in html


def test_mockup_declares_browser_demo_boundary():
    html = MOCKUP_PATH.read_text(encoding="utf-8")

    assert "Browser charts are illustrative" in html
    assert "They do not run the Python models" in html
    assert "Real project claims come from the benchmark artifacts and plots below" in html
    assert "Browser demo uses empirical CDF approximation" in html


def test_mockup_references_promoted_part5_evidence_assets_only():
    parser = ImageSourceParser()
    parser.feed(MOCKUP_PATH.read_text(encoding="utf-8"))

    assert {"methods", "evidence"} <= parser.section_ids
    assert parser.image_sources == [
        "../assets/part5-scale-closure-total-ms.png",
        "../assets/part5-bucket-quality.png",
        "../assets/part5-amortized-runtime-breakdown.png",
    ]
    for source in parser.image_sources:
        assert source.startswith("../assets/")
        assert (MOCKUP_PATH.parent / source).resolve().exists()


def test_mockup_has_no_stale_or_local_only_claims():
    html = MOCKUP_PATH.read_text(encoding="utf-8")

    forbidden_fragments = [
        "MLPRegressor",
        "production uses scikit-learn CDF regressors",
        "artifacts/evidence",
        "datasets/generated",
        "benchmark_methods_",
        "scale_closure_",
        "amortized_benchmark_",
        "C:\\",
        "Users\\",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in html


def test_mockup_keeps_interactive_chart_controls_and_canvases():
    html = MOCKUP_PATH.read_text(encoding="utf-8")

    for expected_id in [
        'id="scenario"',
        'id="n"',
        'id="buckets"',
        'id="run"',
        'id="chart-raw"',
        'id="chart-baseline"',
        'id="chart-learned"',
        'id="metrics-body"',
        'id="correctness"',
    ]:
        assert expected_id in html

    assert "function run()" in html
    assert 'addEventListener("click", run)' in html
