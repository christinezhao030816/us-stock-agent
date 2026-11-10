import unittest

from tools.internal_provenance import internal_traceability_summary


class InternalProvenanceTests(unittest.TestCase):
    def test_traceability_requires_every_required_field(self) -> None:
        complete = {
            "input_fields": {"a": 1},
            "formula_or_rule": "rule/v1",
            "result": 2,
            "calculation_timestamp": "2026-09-21T00:00:00Z",
            "provenance_path": "response.path",
            "deterministic": True,
        }
        incomplete = {**complete, "provenance_path": ""}
        summary = internal_traceability_summary([complete, incomplete])
        self.assertEqual(summary["traceable_internal_claims"], 1)
        self.assertEqual(summary["internal_traceability_coverage_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
