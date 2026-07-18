import numpy as np
import pandas as pd

from fanoseq.information_theory import (
    conditional_mutual_information,
    empirical_entropy,
    information_audit_table,
    interaction_information,
    mutual_information,
    total_correlation,
)


def test_xor_exposes_higher_order_information() -> None:
    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]
    z = [0, 1, 1, 0]

    assert empirical_entropy(x) == 1.0
    assert mutual_information(x, y) == 0.0
    assert conditional_mutual_information(x, y, z) == 1.0
    assert interaction_information(x, y, z) == -1.0
    assert total_correlation(x, y, z) == 1.0


def test_information_audit_reports_pairwise_and_three_way_measures() -> None:
    table = pd.DataFrame(
        {
            "x": np.arange(12),
            "y": np.arange(12) % 3,
            "label": ["a", "b"] * 6,
        }
    )
    audit = information_audit_table(table, ["x", "y", "label"], numeric_bins=3)
    assert set(audit["measure"]) == {
        "mutual_information",
        "interaction_information",
        "total_correlation",
    }
    assert set(audit["estimator"]) == {"empirical_plugin"}
