"""Department association rules (proxy market basket)."""

from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from retail_analytics.config import AssociationConfig, get_config


def build_department_transactions(sales: pd.DataFrame) -> list[list[str]]:
    """Build transactions as co-occurring departments per store-week."""
    grouped = (
        sales.groupby(["Store", "Date"])["Dept"]
        .apply(lambda depts: sorted({str(d) for d in depts}))
        .reset_index(name="Departments")
    )
    return grouped["Departments"].tolist()


def mine_association_rules(
    sales: pd.DataFrame,
    config: AssociationConfig | None = None,
    max_departments: int = 40,
) -> pd.DataFrame:
    """Mine department co-occurrence rules on a capped department universe."""
    config = config or get_config().association
    top_depts = (
        sales.groupby("Dept")["Weekly_Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(max_departments)
        .index
    )
    filtered = sales[sales["Dept"].isin(top_depts)]
    transactions = build_department_transactions(filtered)
    empty = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    if not transactions:
        return empty

    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(transactions)
    df = pd.DataFrame(encoded, columns=encoder.columns_)

    frequent = apriori(df, min_support=config.min_support, use_colnames=True)
    if frequent.empty:
        return empty

    rules = association_rules(
        frequent,
        metric="confidence",
        min_threshold=config.min_confidence,
    )
    rules = rules[rules["lift"] >= config.min_lift]
    rules = rules[
        rules["antecedents"].apply(len) + rules["consequents"].apply(len) >= config.min_length
    ]

    return rules[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ].sort_values("lift", ascending=False).reset_index(drop=True)
