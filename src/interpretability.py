"""Genomic Annotation & Biological Pathway Interpretability.

Maps predictive CpG loci to target gene promoters, CpG island contexts (GRCh38),
and aggregates molecular weights across cellular aging pathways.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from .data import CANONICAL_CPG_ANNOTATIONS


def annotate_cpg_weights(cpg_weights: Dict[str, float], top_n: Optional[int] = None) -> pd.DataFrame:
    """Annotate model CpG coefficients with genomic coordinates and biological pathways.

    Args:
        cpg_weights: Dict of {cpg_id: weight}
        top_n: Optional limit to top N features by absolute weight

    Returns:
        DataFrame with genomic annotations
    """
    records = []
    sorted_cpgs = sorted(cpg_weights.items(), key=lambda item: abs(item[1]), reverse=True)

    if top_n is not None:
        sorted_cpgs = sorted_cpgs[:top_n]

    for cpg_id, weight in sorted_cpgs:
        meta = CANONICAL_CPG_ANNOTATIONS.get(
            cpg_id,
            {
                "gene": "Intergenic / Uncharacterized",
                "chr": "chrN",
                "pos": 0,
                "region": "OpenSea",
                "island": "Non-CpG Island",
                "direction": "hyper" if weight > 0 else "hypo",
                "effect": float(weight),
                "pathway": "General Cellular Homeostasis",
                "description": "Background genomic CpG dinucleotide.",
            },
        )

        records.append({
            "cpg_id": cpg_id,
            "weight": round(weight, 4),
            "abs_weight": round(abs(weight), 4),
            "direction": "Hypermethylated (+)" if weight > 0 else "Hypomethylated (-)",
            "gene": meta["gene"],
            "chromosome": meta["chr"],
            "position": meta["pos"],
            "regulatory_region": meta["region"],
            "cpg_island_context": meta["island"],
            "pathway": meta["pathway"],
            "functional_annotation": meta["description"],
        })

    return pd.DataFrame(records)


def compute_pathway_enrichment(annotated_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate epigenetic model weights by biological pathway.

    Quantifies which cellular processes (DNA repair, senescence, lipid metabolism)
    dominate the clock's age readout.
    """
    grouped = (
        annotated_df.groupby("pathway")
        .agg(
            total_abs_weight=("abs_weight", "sum"),
            mean_weight=("weight", "mean"),
            cpg_count=("cpg_id", "count"),
            genes=("gene", lambda g: ", ".join(sorted(set(g)))),
        )
        .reset_index()
    )

    grouped["relative_contribution_pct"] = (
        100.0 * grouped["total_abs_weight"] / (grouped["total_abs_weight"].sum() + 1e-12)
    )
    grouped = grouped.sort_values(by="total_abs_weight", ascending=False).reset_index(drop=True)
    grouped["relative_contribution_pct"] = grouped["relative_contribution_pct"].round(1)
    grouped["total_abs_weight"] = grouped["total_abs_weight"].round(3)
    return grouped
