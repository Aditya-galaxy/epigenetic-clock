"""DNA Methylation Data Ingestion and Multi-Tissue Cohort Simulator.

Simulates and processes Illumina HumanMethylation array beta-values (0.0 to 1.0)
across diverse human tissues and cellular reprogramming trajectories.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# Canonical human aging-associated CpG loci identified in Horvath (2013), Hannum (2013),
# and Levine (PhenoAge 2018) landmark geroscience studies.
CANONICAL_CPG_ANNOTATIONS: Dict[str, dict] = {
    # Hypermethylated with age (positive weight)
    "cg16867657": {
        "gene": "ELOVL2",
        "chr": "chr6",
        "pos": 11044634,
        "region": "TSS200",
        "island": "Island",
        "direction": "hyper",
        "effect": 0.85,
        "pathway": "Fatty Acid Elongation & Lipid Homeostasis",
        "description": "Elongation of very long chain fatty acids-like 2. Premier pan-tissue aging marker.",
    },
    "cg24724428": {
        "gene": "ELOVL2",
        "chr": "chr6",
        "pos": 11044670,
        "region": "5'UTR",
        "island": "Island",
        "direction": "hyper",
        "effect": 0.80,
        "pathway": "Fatty Acid Elongation & Lipid Homeostasis",
        "description": "Secondary ELOVL2 CpG marker; hypermethylation alters cell membrane composition.",
    },
    "cg06639320": {
        "gene": "FHL2",
        "chr": "chr2",
        "pos": 105399284,
        "region": "TSS1500",
        "island": "Island",
        "direction": "hyper",
        "effect": 0.78,
        "pathway": "Cardiac Remodeling & Cytoskeletal Organization",
        "description": "Four and a half LIM domains 2; promoter hypermethylation associated with cardiovascular aging.",
    },
    "cg22454769": {
        "gene": "FHL2",
        "chr": "chr2",
        "pos": 105399510,
        "region": "TSS200",
        "island": "Island",
        "direction": "hyper",
        "effect": 0.72,
        "pathway": "Cardiac Remodeling & Cytoskeletal Organization",
        "description": "FHL2 proximal promoter CpG.",
    },
    "cg16419235": {
        "gene": "PENK",
        "chr": "chr8",
        "pos": 56434440,
        "region": "TSS200",
        "island": "N_Shore",
        "direction": "hyper",
        "effect": 0.74,
        "pathway": "Neuroendocrine & Opioid Signaling",
        "description": "Proenkephalin; hypermethylation linked to neuroendocrine decline.",
    },
    "cg07955995": {
        "gene": "KLF14",
        "chr": "chr7",
        "pos": 130733857,
        "region": "Body",
        "island": "OpenSea",
        "direction": "hyper",
        "effect": 0.69,
        "pathway": "Adipose Tissue Homeostasis & Metabolic Regulation",
        "description": "Krüppel-like factor 14; master regulator of age-dependent metabolic drift.",
    },
    "cg07553198": {
        "gene": "TRIM59",
        "chr": "chr3",
        "pos": 160450119,
        "region": "TSS1500",
        "island": "Island",
        "direction": "hyper",
        "effect": 0.71,
        "pathway": "Cellular Senescence & Innate Immunity",
        "description": "Tripartite motif containing 59; regulates cellular senescence checkpoints.",
    },
    "cg01530101": {
        "gene": "GRIA2",
        "chr": "chr4",
        "pos": 157335122,
        "region": "TSS200",
        "island": "Island",
        "direction": "hyper",
        "effect": 0.65,
        "pathway": "Synaptic Plasticity & Neurotransmission",
        "description": "Glutamate ionotropic receptor AMPA type subunit 2.",
    },
    "cg09809672": {
        "gene": "EDARADD",
        "chr": "chr1",
        "pos": 236528822,
        "region": "TSS200",
        "island": "Island",
        "direction": "hyper",
        "effect": 0.66,
        "pathway": "Ectodermal Development & NF-kB Signaling",
        "description": "EDAR associated death domain; strongly linear with chronological age.",
    },
    "cg14361627": {
        "gene": "OTX2",
        "chr": "chr14",
        "pos": 56839210,
        "region": "TSS1500",
        "island": "Island",
        "direction": "hyper",
        "effect": 0.62,
        "pathway": "Developmental Homeostasis & Chromatin Organization",
        "description": "Orthodenticle homeobox 2.",
    },
    # Hypomethylated with age (negative weight - global heterochromatin erosion)
    "cg19761276": {
        "gene": "TFAP2E",
        "chr": "chr1",
        "pos": 28628045,
        "region": "TSS200",
        "island": "N_Shore",
        "direction": "hypo",
        "effect": -0.68,
        "pathway": "Transcriptional Regulation & Tumor Suppression",
        "description": "Transcription factor AP-2 epsilon; progressive loss of methylation with age.",
    },
    "cg25547614": {
        "gene": "PITX2",
        "chr": "chr4",
        "pos": 110620890,
        "region": "TSS1500",
        "island": "Island",
        "direction": "hypo",
        "effect": -0.64,
        "pathway": "Developmental Patterning & Left-Right Asymmetry",
        "description": "Paired like homeodomain 2; age-associated hypomethylation.",
    },
    "cg02228185": {
        "gene": "ASPA",
        "chr": "chr17",
        "pos": 3390821,
        "region": "Body",
        "island": "OpenSea",
        "direction": "hypo",
        "effect": -0.67,
        "pathway": "Myelin Maintenance & Acetate Metabolism",
        "description": "Aspartoacylase; hypomethylation correlates with loss of myelin integrity.",
    },
    "cg08090772": {
        "gene": "CSNK1D",
        "chr": "chr17",
        "pos": 80198421,
        "region": "TSS1500",
        "island": "Island",
        "direction": "hypo",
        "effect": -0.60,
        "pathway": "Circadian Rhythm & DNA Damage Checkpoints",
        "description": "Casein kinase 1 delta; clock gene exhibiting age-dependent hypomethylation.",
    },
    "cg22736354": {
        "gene": "NHLRC1",
        "chr": "chr6",
        "pos": 18121900,
        "region": "Body",
        "island": "OpenSea",
        "direction": "hypo",
        "effect": -0.58,
        "pathway": "Ubiquitin-Proteasome System & Glycogen Metabolism",
        "description": "NHL repeat containing E3 ubiquitin protein ligase 1.",
    },
}


@dataclass
class MethylationCohort:
    """Container for human DNA methylation matrix, metadata, and ground truth ages."""
    beta_values: np.ndarray       # Shape (N_samples, M_cpgs)
    cpg_ids: List[str]            # Length M
    sample_ids: List[str]         # Length N
    chronological_age: np.ndarray # (N,) in years
    tissue: List[str]             # (N,) tissue labels ('Blood', 'Brain', 'Liver', etc.)
    is_reprogrammed: np.ndarray   # (N,) boolean flag for iPSCs or OSKM treated

    def to_dataframe(self) -> pd.DataFrame:
        """Export as a tidy pandas DataFrame."""
        df = pd.DataFrame(self.beta_values, columns=self.cpg_ids, index=self.sample_ids)
        df.insert(0, "chronological_age", self.chronological_age)
        df.insert(1, "tissue", self.tissue)
        df.insert(2, "is_reprogrammed", self.is_reprogrammed)
        return df


class MethylationCohortGenerator:
    """Simulates realistic human multi-tissue DNA methylation array cohorts.

    Includes canonical age-associated CpGs with non-linear kinetic trajectories,
    heterogeneous tissue baselines, biological technical noise, and cellular reprogramming (iPSC).
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_cohort(
        self,
        n_samples: int = 500,
        n_background_cpgs: int = 250,
        tissues: Optional[List[str]] = None,
        min_age: float = 0.5,
        max_age: float = 95.0,
    ) -> MethylationCohort:
        """Generate a multi-tissue human cohort with realistic age distribution.

        Args:
            n_samples: Number of individual donors
            n_background_cpgs: Number of non-age-associated background CpGs
            tissues: List of tissue types to sample
            min_age: Minimum donor age in years
            max_age: Maximum donor age in years

        Returns:
            MethylationCohort instance
        """
        if tissues is None:
            tissues = ["Whole_Blood", "Cortex_Brain", "Liver", "Skin_Fibroblast", "Heart"]

        # 1. Sample realistic age distribution (log-uniform + uniform mix across lifespan)
        u = self.rng.uniform(0.0, 1.0, size=n_samples)
        # 20% pediatric/young adults (0-25 yrs), 50% middle age (25-65 yrs), 30% elderly (65-95 yrs)
        ages = np.where(
            u < 0.20,
            self.rng.uniform(min_age, 25.0, size=n_samples),
            np.where(
                u < 0.70,
                self.rng.uniform(25.0, 65.0, size=n_samples),
                self.rng.uniform(65.0, max_age, size=n_samples),
            ),
        )
        ages = np.round(ages, 1)

        # 2. Assign tissue types
        sample_tissues = self.rng.choice(tissues, size=n_samples)
        sample_ids = [f"donor_{i+1:04d}_{sample_tissues[i][:3]}" for i in range(n_samples)]

        # 3. Assemble CpG probes: canonical markers + background markers
        canonical_cpg_ids = list(CANONICAL_CPG_ANNOTATIONS.keys())
        bg_cpg_ids = [f"cg{self.rng.integers(10000000, 99999999)}" for _ in range(n_background_cpgs)]
        all_cpg_ids = canonical_cpg_ids + bg_cpg_ids
        total_cpgs = len(all_cpg_ids)

        # 4. Generate beta values
        beta_matrix = np.zeros((n_samples, total_cpgs), dtype=np.float32)

        # Normalized age coordinate [0, 1]
        norm_age = (ages - min_age) / (max_age - min_age)

        # A. Fill canonical aging CpGs with physiological biological kinetics
        for j, cpg_id in enumerate(canonical_cpg_ids):
            meta = CANONICAL_CPG_ANNOTATIONS[cpg_id]
            effect = meta["effect"]
            direction = meta["direction"]

            # Baseline baseline beta-value at birth
            if direction == "hyper":
                base_beta = self.rng.uniform(0.08, 0.22)
                # Hypermethylation curve (steepest in early life, plateauing near senescence)
                # Beta(age) = base + delta * (1 - exp(-k * age))
                delta = 0.55 * abs(effect)
                k = 0.035
                age_trend = delta * (1.0 - np.exp(-k * ages))
            else:
                base_beta = self.rng.uniform(0.75, 0.90)
                # Hypomethylation curve
                delta = 0.50 * abs(effect)
                k = 0.030
                age_trend = -delta * (1.0 - np.exp(-k * ages))

            # Add tissue-specific baseline shift
            tissue_shift = np.array(
                [0.04 if t == "Cortex_Brain" else (-0.03 if t == "Liver" else 0.0) for t in sample_tissues]
            )

            # Biological noise + technical measurement error
            noise = self.rng.normal(0.0, 0.035, size=n_samples)

            beta_col = base_beta + age_trend + tissue_shift + noise
            beta_matrix[:, j] = np.clip(beta_col, 0.01, 0.99)

        # B. Fill background CpGs (tissue-specific, biological variance, no global age correlation)
        n_canonical = len(canonical_cpg_ids)
        for j_offset in range(n_background_cpgs):
            col_idx = n_canonical + j_offset
            # Unimodal or bimodal methylation distribution typical of human methylome
            is_high_meth = self.rng.choice([True, False], p=[0.45, 0.55])
            if is_high_meth:
                base = self.rng.uniform(0.70, 0.88)
            else:
                base = self.rng.uniform(0.10, 0.30)

            # Random tissue variance and measurement noise
            t_noise = self.rng.normal(0.0, 0.06, size=n_samples)
            beta_col = base + t_noise
            beta_matrix[:, col_idx] = np.clip(beta_col, 0.01, 0.99)

        is_reprogrammed = np.zeros(n_samples, dtype=bool)

        return MethylationCohort(
            beta_values=beta_matrix,
            cpg_ids=all_cpg_ids,
            sample_ids=sample_ids,
            chronological_age=ages,
            tissue=list(sample_tissues),
            is_reprogrammed=is_reprogrammed,
        )

    def generate_reprogramming_timecourse(
        self,
        donor_age: float = 62.0,
        cpg_ids: Optional[List[str]] = None,
        timepoints: Optional[List[int]] = None,
        replicates: int = 4,
    ) -> MethylationCohort:
        """Simulate Yamanaka OSKM induction timecourse (Days 0, 3, 7, 12, 21 iPSC).

        Models the step-wise rejuvenation kinetics observed in partial reprogramming
        (Ocampo et al. 2016, Gill et al. 2022, Horvath 2013).
        """
        if cpg_ids is None:
            cpg_ids = list(CANONICAL_CPG_ANNOTATIONS.keys())
        if timepoints is None:
            timepoints = [0, 3, 7, 12, 21]

        samples = []
        ages = []
        tissues = []
        reprogrammed_flags = []
        beta_rows = []

        for t in timepoints:
            # Rejuvenation fraction: at Day 0, 0% rejuvenated. At Day 21 (iPSC), 100% reset to embryonic state.
            # Follows sigmoidal kinetics of OSKM chromatin remodeling
            rejuvenation_factor = 1.0 / (1.0 + np.exp(-0.35 * (t - 9.0)))
            effective_bio_age = donor_age * (1.0 - rejuvenation_factor)

            for rep in range(replicates):
                s_id = f"OSKM_Day{t:02d}_rep{rep+1}"
                samples.append(s_id)
                # Chronological donor age remains 62.0
                ages.append(donor_age)
                tissues.append("Reprogramming_Fibroblast" if t < 21 else "Induced_Pluripotent_Stem_Cell")
                reprogrammed_flags.append(t > 0)

                # Generate beta vector matching effective biological age
                row = []
                for cpg in cpg_ids:
                    if cpg in CANONICAL_CPG_ANNOTATIONS:
                        meta = CANONICAL_CPG_ANNOTATIONS[cpg]
                        direction = meta["direction"]
                        effect = meta["effect"]
                        if direction == "hyper":
                            val = 0.15 + (0.50 * abs(effect)) * (1.0 - np.exp(-0.035 * effective_bio_age))
                        else:
                            val = 0.85 - (0.45 * abs(effect)) * (1.0 - np.exp(-0.030 * effective_bio_age))
                    else:
                        val = self.rng.uniform(0.2, 0.8)

                    noise = self.rng.normal(0.0, 0.02)
                    row.append(np.clip(val + noise, 0.01, 0.99))
                beta_rows.append(row)

        return MethylationCohort(
            beta_values=np.array(beta_rows, dtype=np.float32),
            cpg_ids=cpg_ids,
            sample_ids=samples,
            chronological_age=np.array(ages, dtype=np.float32),
            tissue=tissues,
            is_reprogrammed=np.array(reprogrammed_flags, dtype=bool),
        )
