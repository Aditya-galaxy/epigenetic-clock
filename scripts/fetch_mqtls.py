"""Ask GoDMC which of each clock's CpGs can be instrumented at all.

Mendelian randomisation needs genetic instruments. For a CpG that means cis-mQTLs: SNPs whose
genotype moves methylation at that site. Before anything can be said about whether a clock is
built from causal sites, there is a prior question with a factual answer: do its CpGs have
instruments in the first place?

That question also carries a prediction. CausAge, DamAge and AdaptAge were selected by
epigenome-wide Mendelian randomisation, which can only consider CpGs that are instrumentable, so
essentially all of their sites should have mQTLs. Horvath and Hannum chose sites by elastic net
on methylation alone, with no genetic step, so their coverage should fall back to whatever the
background rate is. If that difference does not appear, I have misunderstood how the Ying clocks
were built.

Instruments come from GoDMC (Min et al., Nature Genetics 2021), which is public and needs no
account. Only cis associations are used, at genome-wide significance, and only the rows GoDMC
marks as clumped, which are the approximately independent signals per CpG.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import torch_stub  # noqa: F401,E402

from biolearn.model_gallery import ModelGallery  # noqa: E402

API = "http://api.godmc.org.uk/v0.1/query"
UA = "epigenetic-clock-study/0.1 (student project; contact via repository)"
CLOCKS = ["Horvathv1", "Hannum", "PhenoAge", "YingCausAge", "YingDamAge", "YingAdaptAge"]
PVAL = 5e-8
CACHE = "data/godmc_cis_mqtls.json"


def clock_cpgs(gallery):
    out = {}
    for name in CLOCKS:
        idx = gallery.get(name).coefficients.index
        out[name] = sorted({c for c in idx if c != "intercept"})
    return out


def query(cpgs, pval=PVAL, tries=4):
    payload = json.dumps({"cpgs": list(cpgs), "cistrans": "cis", "pval": pval}).encode()
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                API, data=payload,
                headers={"Content-Type": "application/json", "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as fh:
                return json.load(fh)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"GoDMC query failed after {tries} tries: {last}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch", type=int, default=40)
    args = ap.parse_args()

    gallery = ModelGallery()
    by_clock = clock_cpgs(gallery)
    all_cpgs = sorted({c for v in by_clock.values() for c in v})
    print(f"{len(all_cpgs)} unique CpGs across {len(CLOCKS)} clocks", flush=True)

    have = {}
    if os.path.exists(CACHE):
        have = json.load(open(CACHE))
        print(f"cache holds {len(have)} CpGs already", flush=True)

    todo = [c for c in all_cpgs if c not in have]
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        rows = query(chunk)
        for cpg in chunk:
            have[cpg] = []
        for r in rows:
            have.setdefault(r["cpg"], []).append(
                {k: r.get(k) for k in ("rsid", "beta_a1", "se", "pval", "samplesize",
                                       "clumped", "cistrans", "a1", "a2", "freq_a1")})
        done = i + len(chunk)
        print(f"  {done}/{len(todo)} queried, {sum(1 for v in have.values() if v)} with hits",
              flush=True)
        os.makedirs("data", exist_ok=True)
        with open(CACHE, "w") as fh:
            json.dump(have, fh)
        time.sleep(1)

    with open(CACHE, "w") as fh:
        json.dump(have, fh)
    print(f"written {CACHE}: {len(have)} CpGs")


if __name__ == "__main__":
    main()
