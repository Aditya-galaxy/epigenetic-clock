"""Pull the parental-lifespan effects for our instrument SNPs only.

Outcome: Timmers et al. 2019 (eLife), lifespan of both parents combined, from the GWAS Catalog
at GCST006698. That accession is chosen rather than its two siblings because its standard errors
are the smallest of the three, which is what the combined analysis should look like beside the
father-only and mother-only ones.

The GWAS Catalog FTP is used rather than OpenGWAS because OpenGWAS has required an access token
since 1 May 2024 and this should run for anyone who clones the repository.

The file holds about ten million rows and we need a few thousand, so it is streamed and filtered
rather than downloaded. Nothing but the matching rows is ever held in memory.
"""
import argparse
import gzip
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

URL = ("https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
       "GCST006001-GCST007000/GCST006698/9548053.gz")
UA = "epigenetic-clock-study/0.1 (student project; contact via repository)"
MQTLS = "data/godmc_cis_mqtls.json"
OUT = "data/parental_lifespan_instruments.tsv"


def wanted_rsids(path):
    mqtls = json.load(open(path))
    return {h["rsid"] for v in mqtls.values() for h in v
            if h.get("clumped") == 1 and h.get("rsid")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    if not os.path.exists(MQTLS):
        sys.exit(f"missing {MQTLS}: run scripts/fetch_mqtls.py first")
    want = wanted_rsids(MQTLS)
    print(f"looking for {len(want)} instrument SNPs in the outcome GWAS", flush=True)

    req = urllib.request.Request(URL, headers={"User-Agent": UA})
    found, scanned = 0, 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with urllib.request.urlopen(req, timeout=600) as raw, \
            gzip.open(raw, "rt") as fh, open(args.out, "w") as out:
        header = fh.readline().rstrip("\n").split("\t")
        out.write("\t".join(header) + "\n")
        idx = header.index("SNP")
        for line in fh:
            scanned += 1
            if scanned % 2_000_000 == 0:
                print(f"  scanned {scanned:,} rows, matched {found}", flush=True)
            fields = line.rstrip("\n").split("\t")
            if fields[idx] in want:
                out.write(line)
                found += 1
                if found == len(want):
                    break

    print(f"scanned {scanned:,} rows; matched {found} of {len(want)} instruments")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
