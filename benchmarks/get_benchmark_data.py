"""Fetch the inborn-error-of-metabolism benchmark data.

The untargeted plasma metabolomics z-scores of Miller et al. (2015) and
Thistlethwaite et al. (2020, Baylor), with each sample's confirmed diagnosis,
are distributed as datasets in the CTD R package on CRAN (MIT licence). This
script downloads the pinned package source, and uses R to export:

    data/benchmark/thistlethwaite2020.csv   metabolites x samples (+ annotations)
    data/benchmark/miller2015.csv           metabolites x samples (+ annotations, diagnosis row)
    data/benchmark/cohorts_coded.tsv        cohort -> sample id (Thistlethwaite2020)

Requires R (Rscript) on this machine; no R packages are needed.

    python benchmarks/get_benchmark_data.py
"""
import hashlib
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

CTD_URL = "https://cran.r-project.org/src/contrib/Archive/CTD/CTD_1.3.tar.gz"
CTD_URL_CURRENT = "https://cran.r-project.org/src/contrib/CTD_1.3.tar.gz"
OUT = Path("data/benchmark")

EXPORT_R = r"""
args <- commandArgs(trailingOnly = TRUE)
pkg <- args[1]; out <- args[2]
env <- new.env()
for (f in c("Thistlethwaite2020", "Miller2015", "cohorts_coded")) {
  load(file.path(pkg, "data", paste0(f, ".RData")), envir = env)
}
write.csv(env$Thistlethwaite2020, file.path(out, "thistlethwaite2020.csv"))
write.csv(env$Miller2015, file.path(out, "miller2015.csv"))
cc <- env$cohorts_coded
long <- data.frame(cohort = rep(names(cc), lengths(cc)), sample = unlist(cc, use.names = FALSE))
write.table(long, file.path(out, "cohorts_coded.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
"""


def find_rscript():
    exe = shutil.which("Rscript")
    if exe:
        return exe
    for root in (Path("C:/Program Files/R"), Path("C:/Program Files (x86)/R")):
        hits = sorted(root.glob("R-*/bin/Rscript.exe"))
        if hits:
            return str(hits[-1])
    raise SystemExit("Rscript not found. Install R (https://cran.r-project.org) or put Rscript on PATH.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tar = OUT / "CTD_1.3.tar.gz"
    if not tar.exists():
        for url in (CTD_URL_CURRENT, CTD_URL):
            try:
                print(f"Downloading {url}")
                urllib.request.urlretrieve(url, tar)
                break
            except Exception as err:  # the current URL moves to Archive/ once a newer version appears
                print(f"  failed: {err}")
        else:
            raise SystemExit("Could not download the CTD package from CRAN.")
    print("CTD_1.3.tar.gz sha256", hashlib.sha256(tar.read_bytes()).hexdigest())
    with tarfile.open(tar) as t:
        t.extractall(OUT / "CTD_pkg")
    script = OUT / "export_ctd.R"
    script.write_text(EXPORT_R, encoding="ascii")
    subprocess.run([find_rscript(), str(script), str(OUT / "CTD_pkg" / "CTD"), str(OUT)], check=True)
    for f in ("thistlethwaite2020.csv", "miller2015.csv", "cohorts_coded.tsv"):
        print(f"wrote {OUT / f} ({(OUT / f).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
