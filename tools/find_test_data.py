"""Find candidate evaluation data: human metabolomics where a named gene was perturbed.

Sweeps three sources end to end instead of relying on search-engine recall:
  workbench    every Metabolomics Workbench study (REST): title, species, sample
               factors and the mwTab study/project summary
  metabolights every MetaboLights study (EBI web service): title, description,
               organisms, factor names
  literature   Europe PMC: papers with metabolomics in the title and a gene
               perturbation in the abstract, plus papers that cite a MetaboLights
               or Metabolomics Workbench deposit
Gene symbols are matched against Human-GEM's genes and flagged when they are in the
three_rule network (the ones a causal-gene test can use). A symbol counts only when a
perturbation word sits near it (knockout, siRNA, deficiency, inhibitor, ...) or it
appears in a sample factor. Human only (D25). Rows already in
benchmarks/data_registry.tsv are flagged, not dropped.

    python tools/find_test_data.py                      # all sources
    python tools/find_test_data.py --sources workbench  # one source
    python tools/find_test_data.py --refresh            # ignore the cache

Writes results/data_discovery/: candidates.tsv (one row per study or paper and gene),
report.md, and cache/ (raw API responses, so reruns only fetch what is new).
"""
import argparse
import hashlib
import json
import os
import re
import ssl
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "data_discovery"
REGISTRY = REPO / "benchmarks" / "data_registry.tsv"

MW = "https://www.metabolomicsworkbench.org/rest/study/study_id"
ML = "https://www.ebi.ac.uk/metabolights/ws/studies"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
LITERATURE = {
    "title_metabolomics_abstract_perturbation":
        'TITLE:(metabolomic OR metabolomics OR metabolome OR "metabolic profiling" OR "metabolite profiling") AND '
        'ABSTRACT:("inborn error" OR knockout OR knockdown OR CRISPR OR siRNA OR shRNA OR "loss-of-function" OR '
        'deficiency OR deficient OR "gene deletion")',
    "cites_metabolights": "ACCESSION_TYPE:metabolights",
    "cites_workbench": '"Metabolomics Workbench" AND (knockout OR knockdown OR deficiency OR deficient OR CRISPR OR siRNA OR shRNA)',
}

# Perturbation words; the class decides how useful the unit is (D22: loss of the named gene first).
LOSS = r"knock-?outs?|knock-?downs?|\bKO\b|\bKD\b|deficien\w*|\bnull\b|-/-|mutants?|mutations?|mutated|" \
       r"loss[- ]of[- ]function|crispr|sgRNA|shRNA|siRNA|silenc\w*|delet\w*|depletion|ablat\w*|inborn|disorder"
CHEM = r"inhibit\w*|antagonist|blockade"
GAIN = r"overexpress\w*|over-express\w*|gain[- ]of[- ]function|transgenic"
CONTEXT = re.compile(f"({LOSS}|{CHEM}|{GAIN})", re.I)
# Gene-symbol look-alikes common in metabolomics text.
STOP = {"MS", "LC", "GC", "CE", "NMR", "PCA", "ATP", "ADP", "AMP", "NAD", "FAD", "GSH", "ROS", "TCA", "SET", "CAT",
        "MAX", "IMPACT", "PC", "HR", "MET", "CA", "AR", "TG", "CD", "DM", "GH", "HP", "MR", "PS", "PE", "PI", "SM",
        "CO", "NO", "HD", "RA", "SD", "CT", "ER", "OS", "UV", "PH", "TPA", "DCA", "PAF", "SDS", "ACE", "APP", "CAD",
        "ENO", "GLUL", "SRM", "MAT", "PAH", "AHR", "BMP", "CHI", "COX", "CPS", "DES", "EPO", "HPD", "LPL", "NPC",
        "PKM", "POR", "PTS", "SDC", "SMS", "TAT", "TH", "TST", "AGT", "AFP", "CRP", "GPI", "MIF", "PGD", "PLA", "SOD"}
# PAH, TH, CPS, TAT ... are real causal genes but also common abbreviations; they still count from sample factors.
# Designs the evaluation rules set aside (D22): flagged and scored down, not dropped.
DESIGN = {"tracing": r"\b13C\b|\b15N\b|isotop\w* (?:tracing|labell?ing)|tracer|fluxomic|\btracing\b",
          "lipid only": r"\blipidom\w*|\bsphingolipid\w*|\boxylipin\w*"}
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "metaboenrich-find-test-data (research; contact via repo)"
FAILED = []


def trust_windows_roots(cache):
    """Behind a TLS-inspecting proxy (e.g. Zscaler) the proxy's root is in the Windows store
    but not in certifi. Verify against certifi plus the Windows roots; never disable checks."""
    if sys.platform != "win32" or os.environ.get("REQUESTS_CA_BUNDLE"):
        return
    import certifi
    pems = [Path(certifi.where()).read_text(encoding="utf-8")]
    for store in ("ROOT", "CA"):
        pems += [ssl.DER_cert_to_PEM_cert(c) for c, enc, _ in ssl.enum_certificates(store) if enc == "x509_asn"]
    f = cache / "ca_bundle.pem"
    f.write_text("\n".join(pems), encoding="utf-8")
    SESSION.verify = str(f)


# ---- fetching with a cache -------------------------------------------------------
def fetch(url, cache, params=None, kind="json", max_bytes=None, refresh=False, timeout=90):
    key = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    f = cache / f"{key}.{ 'json' if kind == 'json' else 'txt'}"
    if f.exists() and not refresh:
        txt = f.read_text(encoding="utf-8")
        return json.loads(txt) if kind == "json" else txt
    for attempt in range(3):
        try:
            r = SESSION.get(url, params=params, timeout=timeout, stream=max_bytes is not None)
            if max_bytes:                         # mwTab: stop after the header sections
                buf = b""
                for chunk in r.iter_content(65536):
                    buf += chunk
                    if len(buf) > max_bytes or b"#SUBJECT_SAMPLE_FACTORS" in buf or b"#COLLECTION" in buf:
                        break
                r.close()
                txt = buf.decode("utf-8", "replace")
            else:
                txt = r.text
            if r.status_code >= 500:
                raise requests.RequestException(r.status_code)
            f.write_text(txt, encoding="utf-8")
            return json.loads(txt) if kind == "json" else txt
        except (requests.RequestException, json.JSONDecodeError) as e:
            err = e
            time.sleep(2 * (attempt + 1))
    FAILED.append(f"{url} {params or ''}: {type(err).__name__}")
    return None


def pmap(fn, items, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


# ---- genes -------------------------------------------------------------------------
def gene_vocabulary(cache):
    """Human-GEM gene symbols, and whether each is in the three_rule network."""
    f = cache / "genes.tsv"
    if f.exists():
        return pd.read_csv(f, sep="\t")
    from metaboenrich import structure
    from metaboenrich.gem import load_sbml
    from metaboenrich.network import build_graph
    model = load_sbml(str(REPO / "data/Human-GEM.xml"))
    st = next(s for s in structure.stages(model, "three_rule", atom_dir=str(REPO / "data")) if s[0] == "three_rule")
    graph = build_graph(model, st[3], st[4], st[5], **st[6])
    net = set().union(*graph.reaction_genes.values())
    g = pd.DataFrame({"ensembl": list(model.genes), "symbol": list(model.genes.values())})
    g["in_network"] = g.ensembl.isin(net)
    g = g[g.symbol.astype(str).str.len() > 0].drop_duplicates("symbol")
    g.to_csv(f, sep="\t", index=False)
    return g


def find_genes(text, symbols, window=80):
    """Gene symbols with a perturbation word within `window` characters, and that word's class."""
    hits = {}
    for m in re.finditer(r"(?<![A-Za-z0-9])(?:sh|si|sg)?([A-Z][A-Z0-9]{1,9})(?![a-z0-9])", text or ""):
        s = m.group(1)
        if s not in symbols or s in STOP:
            continue
        near = text[max(0, m.start() - window): m.end() + window]
        c = CONTEXT.findall(near)
        if c or m.group(0) != s:                  # an sh/si/sg prefix is itself a perturbation
            words = [w.lower() for w in c] or ["rnai/crispr prefix"]
            hits.setdefault(s, set()).update(words)
    return hits


def perturbation_class(words):
    w = " ".join(words)
    if re.search(LOSS, w, re.I) or "prefix" in w:
        return "loss"
    if re.search(CHEM, w, re.I):
        return "chemical"
    if re.search(GAIN, w, re.I):
        return "gain"
    return ""


def section(txt, tag):
    """Concatenated values of an mwTab field (multi-line fields repeat the tag)."""
    return " ".join(l.split("\t", 1)[-1].strip() for l in (txt or "").splitlines() if l.startswith(tag))


# ---- sources ------------------------------------------------------------------------
def workbench(cache, refresh):
    allst = fetch(f"{MW}/ST/summary", cache, refresh=refresh, timeout=600)   # one large listing
    if not allst:
        raise SystemExit(f"workbench: could not list studies ({FAILED[-1] if FAILED else 'empty'})")
    studies = [s for s in allst.values() if "sapiens" in s.get("species", "")]
    print(f"workbench: {len(allst)} studies, {len(studies)} human", flush=True)

    def one(s):
        sid = s["study_id"]
        fac = fetch(f"{MW}/{sid}/factors", cache, refresh=refresh) or {}
        if isinstance(fac, dict) and "study_id" in fac:
            fac = {"1": fac}
        factors = sorted({v.get("factors", "") for v in (fac.values() if isinstance(fac, dict) else [])})
        mw = fetch(f"{MW}/{sid}/mwtab/txt", cache, kind="txt", max_bytes=400_000, refresh=refresh) or ""
        return {"source": "workbench", "id": sid, "title": s["study_title"], "species": s["species"],
                "n_samples": s.get("number_of_samples", ""), "license": s.get("license", ""),
                "text": " ".join([s["study_title"], section(mw, "ST:STUDY_SUMMARY"), section(mw, "PR:PROJECT_SUMMARY"),
                                  section(mw, "PR:PROJECT_TITLE")]),
                "factors": " | ".join(factors)[:4000], "url": f"https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?StudyID={sid}"}
    return pmap(one, studies)


def metabolights(cache, refresh):
    ids = (fetch(ML, cache, refresh=refresh) or {}).get("content", [])
    if not ids:
        raise SystemExit(f"metabolights: could not list studies ({FAILED[-1] if FAILED else 'empty'})")
    print(f"metabolights: {len(ids)} studies", flush=True)

    def one(sid):
        org = fetch(f"{ML}/{sid}/organisms", cache, refresh=refresh) or {}
        orgs = [o.get("Characteristics[Organism]", "") for o in org.get("organisms", []) or []]
        if not any("sapiens" in str(o) for o in orgs):
            return None
        title = (fetch(f"{ML}/{sid}/title", cache, refresh=refresh) or {}).get("title", "")
        desc = (fetch(f"{ML}/{sid}/description", cache, refresh=refresh) or {}).get("description", "")
        fac = (fetch(f"{ML}/{sid}/factors", cache, refresh=refresh) or {}).get("factors", []) or []
        return {"source": "metabolights", "id": sid, "title": title, "species": "; ".join(sorted(set(map(str, orgs)))),
                "n_samples": "", "license": "", "text": f"{title} {re.sub('<[^>]+>', ' ', desc)}",
                "factors": " | ".join(f.get("factorName", "") for f in fac),
                "url": f"https://www.ebi.ac.uk/metabolights/{sid}"}
    return [r for r in pmap(one, ids) if r]


def literature(cache, refresh, cap=3000):
    rows = {}
    for name, q in LITERATURE.items():
        cursor, n = "*", 0
        while cursor and n < cap:
            d = fetch(EPMC, cache, params={"query": q, "format": "json", "resultType": "core", "pageSize": 1000,
                                           "cursorMark": cursor}, refresh=refresh) or {}
            res = d.get("resultList", {}).get("result", [])
            for p in res:
                pid = p.get("pmid") or p.get("pmcid") or p.get("id")
                text = f"{p.get('title', '')} {p.get('abstractText', '')}"
                acc = sorted(set(re.findall(r"\b(MTBLS\d+|ST\d{6}|MSV\d{9})\b", text)))
                r = rows.setdefault(pid, {
                    "source": "literature", "id": f"PMID:{p['pmid']}" if p.get("pmid") else pid,
                    "title": p.get("title", ""), "species": "", "n_samples": "", "license": "",
                    "text": re.sub("<[^>]+>", " ", text), "factors": "",
                    "url": f"https://doi.org/{p['doi']}" if p.get("doi") else f"https://europepmc.org/article/{p.get('source')}/{p.get('id')}",
                    "year": p.get("pubYear", ""), "open_access": p.get("isOpenAccess", ""),
                    "has_supplement": p.get("hasSuppl", ""), "accessions": ";".join(acc), "found_by": set()})
                r["found_by"].add(name)
            n += len(res)
            nxt = d.get("nextCursorMark")
            cursor = nxt if res and nxt != cursor else None
        print(f"literature: {name}: {n} papers", flush=True)
    for r in rows.values():
        r["found_by"] = ";".join(sorted(r["found_by"]))
        # human-only screen on text: drop papers that are clearly animal or plant work
        r["species"] = "non-human?" if re.search(r"\b(mice|mouse|murine|rats?|zebrafish|drosophila|yeast|arabidopsis|"
                                                 r"elegans|plants?)\b", r["text"], re.I) and not re.search(
            r"\b(patients?|human|humans|individuals?|children|participants)\b", r["text"], re.I) else "human or unstated"
    return list(rows.values())


# ---- assemble -----------------------------------------------------------------------
def registered():
    reg = pd.read_csv(REGISTRY, sep="\t", dtype=str).fillna("")
    return set(reg.study), set(reg.loc[reg.status == "active", "study"])


def candidates(records, genes):
    symbols = set(genes.symbol.astype(str))
    in_net = dict(zip(genes.symbol, genes.in_network))
    reg_any, reg_active = registered()
    out = []
    for r in records:
        hits = find_genes(r["text"], symbols)
        design = ";".join(k for k, pat in DESIGN.items() if re.search(pat, f"{r['text']} {r['factors']}", re.I))
        for s in re.findall(r"(?<![A-Za-z0-9])(?:sh|si|sg)?([A-Z][A-Z0-9]{1,9})(?![a-z0-9])", r["factors"]):
            if s in symbols and (s not in STOP or len(s) > 1):
                hits.setdefault(s, set()).add("sample factor")
        for g, words in hits.items():
            cls = perturbation_class(words)
            reg = "active" if r["id"] in reg_active else ("registered, not active" if r["id"] in reg_any else "")
            score = 3 * bool(in_net.get(g)) + {"loss": 3, "chemical": 1, "gain": 0, "": 0}[cls] \
                + 2 * ("sample factor" in words) + (r["source"] != "literature") \
                + (r["source"] == "literature" and bool(r.get("accessions"))) - 2 * bool(design)
            snippet = ""
            m = re.search(rf"(?<![A-Za-z0-9])(?:sh|si|sg)?{re.escape(g)}(?![a-z0-9])", r["text"])
            if m:
                snippet = r["text"][max(0, m.start() - 90): m.end() + 90].replace("\n", " ").strip()
            out.append({"score": score, "source": r["source"], "id": r["id"], "gene": g,
                        "in_network": bool(in_net.get(g)), "perturbation": cls,
                        "evidence": ", ".join(sorted(words))[:120], "in_registry": reg, "design_flags": design,
                        "title": r["title"][:200], "species": r["species"], "n_samples": r["n_samples"],
                        "license": r["license"], "factors": r["factors"][:300], "snippet": snippet,
                        "year": r.get("year", ""), "open_access": r.get("open_access", ""),
                        "has_supplement": r.get("has_supplement", ""), "accessions": r.get("accessions", ""),
                        "found_by": r.get("found_by", ""), "url": r["url"]})
    df = pd.DataFrame(out)
    if len(df):
        df = df.sort_values(["score", "source", "id"], ascending=[False, True, True])
    return df


def report(df, counts, path):
    new = df[(df.in_registry == "") & df.in_network & (df.perturbation == "loss") & (df.species != "non-human?")
             & (df.design_flags == "")]
    L = ["# Candidate evaluation data", "",
         "Sources swept: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + ".", "",
         f"Rows (study or paper x gene): {len(df)}. Gene in the network, loss of function, not in the registry: "
         f"**{new.id.nunique()} studies or papers, {new.gene.nunique()} genes**.", "",
         "By source (not in registry, gene in network, loss of function):", ""]
    L += [f"- {s}: {g.id.nunique()} ids, {g.gene.nunique()} genes" for s, g in new.groupby("source")]
    L += ["", "## Top repository candidates", "",
          "| score | id | gene | evidence | n | title |", "|---|---|---|---|---|---|"]
    for r in new[new.source != "literature"].head(60).itertuples():
        L.append(f"| {r.score} | [{r.id}]({r.url}) | {r.gene} | {r.evidence[:50]} | {r.n_samples} | {r.title[:90]} |")
    L += ["", "## Top literature candidates (open access first)", "",
          "| score | id | gene | year | OA | suppl | accessions | title |", "|---|---|---|---|---|---|---|---|"]
    lit = new[new.source == "literature"].sort_values(["score", "open_access"], ascending=[False, False])
    for r in lit.head(60).itertuples():
        L.append(f"| {r.score} | [{r.id}]({r.url}) | {r.gene} | {r.year} | {r.open_access} | {r.has_supplement} | "
                 f"{r.accessions} | {r.title[:90]} |")
    L += ["", "Score: +3 gene in network, +3 loss of function (+1 chemical), +2 gene named in a sample factor, "
          "+1 repository study, +1 paper citing a deposit, -2 tracing or lipid-only design (these are left out of "
          "the tables above; see candidates.tsv). Gene matching is text based: check each hit.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", default=["workbench", "metabolights", "literature"],
                   choices=["workbench", "metabolights", "literature"])
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--refresh", action="store_true", help="refetch everything instead of using the cache")
    args = p.parse_args(argv)
    out = Path(args.out); cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    trust_windows_roots(cache)
    genes = gene_vocabulary(cache)
    records, counts = [], {}
    for s in args.sources:
        rec = {"workbench": workbench, "metabolights": metabolights, "literature": literature}[s](cache, args.refresh)
        counts[s] = len(rec)
        records += rec
    df = candidates(records, genes)
    df.to_csv(out / "candidates.tsv", sep="\t", index=False)
    report(df, counts, out / "report.md")
    if FAILED:
        (out / "failed_fetches.txt").write_text("\n".join(FAILED), encoding="utf-8")
    print(f"wrote {out} ({len(df)} rows, {len(FAILED)} failed fetches) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
