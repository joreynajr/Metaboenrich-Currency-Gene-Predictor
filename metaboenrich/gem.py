"""Parse a genome-scale metabolic model (SBML L3 + FBC v2, e.g. Human-GEM.xml).

Only the pieces the current-flow algorithm needs are kept: metabolites (with
cross-references for mapping DAA tables), reactions (participants,
reversibility) and gene-reaction associations.
"""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

HUMAN_GEM_VERSION = "v2.0.1"
HUMAN_GEM_URL = f"https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/{HUMAN_GEM_VERSION}/model/Human-GEM.xml"
HUMAN_GEM_SHA256 = "6ce49b620391f0ad76be24fdbdfc884fa376ff534dd3813c95abbe9d8c66fa5e"
# The SBML release carries no subsystem (pathway) assignments; the YAML of the
# same release does.
HUMAN_GEM_YML_URL = f"https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/{HUMAN_GEM_VERSION}/model/Human-GEM.yml"
HUMAN_GEM_YML_SHA256 = "3b944902a44f5f0e9dfcf23f8b9b4a54891bfb85ae711f8540e72ffbfcfa1f9a"


def load_subsystems(path):
    """{SBML reaction id (R_MAR...): [subsystem, ...]} from Human-GEM.yml.
    Downloads the pinned release to `path` if it is missing."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Human-GEM {HUMAN_GEM_VERSION} YAML (subsystems) to {path} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(HUMAN_GEM_YML_URL, tmp)
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != HUMAN_GEM_YML_SHA256:
            tmp.unlink()
            raise SystemExit("Downloaded Human-GEM.yml has an unexpected SHA-256")
        tmp.replace(path)
    subsystems, rid, in_sub = {}, None, False
    in_reactions = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("- reactions:"):
            in_reactions = True
            continue
        if in_reactions and line.startswith("- ") and not line.startswith("- reactions"):
            break                                     # next top-level section (genes, compartments)
        if not in_reactions:
            continue
        s = line.strip()
        if s.startswith("- id: "):
            rid, in_sub = "R_" + s[6:].strip().strip('"'), False
        elif s == "- subsystem:":
            in_sub = True
        elif s.startswith("- subsystem: "):          # single value on one line
            subsystems.setdefault(rid, []).append(s[13:].strip().strip('"'))
            in_sub = False
        elif in_sub and line.startswith("      - "):
            subsystems.setdefault(rid, []).append(s[2:].strip().strip('"'))
        elif in_sub:
            in_sub = False
    return subsystems

NS = {
    "sbml": "http://www.sbml.org/sbml/level3/version1/core",
    "fbc": "http://www.sbml.org/sbml/level3/version1/fbc/version2",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "bqbiol": "http://biomodels.net/biology-qualifiers/",
}
FBC = "{%s}" % NS["fbc"]
RDF = "{%s}" % NS["rdf"]

# identifiers.org namespaces worth keeping for mapping user metabolite IDs.
XREF_PREFIXES = {
    "hmdb": "hmdb",
    "kegg.compound": "kegg",
    "chebi": "chebi",
    "bigg.metabolite": "bigg",
    "metanetx.chemical": "metanetx",
}


@dataclass
class Species:
    id: str            # SBML species id, e.g. M_MAM01570c
    met_id: str        # compartment-free id, e.g. MAM01570
    name: str
    compartment: str
    xrefs: dict = field(default_factory=dict)   # {"hmdb": [...], "kegg": [...], ...}
    formula: str = ""


@dataclass
class Reaction:
    id: str
    name: str
    substrates: dict   # species id -> stoichiometry
    products: dict
    forward: bool      # flux may be > 0
    backward: bool     # flux may be < 0
    genes: list        # Ensembl ids (without the G_ prefix)
    is_objective: bool = False


@dataclass
class Model:
    id: str
    version: str
    species: dict      # species id -> Species
    reactions: dict    # reaction id -> Reaction
    genes: dict        # Ensembl id -> gene symbol


def download_human_gem(path):
    """Fetch the pinned Human-GEM SBML release to `path` and verify its hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Human-GEM {HUMAN_GEM_VERSION} to {path} ...")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(HUMAN_GEM_URL, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != HUMAN_GEM_SHA256:
        tmp.unlink()
        raise SystemExit(f"Downloaded Human-GEM has unexpected SHA-256 {digest}")
    tmp.replace(path)


def _strip(prefix, s):
    return s[len(prefix):] if s.startswith(prefix) else s


def _met_id(species_id, compartment):
    base = _strip("M_", species_id)
    if compartment and base.endswith(compartment):
        base = base[: -len(compartment)]
    return base


def _xrefs(species_el):
    refs = {}
    for li in species_el.iterfind(".//bqbiol:is/rdf:Bag/rdf:li", NS):
        uri = li.get(RDF + "resource", "")
        if "identifiers.org/" not in uri:
            continue
        ns, _, value = uri.split("identifiers.org/", 1)[1].partition("/")
        key = XREF_PREFIXES.get(ns)
        if key:
            refs.setdefault(key, []).append(value)
    return refs


def load_sbml(path):
    root = ET.parse(path).getroot()
    model_el = root.find("sbml:model", NS)

    version = ""
    for p in model_el.iterfind("sbml:notes//{http://www.w3.org/1999/xhtml}p", NS):
        if (p.text or "").startswith("version:"):
            version = p.text.split(":", 1)[1].strip()

    params = {
        p.get("id"): float(p.get("value"))
        for p in model_el.iterfind("sbml:listOfParameters/sbml:parameter", NS)
    }

    species = {}
    for el in model_el.iterfind("sbml:listOfSpecies/sbml:species", NS):
        sid, comp = el.get("id"), el.get("compartment")
        species[sid] = Species(sid, _met_id(sid, comp), el.get("name", ""), comp, _xrefs(el),
                               el.get(FBC + "chemicalFormula", ""))

    genes = {}
    for el in model_el.iterfind("fbc:listOfGeneProducts/fbc:geneProduct", NS):
        genes[_strip("G_", el.get(FBC + "id"))] = el.get(FBC + "name", "")

    objectives = {
        el.get(FBC + "reaction")
        for el in model_el.iterfind(".//fbc:fluxObjective", NS)
    }

    reactions = {}
    for el in model_el.iterfind("sbml:listOfReactions/sbml:reaction", NS):
        rid = el.get("id")
        lb = params.get(el.get(FBC + "lowerFluxBound"), -1000.0)
        ub = params.get(el.get(FBC + "upperFluxBound"), 1000.0)

        def refs(tag):
            return {
                sr.get("species"): float(sr.get("stoichiometry", 1))
                for sr in el.iterfind(f"sbml:{tag}/sbml:speciesReference", NS)
            }

        gene_ids = sorted({
            _strip("G_", g.get(FBC + "geneProduct"))
            for g in el.iterfind(".//fbc:geneProductRef", NS)
        })
        reactions[rid] = Reaction(
            id=rid,
            name=el.get("name", ""),
            substrates=refs("listOfReactants"),
            products=refs("listOfProducts"),
            forward=ub > 0,
            backward=lb < 0,
            genes=gene_ids,
            is_objective=rid in objectives,
        )

    return Model(model_el.get("id"), version, species, reactions, genes)
