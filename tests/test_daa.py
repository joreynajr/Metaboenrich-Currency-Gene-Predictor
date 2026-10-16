import unittest
from types import SimpleNamespace

import pandas as pd

from metaboenrich.daa import load_daa, loose_key, name_variants


def stub_model(n):
    """A model with metabolites MAM00001..MAM0000n and no cross-references."""
    species = {f"MAM{i:05d}c": SimpleNamespace(met_id=f"MAM{i:05d}", id=f"MAM{i:05d}c", name=f"metabolite {i}", xrefs={})
               for i in range(1, n + 1)}
    return SimpleNamespace(species=species)


class NameMatchingTest(unittest.TestCase):
    def test_loose_key_unifies_common_variants(self):
        self.assertEqual(loose_key("O-acetylcarnitine"), loose_key("Acetylcarnitine"))
        self.assertEqual(loose_key("(R)-3-hydroxybutanoate"), loose_key("3-Hydroxybutyric acid"))
        self.assertEqual(loose_key("Methionine-Sulfoxide"), loose_key("methionine sulfoxide"))
        self.assertEqual(loose_key("L-Isoleucine"), loose_key("isoleucine"))

    def test_loose_key_keeps_distinct_compounds_apart(self):
        # cis/trans name different compounds
        self.assertNotEqual(loose_key("cis-4-hydroxy-L-proline"), loose_key("trans-4-hydroxy-L-proline"))
        # a lipid's parenthetical is part of its name, not an abbreviation
        self.assertNotEqual(loose_key("PI(TXB2/20:0)"), loose_key("Pi"))

    def test_name_variants_strip_trailing_abbreviation(self):
        self.assertEqual(name_variants("Phenyllactate (PLA)"), ["phenyllactate (pla)", "phenyllactate"])
        self.assertEqual(name_variants("glutamine*"), ["glutamine"])


class OrientationTest(unittest.TestCase):
    """D20: by default the significantly decreased metabolites are the sources."""
    daa = pd.DataFrame({"id": [f"MAM{i:05d}" for i in range(1, 6)],
                        "log2FC": [-2.0, 1.5, 0.0, -0.4, 3.0],
                        "p": [0.001, 0.001, 0.001, 0.5, 0.01]})

    def roles(self, **kw):
        df, _, _ = load_daa(self.daa, stub_model(5), "id", "log2FC", "p", 0.05, **kw)
        return dict(zip(df.met_id, df.role)), dict(zip(df.met_id, df.phi))

    def test_default_sources_are_decreased(self):
        roles, _ = self.roles()
        self.assertEqual(roles, {"MAM00001": "source", "MAM00002": "target", "MAM00003": "",
                                 "MAM00004": "", "MAM00005": "target"})

    def test_increased_restores_protocol_orientation(self):
        roles, _ = self.roles(sources="increased")
        self.assertEqual(roles, {"MAM00001": "target", "MAM00002": "source", "MAM00003": "",
                                 "MAM00004": "", "MAM00005": "source"})

    def test_phi_does_not_depend_on_orientation(self):
        _, phi_dec = self.roles(sources="decreased")
        _, phi_inc = self.roles(sources="increased")
        self.assertEqual(phi_dec, phi_inc)
        self.assertEqual(phi_dec["MAM00001"], 2.0)

    def test_unknown_orientation_is_rejected(self):
        with self.assertRaises(ValueError):
            self.roles(sources="up")


if __name__ == "__main__":
    unittest.main()
