import unittest

from metaboenrich.daa import loose_key, name_variants


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


if __name__ == "__main__":
    unittest.main()
