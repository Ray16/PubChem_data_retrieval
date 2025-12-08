#!/usr/bin/env python3
"""
Generate chemistry Q&A pairs from PubChem for LLM training.
Outputs:
  - molecular_properties.csv
  - chemistry_qa_dataset.json
"""

import time
import json
import re
import random
from typing import List, Dict, Optional, Tuple, Any, Set
import pandas as pd
import pubchempy as pcp
import numpy as np

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Optional RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import Descriptors
    from rdkit.Chem.rdFMCS import FindMCS
    rdkit_available = True
except ImportError:
    rdkit_available = False
    print("Warning: RDKit not found. Advanced features (Rings, MCS) will be skipped.")

# ---------- Functional group hierarchy ----------
FUNCTIONAL_GROUP_HIERARCHY = {
    'ester': ['ether'], 'lactone': ['ether', 'ester'], 'carboxylic_acid': ['ketone'],
    'carboxylate': ['ketone'], 'primary_amide': ['ketone', 'primary_amine'], 'secondary_amide': ['ketone', 'secondary_amine'],
    'tertiary_amide': ['ketone', 'tertiary_amine'], 'acyl_chloride': ['ketone'], 'anhydride': ['ketone', 'ester'],
    'aldehyde': ['ketone'], 'thioester': ['thioketone'], 'carbamate': ['ester', 'ether'],
    'carbonate': ['ester', 'ether'],
    'urea': ['primary_amide', 'secondary_amide', 'tertiary_amide'],
    'sulfonamide': ['sulfone', 'primary_amine', 'secondary_amine', 'tertiary_amine'], 'sulfonic_acid': ['sulfone'], 'hemiacetal': ['ether', 'secondary_alcohol'],
    'acetal': ['ether'], 'phosphate': ['ester', 'ether'], 'phosphonate': ['ester', 'ether'],
}

# Aliases for counting coarse-grained groups like "alcohol" -> sum of primary/secondary/tertiary
FG_ALIASES = {
    'alcohol': ['primary_alcohol', 'secondary_alcohol', 'tertiary_alcohol'],
    'amine': ['primary_amine', 'secondary_amine', 'tertiary_amine'],
}

# ---------- Helper: filter subgroups using counts and atom overlap ----------
def filter_subgroups_by_location_counts(mol, fg_counts: Dict[str, int], fg_smarts: Dict[str, str]) -> Dict[str, int]:
    """
    Given a mol and a dict of counts per functional group (fg_counts),
    return a filtered dict where subgroup matches that overlap strongly
    with parent functional groups are decremented/removed.
    """
    if not fg_counts or mol is None:
        return fg_counts

    group_atoms: Dict[str, List[Set[int]]] = {}

    # Collect atom index sets for each group
    for group, count in fg_counts.items():
        if count == 0 or group not in fg_smarts:
            continue
        try:
            patt = Chem.MolFromSmarts(fg_smarts[group])
            if patt:
                group_atoms[group] = [set(m) for m in mol.GetSubstructMatches(patt)]
        except Exception:
            # ignore SMARTS parse failures
            continue

    # For each parent->subgroup relation, remove subgroup matches that are
    # clearly inside parent matches (to avoid double counting)
    for parent, subgroups in FUNCTIONAL_GROUP_HIERARCHY.items():
        if parent not in group_atoms:
            continue
        parent_atoms = group_atoms[parent]

        for subgroup in subgroups:
            if subgroup not in group_atoms:
                continue

            keep = []
            for sub_atoms in group_atoms[subgroup]:
                overlaps = any(
                    sub_atoms.issubset(p) or len(sub_atoms & p) >= 0.5 * len(sub_atoms)
                    for p in parent_atoms
                )
                # keep only those subgroup matches that do NOT overly overlap with parent
                if not overlaps:
                    keep.append(sub_atoms)

            group_atoms[subgroup] = keep
            fg_counts[subgroup] = len(keep)

    # Return only non-zero counts
    return {k: v for k, v in fg_counts.items() if v > 0}

# ---------- Utility: RDKit-based derived features ----------
def compute_rdkit_features(smiles: str) -> Dict[str, Any]:
    """
    Compute RDKit-derived features. Outputs:
      - murcko_scaffold (smiles or None)
      - functional_groups (dict mapping group_name -> count) or None
      - rdkit_mol (Mol) or None
      - num_rings (int) or None
      - degree_of_unsaturation (int) or None
    """
    features = {
        'murcko_scaffold': None,
        'functional_groups': None,
        'rdkit_mol': None,
        'num_rings': None,
        'degree_of_unsaturation': None,
        'rdkit_atom_counts': None
    }

    if not rdkit_available or not smiles:
        return features

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return features

        # Murcko Scaffold
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold and scaffold.GetNumAtoms() > 0:
                features['murcko_scaffold'] = Chem.MolToSmiles(scaffold, isomericSmiles=False)
        except Exception:
            features['murcko_scaffold'] = None

        # Functional group SMARTS dictionary
        fg_smarts = {
            'primary_amine': '[#7X3;H2;!$(NC=O);!$(NS=O)]',
            'secondary_amine': '[#7X3;H1;!$(NC=O);!$(NS=O)]([#6])[#6]',
            'tertiary_amine': '[#7X3;H0;!$(NC=O);!$(NS=O)]([#6])([#6])[#6]',
            'quaternary_ammonium': '[#7+;H0]([#6])([#6])([#6])[#6]',
            'primary_amide': '[NX3;H2][CX3](=O)',
            'secondary_amide': '[NX3;H1][CX3](=O)',
            'tertiary_amide': '[NX3;H0][CX3](=O)',
            'urea': '[#7X3;!@R][#6X3](=O)[#7X3;!@R]',
            'carbamate': '[#7X3][#6X3](=O)[OX2]',
            'imine': '[NX2]=[CX3]', 'nitrile': '[N+](=O)[O-]',
            'nitro': '[$([#7X3](=O)=O),$([#7X3+](=O)[O-])][!#8]',
            'nitroso': '[#7X2](=O)[#6]', 'azide': '[#7X2]=[#7X2+]=[#7X1-]',
            'primary_alcohol': '[#6X4][OX2H]', 'secondary_alcohol': '[#6X4H]([#6])[OX2H]',
            'tertiary_alcohol': '[#6X4]([#6])([#6])([#6])[OX2H]', 'phenol': 'c[OX2H]',
            'carboxylic_acid': '[#6X3](=O)[OX2H1]', 'carboxylate': '[#6X3](=O)[OX1-,OX2-]',
            'ester': '[#6X3](=O)[OX2][#6;!$(C=O)]', 'lactone': '[OX2r][CX3r](=O)',
            'ketone': '[CX3](=O)[#6]', 'aldehyde': '[CX3H1](=O)',
            'acyl_chloride': '[#6X3](=O)[Cl]',
            'anhydride': '[#6X3](=O)[OX2][#6X3](=O)', 'ether': '[OX2]([#6])[#6;!$(C=O)]',
            'thiol': '[SX2H]', 'sulfide': '[SX2]([#6])[#6]', 'disulfide': '[SX2][SX2]',
            'sulfoxide': '[SX3](=O)([#6])[#6]', 'sulfone': '[SX4](=O)(=O)([#6])[#6]',
            'sulfonamide': '[SX4](=O)(=O)[#7X3]',
            'phosphate': '[PX4](=O)([OX2-,OX2H])([OX2-,OX2H])[OX2-,OX2H]',
            'fluoride': '[FX1]', 'chloride': '[ClX1]', 'bromide': '[BrX1]', 'iodide': '[IX1]',
            'benzene': 'c1ccccc1', 'pyridine': 'n1ccccc1',
            'pyrrole': '[nH]1cccc1', 'furan': 'o1cccc1', 'thiophene': 's1cccc1',
            'imidazole': 'n1cc[nH]c1',
            'alkene': '[CX3]=[CX3]', 'alkyne': '[CX2]#[CX2]',
            'guanidine': '[NX3][CX3](=[NX3])[NX3]',
            'amidine': '[NX3]=[CX3][NX3]',
            'hemiacetal':'[CX4H0]([OX2H])[OX2][#6]',
            'acetal':'[CX4H0]([OX2][#6])([OX2][#6])'
        }

        # Count each FG occurrence
        fg_counts = {}
        for name, smarts in fg_smarts.items():
            try:
                patt = Chem.MolFromSmarts(smarts)
                if patt:
                    fg_counts[name] = len(mol.GetSubstructMatches(patt))
                else:
                    fg_counts[name] = 0
            except Exception:
                fg_counts[name] = 0

        # Filter counts to avoid double counting (esters vs ethers, etc.)
        fg_counts = filter_subgroups_by_location_counts(mol, fg_counts, fg_smarts)
        features['functional_groups'] = fg_counts if fg_counts else None

        # Number of Rings
        ring_info = mol.GetRingInfo()
        features['num_rings'] = ring_info.NumRings()

        # Degree of Unsaturation (DOU)
        mol_formula_str = Chem.rdMolDescriptors.CalcMolFormula(mol)
        atom_counts = parse_formula(mol_formula_str) 
        features['rdkit_atom_counts'] = atom_counts # Store this reliable count

        # 2. Re-calculate DOU using this RDKit-derived count
        features['degree_of_unsaturation'] = calculate_dou(atom_counts)

    except Exception:
        # Fail silently but return what we have
        pass

    return features

# ---------- Core Utility Functions ----------
def parse_formula(formula: str) -> Dict[str, int]:
    atom_counts = {}
    if not formula:
        return atom_counts
    for match in re.finditer(r'([A-Z][a-z]?)(\d*)', formula):
        atom, count_str = match.groups()
        atom_counts[atom] = int(count_str) if count_str else 1
    return atom_counts

def calculate_dou(atom_counts: Dict[str, int]) -> int:
    C = atom_counts.get('C', 0)
    H = atom_counts.get('H', 0)
    N = atom_counts.get('N', 0)
    X = sum(atom_counts.get(x, 0) for x in ['F', 'Cl', 'Br', 'I'])
    return (2*C + 2 + N - H - X) // 2

def compute_mcs_info(smiles_a: str, smiles_b: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Returns (smarts_string, num_atoms) for the MCS between smiles_a and smiles_b.
    If RDKit not available or MCS fails, returns (None, None).
    """
    if not rdkit_available:
        return None, None
    try:
        mol1 = Chem.MolFromSmiles(smiles_a)
        mol2 = Chem.MolFromSmiles(smiles_b)
        if mol1 is None or mol2 is None:
            return None, None
        mcs_result = FindMCS([mol1, mol2])
        if mcs_result is None or mcs_result.numAtoms == 0:
            return None, None
        return mcs_result.smartsString, int(mcs_result.numAtoms)
    except Exception:
        return None, None

# ---------- Main generator class ----------
class PubChemQADataset:
    def __init__(self, output_prefix: str = "chemistry_qa_dataset"):
        self.output_prefix = output_prefix
        self.csv_file = f"molecular_properties.csv"
        self.qa_file = f"{output_prefix}.json"
        self.df = pd.DataFrame()
        self.failed_cids: List[Tuple[int, str]] = []
        self.fg_smarts_list = [
            'primary_alcohol', 'secondary_alcohol', 'phenol', 'carboxylic_acid',
            'ketone', 'aldehyde', 'ester', 'ether', 'alkene', 'alkyne',
            'primary_amine', 'secondary_amine', 'primary_amide', 'sulfone',
            'thiol', 'nitrile', 'carbamate', 'urea', 'guanidine' # Note: full list is longer in RDKit
        ]
        # local RNG for pandas sampling reproducibility
        self.pandas_random_state = SEED

    def fetch_compound_basic(self, cid: int, retries: int = 3, backoff: float = 1.5) -> Optional[Dict[str, Any]]:
        """
        Fetch basic PubChem info with simple retry/backoff.
        """
        attempt = 0
        while attempt < retries:
            try:
                compound = pcp.Compound.from_cid(cid)
                if compound is None:
                    return None

                # choose best available smiles field
                canonical_smiles = getattr(compound, 'connectivity_smiles', None) or getattr(compound, 'isomeric_smiles', None) or getattr(compound, 'canonical_smiles', None)

                name = compound.iupac_name or f"Compound {cid}"
                data = {
                    'cid': cid,
                    'name': name,
                    'canonical_smiles': canonical_smiles,
                    'molecular_formula': getattr(compound, 'molecular_formula', None),
                    'molecular_weight': getattr(compound, 'molecular_weight', None),
                    'h_bond_donors': getattr(compound, 'h_bond_donor_count', None),
                    'h_bond_acceptors': getattr(compound, 'h_bond_acceptor_count', None),
                    'rotatable_bonds': getattr(compound, 'rotatable_bond_count', None),
                    'topological_psa': getattr(compound, 'tpsa', None),
                    'logp': getattr(compound, 'xlogp', None),
                    'heavy_atom_count': getattr(compound, 'heavy_atom_count', None),
                }
                return data
            except Exception as e:
                attempt += 1
                time.sleep(backoff * attempt)
        # record failure
        self.failed_cids.append((cid, f"fetch_failed_after_{retries}_attempts"))
        return None

    def build_dataframe(self, cid_list: List[int], use_rdkit_if_available: bool = True) -> None:
        rows = []
        total = len(cid_list)
        print(f"Fetching {total} CIDs (this may take a while)...")
        for i, cid in enumerate(cid_list, start=1):
            print(f"[{i}/{total}] CID {cid} ...", end='')
            raw = self.fetch_compound_basic(cid)
            if raw is None:
                print(" failed")
                continue

            smiles_for_rdkit = raw.get('canonical_smiles')
            rdkit_feats = compute_rdkit_features(smiles_for_rdkit) if use_rdkit_if_available else {}

            mw, hd, ha, logp = raw.get('molecular_weight'), raw.get('h_bond_donors'), raw.get('h_bond_acceptors'), raw.get('logp')
            tpsa, rb = raw.get('topological_psa'), raw.get('rotatable_bonds')

            lipinski_pass = None
            if None not in (mw, hd, ha, logp):
                lipinski_pass = (mw <= 500 and hd <= 5 and ha <= 10 and logp <= 5)

            veber_pass = None
            if None not in (tpsa, rb):
                veber_pass = (rb <= 10 and tpsa <= 140)

            row = {
                **raw,
                **rdkit_feats,
                'lipinski_pass': lipinski_pass,
                'veber_pass': veber_pass,
            }

            rows.append(row)
            print(" done")
            # polite pause (still deterministic because random seeds are set)
            time.sleep(0.3)

        self.df = pd.DataFrame(rows)
        # ensure cid is first column
        cols = list(self.df.columns)
        if 'cid' in cols:
            cols.insert(0, cols.pop(cols.index('cid')))
        self.df = self.df.reindex(columns=cols)

    # ---------- Comprehensive Q&A generation ----------
    def qa_from_dataframe(self):
        qa_list = []

        # Define Acidic and Basic Hierarchies (pKa of the acidic proton or the conjugate acid)
        # Strongest Acid (lowest pKa) comes first
        ACIDIC_HIERARCHY = [
            ('sulfonic_acid', -1.0), ('carboxylic_acid', 4.5), 
            # Note: Active methylene/imide SMARTS are complex, relying on counting 'ketone' or 'primary_amide' might be misleading without specific patterns.
            ('phenol', 10.0), 
            ('thiol', 10.5), 
            ('pyrrole', 17.0), # Representing aromatic N-H acids (e.g., indole, pyrrole)
            ('primary_alcohol', 17.0), ('secondary_alcohol', 17.0), ('tertiary_alcohol', 17.0)
        ]
        
        # Strongest Base (highest pKa of conjugate acid) comes first
        BASIC_HIERARCHY = [
            ('guanidine', 13.6),
            ('amidine', 12.0),
            ('tertiary_amine', 10.0), ('secondary_amine', 10.0), ('primary_amine', 9.5),
            ('imidazole', 7.0), 
            ('pyridine', 5.2),
            ('ketone', -7.0), ('aldehyde', -7.0), ('ether', -3.5), # Oxygen bases (weak)
        ]

        # These are the coarse-grained groups we will ask about; mapping to aliases where needed
        fine_grained_groups = [
        'primary_alcohol', 'secondary_alcohol', 'tertiary_alcohol',
        'primary_amine', 'secondary_amine', 'tertiary_amine',
        'ketone', 'aldehyde', 'ester', 'ether', 'phenol', 'nitrile'
        ]
        countable_atoms = ['Cl', 'Br', 'F', 'I', 'N', 'O', 'S', 'P', 'C', 'H']
        all_possible_groups = self.fg_smarts_list

        for idx, row in self.df.iterrows():
            name = row.get('name', f'compound_{idx}')
            fg = row.get('functional_groups', {}) or {}
            formula = row.get('molecular_formula', '')
            mol = row.get('rdkit_mol', None)
            cid = row.get('cid')
            canonical_smiles = row.get('canonical_smiles')

            atom_counts = row.get('rdkit_atom_counts', {}) # Fetched from the new field
            if not atom_counts:
                formula = row.get('molecular_formula', '')
                atom_counts = parse_formula(formula) if formula else {}

            # --- Counting Questions: prefer non-zero groups ---
            # build list of non-zero coarse groups using aliases
            fg_nonzero = []
            for g in fine_grained_groups:
                if g in FG_ALIASES:
                    total = sum(fg.get(sub, 0) for sub in FG_ALIASES[g])
                else:
                    total = fg.get(g, 0)
                if total > 0:
                    fg_nonzero.append(g)

            # Atom counting questions (choose atoms with non-zero count if possible)
            atom_nonzero = [a for a in countable_atoms if atom_counts.get(a, 0) > 0]
            try:
                if len(atom_nonzero) >= 2:
                    atoms_to_count = random.sample(atom_nonzero, k=2)
                else:
                    atoms_to_count = random.sample(countable_atoms, k=min(2, len(countable_atoms)))
            except ValueError:
                atoms_to_count = countable_atoms[:2]

            for atom in atoms_to_count:
                atom_count = atom_counts.get(atom, 0)
                qa_list.append({
                    'question': f"How many {atom} atoms does {name} have?",
                    'answer': str(atom_count),
                    'cid': cid,
                    'answer_source': 'molecular_formula (parsed)'
                })

            # --- Boolean Functional Group Queries (short answers) ---
            try:
                queried_groups = random.sample(all_possible_groups, k=min(3, len(all_possible_groups)))
            except ValueError:
                queried_groups = all_possible_groups[:3]

            for group in queried_groups:
                has_group = fg.get(group, 0) > 0
                qa_list.append({
                    'question': f"Does {name} contain a {group.replace('_', ' ')} group?",
                    'answer': 'Yes' if has_group else 'No',
                    'cid': cid,
                    'answer_source': f'functional_groups ({group} boolean)'
                })

            # --- Acid/Base Reasoning ---
            
            # 1. Most Acidic Group (Strongest Acid/Lowest pKa)
            most_acidic = next((g for g, _ in ACIDIC_HIERARCHY if fg.get(g,0) > 0), None)
            if most_acidic:
                qa_list.append({
                    'question': f"Based on common functional groups, which is the most acidic group in {name}?",
                    'answer': most_acidic.replace('_',' '),
                    'cid': cid,
                    'answer_source': 'acid_base_hierarchy_acid'
                })
            
            # 2. Most Basic Group (Strongest Base/Highest pKa of Conjugate Acid)
            most_basic = next((g for g, _ in BASIC_HIERARCHY if fg.get(g,0) > 0), None)
            if most_basic:
                qa_list.append({
                    'question': f"Based on common functional groups, which is the most basic group in {name}?",
                    'answer': most_basic.replace('_',' '),
                    'cid': cid,
                    'answer_source': 'acid_base_hierarchy_base'
                })


            if most_basic:
                qa_list.append({
                    'question': f"Based on common functional groups, which is the most basic group in {name}?",
                    'answer': most_basic.replace('_', ' '),
                    'cid': cid,
                    'answer_source': 'acid_base_hierarchy_base'
                })

            # --- Formula Analysis (keep answers short and meaningful) ---
            # Nitrogen count
            if formula:
                qa_list.append({'question': f"How many nitrogen atoms are in {name}?", 'answer': str(atom_counts.get('N', 0)), 'cid': cid, 'answer_source': 'molecular_formula (parsed)'})

                # Carbon to hydrogen ratio only when both > 0
                C = atom_counts.get('C', 0)
                H = atom_counts.get('H', 0)
                if C > 0 and H > 0:
                    qa_list.append({'question': f"What is the carbon to hydrogen ratio in {name}?", 'answer': f"{C}:{H}", 'cid': cid, 'answer_source': 'molecular_formula (parsed)'})

                dou = calculate_dou(atom_counts)
                qa_list.append({'question': f"What is the degree of unsaturation of {name}?", 'answer': str(dou), 'cid': cid, 'answer_source': 'molecular_formula (calculated)'})

            # --- Structural Feature Presence (RDKit) ---
            if mol and rdkit_available:
                ring_info = mol.GetRingInfo()
                n_rings = ring_info.NumRings()
                qa_list.append({'question': f"How many rings does {name} contain?", 'answer': str(n_rings), 'cid': cid, 'answer_source': 'rdkit ring_count'})

                if n_rings > 0:
                    ring_sizes = [len(r) for r in ring_info.AtomRings()]
                    qa_list.append({'question': f"What is the number of heavy atoms in the largest ring cycle (the ring itself) of {name}?", 'answer': str(max(ring_sizes)), 'cid': cid, 'answer_source': 'rdkit ring_sizes'})

                has_5 = any(len(r) == 5 for r in ring_info.AtomRings())
                qa_list.append({'question': f"Does {name} contain a five-membered ring?", 'answer': 'Yes' if has_5 else 'No', 'cid': cid, 'answer_source': 'rdkit ring_size boolean'})

                # New: Is the molecule polycyclic?
                qa_list.append({'question': f"Is {name} polycyclic (more than one ring)?", 'answer': 'Yes' if n_rings > 1 else 'No', 'cid': cid, 'answer_source': 'rdkit polycyclic boolean'})

            # --- Drug-Likeness Comparison (short answer) ---
            if row.get('lipinski_pass') is not None and row.get('veber_pass') is not None:
                mw, hd, ha, logp = row['molecular_weight'], row['h_bond_donors'], row['h_bond_acceptors'], row['logp']
                tpsa, rb = row['topological_psa'], row['rotatable_bonds']

                lipinski_fail = 4 - sum([mw <= 500, hd <= 5, ha <= 10, logp <= 5])
                veber_fail = 2 - sum([rb <= 10, tpsa <= 140])

                comparison_answer = 'Lipinski' if lipinski_fail > veber_fail else \
                                    'Veber' if veber_fail > lipinski_fail else 'Neither'

                qa_list.append({'question': f"Which set does {name} deviate from more: Lipinski or Veber?", 'answer': comparison_answer, 'cid': cid, 'answer_source': 'multi_rule_comparison'})

            # --- Short SMILES-based QA (not exposing long SMILES strings) ---
            if canonical_smiles:
                qa_list.append({'question': f"What is the length of the canonical SMILES for {name}?", 'answer': str(len(canonical_smiles)), 'cid': cid, 'answer_source': 'iupac_to_smiles_length'})
            # ---------- Advanced Functional Group Reasoning Questions ----------
            # Assumes `fg` dict, `name` string, and `cid` available
            # 1. Dominant Functional Group
            if fg_nonzero:
                max_count = max(fg.get(g,0) for g in fg_nonzero)
                dominant_fgs = [g for g in fg_nonzero if fg.get(g,0) == max_count]
                qa_list.append({
                    'question': f"What is the most abundant functional group in {name}?",
                    'answer': ', '.join([g.replace('_',' ') for g in dominant_fgs]),
                    'cid': cid,
                    'answer_source': 'most_abundant_functional_group_with_ties'
                })


            # 2. Functional Group Polarity (Polar vs Nonpolar)
            polar_fgs = {'alcohol','amine','amide','carboxylic_acid','carbamate','nitrile','phenol','urea','sulfonamide'}
            nonpolar_fgs = {'alkene','alkyne','ether','ester','ketone','thiol','sulfide','benzene'}
            polar_count = sum(fg.get(g,0) for g in polar_fgs)
            nonpolar_count = sum(fg.get(g,0) for g in nonpolar_fgs)
            if polar_count > 0 or nonpolar_count > 0:
                polarity_answer = 'polar' if polar_count >= nonpolar_count else 'nonpolar'
                qa_list.append({
                    'question': f"Is the most abundant functional group in {name} polar or nonpolar?",
                    'answer': polarity_answer,
                    'cid': cid,
                    'answer_source': 'functional_group_polarity'
                })

            # 3. Hydrolyzable Functional Groups
            hydrolyzable_fgs = {'ester','amide','carbamate','urea','anhydride','lactone'}
            has_hydrolyzable = any(fg.get(g,0) > 0 for g in hydrolyzable_fgs)
            qa_list.append({
                'question': f"Does {name} contain a hydrolyzable functional group?",
                'answer': 'Yes' if has_hydrolyzable else 'No',
                'cid': cid,
                'answer_source': 'hydrolyzable_functional_groups'
            })

            # 4. Mutual Exclusivity Check: Aldehyde vs Ketone
            has_aldehyde = fg.get('aldehyde',0) > 0
            has_ketone = fg.get('ketone',0) > 0
            qa_list.append({
                'question': f"Does {name} contain both an aldehyde and a ketone group?",
                'answer': 'Yes' if has_aldehyde and has_ketone else 'No',
                'cid': cid,
                'answer_source': 'aldehyde_ketone_exclusivity'
            })

            # 5. Carbonyl Functional Group Type
            carbonyl_fgs = ['aldehyde','ketone','ester','amide','carboxylic_acid','anhydride','lactone','carbamate']
            carbonyl_present = [g for g in carbonyl_fgs if fg.get(g,0) > 0]
            if carbonyl_present:
                qa_list.append({
                    'question': f"(Which carbonyl functional group(s) is present in {name}?",
                    'answer': ' and '.join([g.replace('_',' ') for g in carbonyl_present]),
                    'cid': cid,
                    'answer_source': 'carbonyl_type_presence'
                })

            # 6. Functional Group Diversity
            distinct_fg_count = sum(1 for v in fg.values() if v > 0)
            qa_list.append({
                'question': f"How many distinct functional group types are present in {name}?",
                'answer': str(distinct_fg_count),
                'cid': cid,
                'answer_source': 'functional_group_diversity'
            })

            # 7. Comparative Reasoning: Alcohol vs Amine
            alcohol_count = sum(fg.get(g,0) for g in ['primary_alcohol','secondary_alcohol','tertiary_alcohol'])
            amine_count = sum(fg.get(g,0) for g in ['primary_amine','secondary_amine','tertiary_amine'])
            if alcohol_count != amine_count:
                more_group = 'alcohol' if alcohol_count > amine_count else 'amine'
                qa_list.append({
                    'question': f"Which functional group is more numerous in {name}, alcohol or amine?",
                    'answer': more_group,
                    'cid': cid,
                    'answer_source': 'alcohol_vs_amine_count'
                })
            else:
                qa_list.append({
                    'question': f"Which functional group is more numerous in {name}, alcohol or amine?",
                    'answer': 'Same',
                    'cid': cid,
                    'answer_source': 'alcohol_vs_amine_count'
                })

            # 8. Nitrogen vs Oxygen Dominance
            N_count = sum(fg.get(g,0) for g in fg if 'amine' in g or 'amide' in g or 'nitrile' in g or 'guanidine' in g or 'imidazole' in g)
            O_count = sum(fg.get(g,0) for g in fg if 'alcohol' in g or 'ether' in g or 'ester' in g or 'carbonyl' in g or 'carboxylic' in g or 'phenol' in g)
            if N_count > O_count:
                dominant_element_group = 'nitrogen-containing'
            elif O_count > N_count:
                dominant_element_group = 'oxygen-containing'
            else:
                dominant_element_group = 'neither dominates'
            qa_list.append({
                'question': f"Is {name} primarily a nitrogen-containing or oxygen-containing compound?",
                'answer': dominant_element_group,
                'cid': cid,
                'answer_source': 'element_based_functional_group_dominance'
            })


        # --- Pairwise (MCS-derived) QA: only short/numeric/yes-no answers ---
        if len(self.df) >= 2 and rdkit_available:
            n_pairs = min(5, len(self.df) // 2)
            for i in range(n_pairs):
                # reproducible sampling
                pair_df = self.df.sample(n=2, replace=False, random_state=self.pandas_random_state + i)
                smiles_a = pair_df.iloc[0]['canonical_smiles']
                smiles_b = pair_df.iloc[1]['canonical_smiles']
                name_a, name_b = pair_df.iloc[0]['name'], pair_df.iloc[1]['name']
                cid_a, cid_b = pair_df.iloc[0]['cid'], pair_df.iloc[1]['cid']

                if smiles_a and smiles_b:
                    smarts, mcs_atoms = compute_mcs_info(smiles_a, smiles_b)
                    if mcs_atoms:
                        qa_list.append({
                            'question': f"How many atoms are in the maximum common substructure of {name_a} and {name_b}?",
                            'answer': str(mcs_atoms),
                            'cid': f'{cid_a}|{cid_b}',
                            'answer_source': 'rdkit:mcs_num_atoms'
                        })
                        # Short aromatic-check derived from MCS
                        has_aromatic = 'c' in (smarts or '')
                        qa_list.append({
                            'question': f"Do {name_a} and {name_b} share an aromatic substructure?",
                            'answer': 'Yes' if has_aromatic else 'No',
                            'cid': f'{cid_a}|{cid_b}',
                            'answer_source': 'rdkit:mcs_aromatic_bool'
                        })

                # Compare H-bond acceptors (short answer: name)
                ha_a = pair_df.iloc[0].get('h_bond_acceptors')
                ha_b = pair_df.iloc[1].get('h_bond_acceptors')
                if ha_a is not None and ha_b is not None and ha_a != ha_b:
                    higher_ha_name = name_a if ha_a > ha_b else name_b
                    qa_list.append({
                        'question': f"Which compound, {name_a} or {name_b}, has a higher number of H-bond acceptors?",
                        'answer': higher_ha_name,
                        'cid': f'{cid_a}|{cid_b}',
                        'answer_source': 'h_bond_acceptors_comparison'
                    })

        # Deduplicate QA entries by (question, cid, answer) to avoid duplicates
        unique_idx = {}
        deduped = []
        for q in qa_list:
            key = (q.get('question'), q.get('cid'), q.get('answer'))
            if key not in unique_idx:
                unique_idx[key] = True
                deduped.append(q)

        return deduped

    def save_outputs(self, qa_list: List[Dict[str, Any]]) -> None:
        # Save dataframe and QA JSON
        try:
            self.df.to_csv(self.csv_file, index=False)
        except Exception as e:
            print(f"Warning: could not save CSV: {e}")
        try:
            with open(self.qa_file, 'w') as f:
                json.dump(qa_list, f, indent=2)
        except Exception as e:
            print(f"Warning: could not save QA JSON: {e}")

    def run(self, cid_list: List[int]) -> None:
        t0 = time.time()
        self.build_dataframe(cid_list)
        qa_from_df = self.qa_from_dataframe()
        self.save_outputs(qa_from_df)
        t1 = time.time()
        print("\n" + "="*60)
        print(f"Finished. Compounds processed: {len(self.df)}")
        print(f"QA pairs created: {len(qa_from_df)}")
        print(f"Properties CSV: {self.csv_file}")
        print(f"QA JSON: {self.qa_file}")
        print(f"Time elapsed: {t1 - t0:.2f} s")
        if self.failed_cids:
            print(f"Failed CIDs: {len(self.failed_cids)} (sample: {self.failed_cids[:5]})")


if __name__ == "__main__":
    start_time = time.time()
    # deterministic CID selection
    cid_list = random.sample(range(1, 17700000), 30)
    generator = PubChemQADataset(output_prefix="chemistry_qa_dataset")
    generator.run(cid_list)
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")