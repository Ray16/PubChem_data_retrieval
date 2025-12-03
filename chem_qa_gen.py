#!/usr/bin/env python3

import time
import json
import re
import random
from typing import List, Dict, Optional, Tuple, Any, Set
import pandas as pd
import pubchempy as pcp

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
    'carboxylate': ['ketone'], 'primary_amide': ['ketone'], 'secondary_amide': ['ketone'],
    'tertiary_amide': ['ketone'], 'acyl_chloride': ['ketone'], 'anhydride': ['ketone', 'ester'],
    'aldehyde': ['ketone'], 'thioester': ['thioketone'], 'carbamate': ['ester', 'ether'],
    'carbonate': ['ester', 'ether'], 
    'urea': ['primary_amide', 'secondary_amide', 'tertiary_amide'], # Added tertiary_amide here
    'sulfonamide': ['sulfone'], 'sulfonic_acid': ['sulfone'], 'hemiacetal': ['ether', 'secondary_alcohol'],
    'acetal': ['ether'], 'phosphate': ['ester', 'ether'], 'phosphonate': ['ester', 'ether'],
}

def filter_subgroups_by_location(mol, detected_groups: List[str], fg_smarts: Dict[str, str]) -> List[str]:
    """
    Remove subgroups only when they overlap with parent groups at the same location.
    """
    if not detected_groups or mol is None: return detected_groups
    groups_set = set(detected_groups)
    group_atoms: Dict[str, List[Set[int]]] = {}
    
    for group in detected_groups:
        if group not in fg_smarts: continue
        try:
            patt = Chem.MolFromSmarts(fg_smarts[group])
            if patt: group_atoms[group] = [set(match) for match in mol.GetSubstructMatches(patt)]
        except Exception: continue
    
    to_remove: Set[str] = set()
    for parent, subgroups in FUNCTIONAL_GROUP_HIERARCHY.items():
        if parent not in groups_set or parent not in group_atoms: continue
        parent_atom_sets = group_atoms[parent]
        for subgroup in subgroups:
            if subgroup not in groups_set or subgroup not in group_atoms: continue
            subgroup_atom_sets = group_atoms[subgroup]
            all_overlap = True
            for sub_atoms in subgroup_atom_sets:
                overlaps_with_parent = False
                for parent_atoms in parent_atom_sets:
                    if sub_atoms.issubset(parent_atoms) or len(sub_atoms & parent_atoms) >= len(sub_atoms) * 0.5:
                        overlaps_with_parent = True
                        break
                if not overlaps_with_parent:
                    all_overlap = False
                    break
            if all_overlap: to_remove.add(subgroup)
    return [g for g in detected_groups if g not in to_remove]

# ---------- Utility: RDKit-based derived features ----------
def compute_rdkit_features(smiles: str) -> Dict[str, Any]:
    """
    Given a SMILES string, compute functional groups and scaffolds.
    (Stereochemistry and Fsp3 calculations removed).
    """
    features = {
        'murcko_scaffold': None,
        'functional_groups': None,
        'rdkit_mol': None,
    }

    if not rdkit_available or not smiles:
        return features

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return features

        # Murcko scaffold
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold and scaffold.GetNumAtoms() > 0:
                features['murcko_scaffold'] = Chem.MolToSmiles(scaffold, isomericSmiles=False)
        except Exception:
            features['murcko_scaffold'] = None

        # Functional groups - comprehensive detection
        fg_smarts = {
            'primary_amine': '[#7X3;H2;!$(NC=O);!$(NS=O)]', 
            'secondary_amine': '[#7X3;H1;!$(NC=O);!$(NS=O)]([#6])[#6]',
            'tertiary_amine': '[#7X3;H0;!$(NC=O);!$(NS=O)]([#6])([#6])[#6]', 
            'quaternary_ammonium': '[#7+;H0]([#6])([#6])([#6])[#6]',
            
            # Amides & Ureas (Using generic atom tags to catch cyclic/aromatic variants)
            'primary_amide': '[#7X3;H2][#6X3](=O)[#6]', 
            'secondary_amide': '[#7X3;H1]([#6])[#6X3](=O)[#6]',
            'tertiary_amide': '[#7X3;H0]([#6])([#6])[#6X3](=O)[#6]', 
            'urea': '[#7X3;!@R][#6X3](=O)[#7X3;!@R]',
            'carbamate': '[#7X3][#6X3](=O)[OX2]', 
            
            'imine': '[#7X2]=[#6X3]', 'nitrile': '[#7X1]#[#6X2]',
            'nitro': '[$([#7X3](=O)=O),$([#7X3+](=O)[O-])][!#8]', 
            'nitroso': '[#7X2](=O)[#6]', 'azide': '[#7X2]=[#7X2+]=[#7X1-]',
            
            # Oxygen groups
            'primary_alcohol': '[#6X4][OX2H]', 'secondary_alcohol': '[#6X4H]([#6])[OX2H]',
            'tertiary_alcohol': '[#6X4]([#6])([#6])([#6])[OX2H]', 'phenol': 'c[OX2H]',
            'carboxylic_acid': '[#6X3](=O)[OX2H1]', 'carboxylate': '[#6X3](=O)[OX1-,OX2-]',
            'ester': '[#6X3](=O)[OX2][#6;!$(C=O)]', 'lactone': '[#6]~1~[#6]~[#6](=O)[OX2]~[#6]~[#6]~1',
            'ketone': '[#6][#6X3](=O)[#6]', 'aldehyde': '[#6X3H1](=O)[#6]', 
            'acyl_chloride': '[#6X3](=O)[Cl]',
            'anhydride': '[#6X3](=O)[OX2][#6X3](=O)', 'ether': '[OD2]([#6])[#6]', 
            
            # Sulfur/Phosphorus
            'thiol': '[SX2H]', 'sulfide': '[SX2]([#6])[#6]', 'disulfide': '[SX2][SX2]',
            'sulfoxide': '[SX3](=O)([#6])[#6]', 'sulfone': '[SX4](=O)(=O)([#6])[#6]',
            'sulfonamide': '[SX4](=O)(=O)[#7X3]', 
            'phosphate': '[PX4](=O)([OX2])([OX2])[OX2]', 
            
            # Halogens & Aromatics
            'fluoride': '[FX1]', 'chloride': '[ClX1]', 'bromide': '[BrX1]', 'iodide': '[IX1]',
            'benzene': 'c1ccccc1', 'pyridine': 'n1ccccc1',
            'pyrrole': '[nH]1cccc1', 'furan': 'o1cccc1', 'thiophene': 's1cccc1', 
            
            # FIX: Imidazole now matches c1ncn1 (generic) or substituted N
            'imidazole': 'c1nc[n,nH,nX3]c1', 
            
            'alkene': '[CX3]=[CX3]', 'alkyne': '[CX2]#[CX2]',
        }
        
        matched = []
        for name, smarts in fg_smarts.items():
            try:
                patt = Chem.MolFromSmarts(smarts)
                if patt and mol.HasSubstructMatch(patt): matched.append(name)
            except Exception: pass
        
        filtered_groups = filter_subgroups_by_location(mol, matched, fg_smarts)
        features['functional_groups'] = filtered_groups if filtered_groups else None
        
        # Save RDKit mol object for ring logic later
        features['rdkit_mol'] = mol

    except Exception:
        pass

    return features

# ---------- Core Utility Functions ----------
def parse_formula(formula: str) -> Dict[str, int]:
    atom_counts = {}
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

def compute_lcs_smarts(mol1_smiles: str, mol2_smiles: str) -> Optional[str]:
    """Computes the SMARTS of the Maximum Common Substructure (MCS)."""
    if not rdkit_available: return None
    try:
        mol1 = Chem.MolFromSmiles(mol1_smiles)
        mol2 = Chem.MolFromSmiles(mol2_smiles)
        if mol1 and mol2:
            mcs_result = FindMCS([mol1, mol2])
            if mcs_result.numAtoms > 0:
                return mcs_result.smartsString
    except Exception:
        return None
    return None

# ---------- Main generator class ----------
class PubChemQADataset:
    def __init__(self, output_prefix: str = "chemistry_qa_dataset"):
        self.output_prefix = output_prefix
        self.csv_file = f"{output_prefix}_properties.csv"
        self.qa_file = f"{output_prefix}_qa.json"
        self.df = pd.DataFrame()
        self.failed_cids: List[Tuple[int, str]] = []
        self.fg_smarts_list = [
            'primary_alcohol', 'secondary_alcohol', 'phenol', 'carboxylic_acid',
            'ketone', 'aldehyde', 'ester', 'ether', 'alkene', 'alkyne',
            'primary_amine', 'secondary_amine', 'primary_amide', 'sulfone',
            'thiol', 'nitrile'
        ]

    def fetch_compound_basic(self, cid: int) -> Optional[Dict[str, Any]]:
        """
        Fetch compound basic properties from PubChem.
        Stereo queries and Exact Mass are disabled.
        """
        try:
            compound = pcp.Compound.from_cid(cid)
            if compound is None:
                return None

            name = compound.iupac_name or f"Compound {cid}"
            canonical_smiles = getattr(compound, 'connectivity_smiles', None)
            
            data = {
                'cid': cid,
                'name': name,
                'canonical_smiles': canonical_smiles,
                'molecular_formula': getattr(compound, 'molecular_formula', None),
                'molecular_weight': getattr(compound, 'molecular_weight', None), # Kept
                # 'exact_mass': getattr(compound, 'exact_mass', None),  <-- Removed as requested
                'h_bond_donors': getattr(compound, 'h_bond_donor_count', None),
                'h_bond_acceptors': getattr(compound, 'h_bond_acceptor_count', None),
                'rotatable_bonds': getattr(compound, 'rotatable_bond_count', None),
                'topological_psa': getattr(compound, 'tpsa', None),
                'logp': getattr(compound, 'xlogp', None),
                'heavy_atom_count': getattr(compound, 'heavy_atom_count', None),
            }

            return data

        except Exception as e:
            self.failed_cids.append((cid, str(e)))
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

            # Use canonical_smiles (ignoring stereo)
            smiles_for_rdkit = raw.get('canonical_smiles')
            rdkit_feats = compute_rdkit_features(smiles_for_rdkit) if use_rdkit_if_available else {}

            # Deterministic derived fields
            mw, hd, ha, logp = raw.get('molecular_weight'), raw.get('h_bond_donors'), raw.get('h_bond_acceptors'), raw.get('logp')
            tpsa, rb = raw.get('topological_psa'), raw.get('rotatable_bonds')
            
            # Lipinski's Rule of Five
            lipinski_pass = None
            if None not in (mw, hd, ha, logp):
                lipinski_pass = (mw <= 500 and hd <= 5 and ha <= 10 and logp <= 5)
            
            # Veber's Rule
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

        self.df = pd.DataFrame(rows)
        # reorder columns
        cols = list(self.df.columns)
        if 'cid' in cols:
            cols.insert(0, cols.pop(cols.index('cid')))
        self.df = self.df.reindex(columns=cols)

    # ---------- Deterministic Q&A generation ----------
    def qa_from_dataframe(self):
        qa_list = []
        
        countable_functional_groups = ['alcohol', 'amine', 'ketone', 'ether', 'ester', 'phenol', 'aldehyde', 'nitrile']
        countable_atoms = ['Cl', 'Br', 'F', 'I', 'N', 'O', 'S', 'P', 'C', 'H']
        all_possible_groups = self.fg_smarts_list + ['carbamate', 'urea', 'guanidine'] 

        for idx, row in self.df.iterrows():
            name = row.get('name', f'compound_{idx}')
            fg = row.get('functional_groups', [])
            formula = row.get('molecular_formula', '')
            mol = row.get('rdkit_mol', None)
            cid = row.get('cid')
            
            atom_counts = parse_formula(formula) if formula else {}

            # 1. Counting Questions
            fg_to_count = random.sample(countable_functional_groups, k=min(3, len(countable_functional_groups)))
            for group in fg_to_count:
                count = sum(1 for f in fg if group in f) if isinstance(fg, (list, set)) else 0
                qa_list.append({'question': f"How many {group.replace('_',' ')} groups does {name} contain?", 'answer': str(count), 'cid': cid, 'answer_source': 'functional_groups (counted)'})
            
            if formula:
                atoms_to_count = random.sample(countable_atoms, k=min(3, len(countable_atoms)))
                for atom in atoms_to_count:
                    atom_count = atom_counts.get(atom, 0)
                    qa_list.append({'question': f"How many {atom} atoms does {name} have?", 'answer': str(atom_count), 'cid': cid, 'answer_source': 'molecular_formula (parsed)'})

            # 2. Boolean Functional Group Queries
            queried_groups = random.sample(all_possible_groups, k=3)
            for group in queried_groups:
                has_group = any(group in x for x in fg) if isinstance(fg, (list, set)) else False
                qa_list.append({'question': f"Does {name} contain a {group.replace('_', ' ')} group?", 'answer': 'Yes' if has_group else 'No', 'cid': cid, 'answer_source': f'functional_groups ({group} boolean)'})

            # 3. Formula Analysis
            if formula:
                qa_list.append({'question': f"How many nitrogen atoms are in {name}?", 'answer': str(atom_counts.get('N', 0)), 'cid': cid, 'answer_source': 'molecular_formula (parsed)'})
                qa_list.append({'question': f"What is the carbon to hydrogen ratio in {name}?", 'answer': f"{atom_counts.get('C', 0)}:{atom_counts.get('H', 0)}", 'cid': cid, 'answer_source': 'molecular_formula (parsed)'})
                dou = calculate_dou(atom_counts)
                qa_list.append({'question': f"What is the degree of unsaturation of {name}?", 'answer': str(dou), 'cid': cid, 'answer_source': 'molecular_formula (calculated)'})

            # 4. Structural Feature Presence (RDKit)
            if mol and rdkit_available:
                ring_info = mol.GetRingInfo()
                n_rings = ring_info.NumRings()
                qa_list.append({'question': f"How many rings does {name} contain?", 'answer': str(n_rings), 'cid': cid, 'answer_source': 'rdkit ring_count'})
                if n_rings > 0:
                    ring_sizes = [len(r) for r in ring_info.AtomRings()]
                    qa_list.append({'question': f"What is the size of the largest ring in {name}?", 'answer': str(max(ring_sizes)), 'cid': cid, 'answer_source': 'rdkit ring_sizes'})
                has_5 = any(len(r) == 5 for r in ring_info.AtomRings())
                qa_list.append({'question': f"Does {name} contain a five-membered ring?", 'answer': 'Yes' if has_5 else 'No', 'cid': cid, 'answer_source': 'rdkit ring_size boolean'})
            
            # 5. Advanced Reasoning Questions (Stereo & Fsp3 REMOVED)
            
            # C. Drug-Likeness Comparison
            if row.get('lipinski_pass') is not None and row.get('veber_pass') is not None:
                mw, hd, ha, logp = row['molecular_weight'], row['h_bond_donors'], row['h_bond_acceptors'], row['logp']
                tpsa, rb = row['topological_psa'], row['rotatable_bonds']
                
                lipinski_fail = 4 - sum([mw <= 500, hd <= 5, ha <= 10, logp <= 5])
                veber_fail = 2 - sum([rb <= 10, tpsa <= 140])
                
                comparison_answer = 'Lipinski' if lipinski_fail > veber_fail else \
                                    'Veber' if veber_fail > lipinski_fail else 'Neither (violations are equal or zero)'
                                    
                qa_list.append({'question': f"Based on the number of violations, which set of rules does {name} deviate from more: **Lipinski's Rule of Five** or **Veber's Rule**?", 'answer': comparison_answer, 'cid': cid, 'answer_source': 'multi_rule_comparison'})
                
            # D. Structure-Reactivity Prediction
            has_aldehyde = any('aldehyde' in x for x in fg) if isinstance(fg, list) else False
            has_ketone = any('ketone' in x for x in fg) if isinstance(fg, list) else False
            
            if has_aldehyde and has_ketone:
                qa_list.append({'question': f"If {name} were treated with a **mild reducing agent** (e.g., NaBH4), which functional group would be reduced first: the **aldehyde** or the **ketone**?", 'answer': 'Aldehyde (The aldehyde is more electrophilic and therefore more reactive to mild reduction.)', 'cid': cid, 'answer_source': 'chemical_reactivity_comparison'})

        # 6. Relational/Comparative Queries
        if len(self.df) >= 2 and rdkit_available:
            for _ in range(min(5, len(self.df) // 2)):
                pair_df = self.df.sample(n=2, replace=False)
                # Ensure we use canonical smiles for LCS
                smiles_a = pair_df.iloc[0]['canonical_smiles']
                smiles_b = pair_df.iloc[1]['canonical_smiles']
                name_a, name_b = pair_df.iloc[0]['name'], pair_df.iloc[1]['name']
                cid_a, cid_b = pair_df.iloc[0]['cid'], pair_df.iloc[1]['cid']
                
                if smiles_a and smiles_b:
                    # Maximum Common Substructure (LCS/MCS)
                    lcs_smarts = compute_lcs_smarts(smiles_a, smiles_b)
                    if lcs_smarts:
                        qa_list.append({'question': f"What is the SMARTS string for the **Maximum Common Substructure (MCS)** shared by {name_a} and {name_b}?", 'answer': lcs_smarts, 'cid': f'{cid_a}|{cid_b}', 'answer_source': 'rdkit:mcs'})
                
                    # Physicochemical Comparison (e.g., H-bonds)
                    ha_a = pair_df.iloc[0]['h_bond_acceptors']
                    ha_b = pair_df.iloc[1]['h_bond_acceptors']
                    if ha_a is not None and ha_b is not None and ha_a != ha_b:
                        higher_ha_name = name_a if ha_a > ha_b else name_b
                        qa_list.append({'question': f"Which compound, **{name_a}** (HA: {ha_a}) or **{name_b}** (HA: {ha_b}), has a higher number of H-bond **acceptors**?", 'answer': higher_ha_name, 'cid': f'{cid_a}|{cid_b}', 'answer_source': 'h_bond_acceptors_comparison'})
                        
        return qa_list

    def save_outputs(self, qa_list: List[Dict[str, Any]]) -> None:
        self.df.to_csv(self.csv_file, index=False)
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
    cid_list = [2244, 2519, 5904, 338, 54670067] 
    generator = PubChemQADataset(output_prefix="chemistry_qa_dataset")
    generator.run(cid_list)
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")