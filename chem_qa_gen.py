import pubchempy as pcp
import json
import time
import pandas as pd
import requests
from typing import Dict, List, Tuple


class PubChemQAGenerator:
    def __init__(self, output_file: str = "chemistry_qa.json", max_workers: int = 5):
        self.output_file = output_file
        self.qa_file = output_file if output_file.endswith('.json') else output_file + '.json'
        self.csv_file = self.qa_file.replace('.json', '_properties.csv')
        self.qa_pairs = []
        self.failed_compounds = []
        self.compound_data_list = []
        self.max_workers = max_workers
        self.csv_header_written = False
    
    def save_compound_to_csv(self, compound_data: Dict) -> None:
        """Append a single compound to CSV file"""
        df = pd.DataFrame([compound_data])
        
        if not self.csv_header_written:
            df.to_csv(self.csv_file, mode='w', index=False, header=True)
            self.csv_header_written = True
        else:
            df.to_csv(self.csv_file, mode='a', index=False, header=False)
    
    def save_qa_batch_to_json(self, qa_batch: List[Dict]) -> None:
        """Append Q&A pairs to JSON file in real-time"""
        try:
            try:
                with open(self.qa_file, 'r') as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []
            
            existing_data.extend(qa_batch)
            
            with open(self.qa_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
        except Exception as e:
            print(f"  Warning: Could not save Q&A to JSON: {str(e)}")
    
    def get_description_from_api(self, cid: int) -> Dict:
        """Fetch compound description from PubChem REST API with retry logic"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/description/JSON"
                
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                result = {
                    'description': None,
                    'category': None,
                    'synonyms': None
                }
                
                if 'InformationList' in data and 'Information' in data['InformationList']:
                    info_list = data['InformationList']['Information']
                    if isinstance(info_list, list):
                        for info in info_list:
                            if 'Description' in info:
                                result['description'] = info['Description']
                                break
                        
                        if len(info_list) > 0:
                            first_info = info_list[0]
                            if 'Category' in first_info:
                                result['category'] = first_info['Category']
                
                time.sleep(2)
                return result
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 503:
                    if attempt < max_retries - 1:
                        print(f"  Server busy (CID {cid}), retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 1.5
                    else:
                        print(f"  Failed to fetch CID {cid} after {max_retries} retries")
                        return {'description': None, 'category': None, 'synonyms': None}
                else:
                    return {'description': None, 'category': None, 'synonyms': None}
            except Exception as e:
                return {'description': None, 'category': None, 'synonyms': None}
        
        return {'description': None, 'category': None, 'synonyms': None}
    
    def get_compound_data(self, cid: int) -> Dict:
        """Fetch comprehensive compound data from PubChem"""
        try:
            compound = pcp.Compound.from_cid(cid)
            
            data = {
                'name': compound.iupac_name or (compound.synonyms[0] if compound.synonyms else f"Compound {cid}"),
                'cid': cid,
                'molecular_formula': getattr(compound, 'molecular_formula', None),
                'molecular_weight': getattr(compound, 'molecular_weight', None),
                'h_bond_donors': getattr(compound, 'h_bond_donor_count', None),
                'h_bond_acceptors': getattr(compound, 'h_bond_acceptor_count', None),
                'rotatable_bonds': getattr(compound, 'rotatable_bond_count', None),
                'topological_psa': getattr(compound, 'tpsa', None),
                'logp': getattr(compound, 'xlogp', None),
                'heavy_atom_count': getattr(compound, 'heavy_atom_count', None),
                'exact_mass': getattr(compound, 'exact_mass', None),
                'monoisotopic_mass': getattr(compound, 'monoisotopic_mass', None),
                'isotope_atom_count': getattr(compound, 'isotope_atom_count', None),
                'atom_stereo_count': getattr(compound, 'atom_stereo_count', None),
                'covalent_unit_count': getattr(compound, 'covalent_unit_count', None),
                'description': None,
                'category': None,
                'synonyms': None,
            }
            
            desc_data = self.get_description_from_api(cid)
            data['description'] = desc_data['description']
            data['category'] = desc_data['category']
            data['synonyms'] = desc_data['synonyms']
            
            return data
        except Exception as e:
            error_msg = f"CID {cid}: {str(e)}"
            print(f"  ERROR - {error_msg}")
            self.failed_compounds.append((cid, str(e)))
            return None
    
    def extract_factual_statements_from_description(self, desc: str, compound_name: str) -> List[Tuple[str, str]]:
        """
        Extract factual statements from description and generate Q&A pairs.
        
        Returns:
            List of (question, answer) tuples
        """
        if not desc:
            return []
        
        qa_pairs = []
        sentences = [s.strip() for s in desc.split('.') if s.strip()]
        
        # Debug: Print what we're analyzing
        print(f"\n  [DEBUG] Analyzing description for {compound_name[:30]}...")
        print(f"  [DEBUG] Description: {desc[:200]}...")
        print(f"  [DEBUG] Found {len(sentences)} sentences")
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            name_lower = compound_name.lower()
            
            print(f"  [DEBUG] Sentence {i+1}: {sentence[:100]}")
            
            # Pattern 1: "X is a Y" or "X is an Y"
            # More flexible: check if sentence starts with compound name or "it"
            is_patterns = [
                f"{name_lower} is a ",
                f"{name_lower} is an ",
                "it is a ",
                "it is an "
            ]
            
            for pattern in is_patterns:
                if pattern in sentence_lower:
                    print(f"    [DEBUG] Found pattern: '{pattern}'")
                    match_idx = sentence_lower.find(pattern)
                    prefix_len = len(pattern)
                    remainder = sentence[match_idx + prefix_len:].strip()
                    
                    # Take up to the first subordinate clause or end
                    for delimiter in [' that ', ' which ', ' where ', ' and is ', ' and has ']:
                        if delimiter in remainder.lower():
                            remainder = remainder[:remainder.lower().find(delimiter)].strip()
                            break
                    
                    classification = remainder.strip()
                    
                    # Filter: reasonable length and not empty
                    if classification and 2 <= len(classification.split()) <= 10:
                        question = f"Is {compound_name} a {classification}?"
                        qa_pairs.append((question, "Yes"))
                        print(f"    [DEBUG] ✓ Generated Q&A: {question}")
                    break
            
            # Pattern 2: "It is functionally related to X"
            if "functionally related to" in sentence_lower:
                print(f"    [DEBUG] Found 'functionally related to'")
                match = sentence_lower.find("functionally related to")
                remainder = sentence[match + len("functionally related to"):].strip()
                related_compound = remainder.split(',')[0].split('.')[0].split(' and ')[0].strip()
                
                # Remove leading article
                for article in ['a ', 'an ', 'the ']:
                    if related_compound.lower().startswith(article):
                        related_compound = related_compound[len(article):].strip()
                        break
                
                if related_compound and 1 <= len(related_compound.split()) <= 8:
                    question = f"Is {compound_name} functionally related to {related_compound}?"
                    qa_pairs.append((question, "Yes"))
                    print(f"    [DEBUG] ✓ Generated Q&A: {question}")
            
            # Pattern 3: "It is a conjugate acid/base of X"
            for conjugate_type in ["conjugate acid of", "conjugate base of"]:
                if conjugate_type in sentence_lower:
                    print(f"    [DEBUG] Found '{conjugate_type}'")
                    match = sentence_lower.find(conjugate_type)
                    remainder = sentence[match + len(conjugate_type):].strip()
                    conjugate = remainder.split(',')[0].split('.')[0].split(' and ')[0].strip()
                    
                    # Remove leading article
                    for article in ['a ', 'an ', 'the ']:
                        if conjugate.lower().startswith(article):
                            conjugate = conjugate[len(article):].strip()
                            break
                    
                    if conjugate and 1 <= len(conjugate.split()) <= 8:
                        question = f"Is {compound_name} a {conjugate_type} {conjugate}?"
                        qa_pairs.append((question, "Yes"))
                        print(f"    [DEBUG] ✓ Generated Q&A: {question}")
            
            # Pattern 4: "occurs naturally in X" or "found naturally in X"
            for naturally_pattern in ["occurs naturally in", "found naturally in"]:
                if naturally_pattern in sentence_lower:
                    print(f"    [DEBUG] Found '{naturally_pattern}'")
                    match = sentence_lower.find(naturally_pattern)
                    remainder = sentence[match + len(naturally_pattern):].strip()
                    location = remainder.split(' and ')[0].split(',')[0].split('.')[0].strip()
                    
                    if location and 1 <= len(location.split()) <= 8:
                        question = f"Does {compound_name} occur naturally in {location}?"
                        qa_pairs.append((question, "Yes"))
                        print(f"    [DEBUG] ✓ Generated Q&A: {question}")
                    break
            
            # Pattern 5: "found in X" (but not "found naturally")
            if "found in" in sentence_lower and "naturally" not in sentence_lower[:sentence_lower.find("found in") + 20]:
                print(f"    [DEBUG] Found 'found in'")
                match = sentence_lower.find("found in")
                remainder = sentence[match + len("found in"):].strip()
                location = remainder.split(' and ')[0].split(',')[0].split('.')[0].strip()
                
                if location and 1 <= len(location.split()) <= 8:
                    question = f"Is {compound_name} found in {location}?"
                    qa_pairs.append((question, "Yes"))
                    print(f"    [DEBUG] ✓ Generated Q&A: {question}")
            
            # Pattern 6: "has a role as X" or "role as X"
            for role_pattern in ["has a role as", "has role as", "role as"]:
                if role_pattern in sentence_lower:
                    print(f"    [DEBUG] Found '{role_pattern}'")
                    match = sentence_lower.find(role_pattern)
                    remainder = sentence[match + len(role_pattern):].strip()
                    role = remainder.split(' and ')[0].split(',')[0].split('.')[0].strip()
                    
                    # Remove leading article
                    for article in ['a ', 'an ', 'the ']:
                        if role.lower().startswith(article):
                            role = role[len(article):].strip()
                            break
                    
                    if role and 1 <= len(role.split()) <= 10:
                        question = f"Does {compound_name} have a role as {role}?"
                        qa_pairs.append((question, "Yes"))
                        print(f"    [DEBUG] ✓ Generated Q&A: {question}")
                    break
            
            # Pattern 7: "derived from X"
            if "derived from" in sentence_lower:
                print(f"    [DEBUG] Found 'derived from'")
                match = sentence_lower.find("derived from")
                remainder = sentence[match + len("derived from"):].strip()
                source = remainder.split(' and ')[0].split(',')[0].split(' by ')[0].split('.')[0].strip()
                
                # Remove leading article
                for article in ['a ', 'an ', 'the ']:
                    if source.lower().startswith(article):
                        source = source[len(article):].strip()
                        break
                
                if source and 1 <= len(source.split()) <= 8:
                    question = f"Is {compound_name} derived from {source}?"
                    qa_pairs.append((question, "Yes"))
                    print(f"    [DEBUG] ✓ Generated Q&A: {question}")
            
            # Pattern 8: "metabolite of X" or "metabolite in X"
            for metabolite_pattern in ["metabolite of", "metabolite in"]:
                if metabolite_pattern in sentence_lower:
                    print(f"    [DEBUG] Found '{metabolite_pattern}'")
                    match = sentence_lower.find(metabolite_pattern)
                    remainder = sentence[match + len(metabolite_pattern):].strip()
                    organism = remainder.split(' and ')[0].split(',')[0].split('.')[0].strip()
                    
                    if organism and 1 <= len(organism.split()) <= 8:
                        question = f"Is {compound_name} a metabolite in {organism}?"
                        qa_pairs.append((question, "Yes"))
                        print(f"    [DEBUG] ✓ Generated Q&A: {question}")
                    break
            
            # Pattern 9: "obtained from X" or "isolated from X"
            for source_pattern in ["obtained from", "isolated from", "extracted from"]:
                if source_pattern in sentence_lower:
                    print(f"    [DEBUG] Found '{source_pattern}'")
                    match = sentence_lower.find(source_pattern)
                    remainder = sentence[match + len(source_pattern):].strip()
                    source = remainder.split(' and ')[0].split(',')[0].split('.')[0].strip()
                    
                    if source and 1 <= len(source.split()) <= 8:
                        question = f"Is {compound_name} obtained from {source}?"
                        qa_pairs.append((question, "Yes"))
                        print(f"    [DEBUG] ✓ Generated Q&A: {question}")
                    break
        
        print(f"  [DEBUG] Total Q&A pairs extracted: {len(qa_pairs)}\n")
        return qa_pairs
    
    def interpret_classification_from_description(self, desc: str, classification: str) -> str:
        """
        Interpret classification from description using context-aware analysis.
        Returns 'Yes', 'No', or None if not mentioned.
        
        Args:
            desc: Full description text
            classification: Target classification (e.g., 'amino acid', 'drug', 'vitamin')
        """
        if not desc:
            return None
        
        desc_lower = desc.lower()
        classification_lower = classification.lower()
        
        # Check if classification is mentioned at all
        if classification_lower not in desc_lower:
            return None
        
        # Split into sentences for context analysis
        sentences = [s.strip() for s in desc.split('.') if s.strip()]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Skip if classification not in this sentence
            if classification_lower not in sentence_lower:
                continue
            
            # Context patterns that indicate the compound IS the classification
            positive_patterns = [
                f"is a {classification_lower}",
                f"is an {classification_lower}",
                f"as a {classification_lower}",
                f"as an {classification_lower}",
                f"acts as a {classification_lower}",
                f"acts as an {classification_lower}",
                f"classified as a {classification_lower}",
                f"classified as an {classification_lower}",
                f"type of {classification_lower}",
                f"serves as a {classification_lower}",
                f"serves as an {classification_lower}",
            ]
            
            # Check for positive patterns first
            has_positive_pattern = any(pattern in sentence_lower for pattern in positive_patterns)
            
            if has_positive_pattern:
                # Even with positive pattern, check for strong negations
                strong_negations = [
                    "is not a",
                    "is not an",
                    "not classified as",
                    "cannot be classified as",
                ]
                has_strong_negation = any(neg in sentence_lower for neg in strong_negations)
                
                if has_strong_negation:
                    return 'No'
                
                # Check for modifiers that don't negate the base classification
                # e.g., "non-proteinogenic alpha-amino acid" is still an amino acid
                modifiers = ["non-proteinogenic", "non-standard", "non-essential", 
                           "synthetic", "artificial", "modified", "derivative"]
                
                # If it has a modifier but still uses positive pattern, it's still that class
                return 'Yes'
            
            # Check for explicit negations without positive patterns
            explicit_negations = [
                f"not a {classification_lower}",
                f"not an {classification_lower}",
                f"is not {classification_lower}",
                f"not classified as {classification_lower}",
                f"lacks {classification_lower}",
                f"without {classification_lower}",
            ]
            
            has_explicit_negation = any(neg in sentence_lower for neg in explicit_negations)
            if has_explicit_negation:
                return 'No'
        
        # If mentioned but context unclear, return None to avoid false positives
        return None
    
    def generate_qa_pairs(self, compound_data: Dict) -> List[Tuple[str, str, bool]]:
        """Generate Q&A pairs from compound properties - only exact answers
        
        Returns:
            List of tuples: (question, answer, is_from_description)
        """
        if not compound_data:
            return []
        
        name = compound_data['name']
        pairs = []
        
        # Category question
        if compound_data['category']:
            pairs.append((
                f"What category does {name} belong to?",
                compound_data['category'],
                False  # Not from description interpretation
            ))
        
        # Hydrogen bond donor questions
        if compound_data['h_bond_donors'] is not None:
            pairs.append((
                f"How many hydrogen bond donors does {name} have?",
                f"{compound_data['h_bond_donors']}",
                False
            ))
        
        # Hydrogen bond acceptor questions
        if compound_data['h_bond_acceptors'] is not None:
            pairs.append((
                f"What is the number of hydrogen bond acceptors in {name}?",
                f"{compound_data['h_bond_acceptors']}",
                False
            ))
        
        # Molecular weight questions
        if compound_data['molecular_weight']:
            pairs.append((
                f"What is the molecular weight of {name}?",
                f"{compound_data['molecular_weight']:.2f} g/mol",
                False
            ))
        
        # Molecular formula
        if compound_data['molecular_formula']:
            pairs.append((
                f"What is the molecular formula of {name}?",
                compound_data['molecular_formula'],
                False
            ))
        
        # Lipophilicity questions
        if compound_data['logp'] is not None:
            pairs.append((
                f"What is the logP (lipophilicity) of {name}?",
                f"{compound_data['logp']:.2f}",
                False
            ))
        
        # Rotatable bonds
        if compound_data['rotatable_bonds'] is not None:
            pairs.append((
                f"How many rotatable bonds does {name} have?",
                f"{compound_data['rotatable_bonds']}",
                False
            ))
        
        # Topological Polar Surface Area
        if compound_data['topological_psa'] is not None:
            pairs.append((
                f"What is the topological polar surface area (TPSA) of {name}?",
                f"{compound_data['topological_psa']:.2f}",
                False
            ))
        
        # Heavy atom count
        if compound_data['heavy_atom_count'] is not None:
            pairs.append((
                f"How many heavy atoms does {name} contain?",
                f"{compound_data['heavy_atom_count']}",
                False
            ))
        
        # Exact mass
        if compound_data['exact_mass'] is not None:
            pairs.append((
                f"What is the exact mass of {name}?",
                f"{compound_data['exact_mass']:.4f}",
                False
            ))
        
        # Isotope atom count
        if compound_data['isotope_atom_count'] is not None:
            pairs.append((
                f"How many isotope atoms does {name} contain?",
                f"{compound_data['isotope_atom_count']}",
                False
            ))
        
        # Stereochemistry
        if compound_data['atom_stereo_count'] is not None:
            pairs.append((
                f"How many stereogenic centers does {name} have?",
                f"{compound_data['atom_stereo_count']}",
                False
            ))
        
        # Covalent units
        if compound_data['covalent_unit_count'] is not None:
            pairs.append((
                f"How many covalent units does {name} have?",
                f"{compound_data['covalent_unit_count']}",
                False
            ))
        
        # Lipinski's rule of five check (KEEP - has scientific significance)
        if (compound_data['molecular_weight'] is not None and 
            compound_data['h_bond_donors'] is not None and
            compound_data['h_bond_acceptors'] is not None and
            compound_data['logp'] is not None):
            
            h_donors = compound_data['h_bond_donors']
            h_acceptors = compound_data['h_bond_acceptors']
            mw = compound_data['molecular_weight']
            logp = compound_data['logp']
            
            passes_lipinski = (mw <= 500 and h_donors <= 5 and 
                             h_acceptors <= 10 and logp <= 5)
            
            pairs.append((
                f"Does {name} satisfy Lipinski's rule of five?",
                f"{'Yes' if passes_lipinski else 'No'}",
                False
            ))
        
        # REMOVED: Arbitrary threshold questions (TPSA > 140, logP > 5, MW > 500)
        # These lack scientific context and are not meaningful without application context
        
        # Description-based classification with improved interpretation
        # ALL QUESTIONS BELOW ARE MARKED AS description_question=True
        if compound_data['description']:
            desc = compound_data['description']
            
            # FIRST: Extract factual statements from description
            factual_qa = self.extract_factual_statements_from_description(desc, name)
            for question, answer in factual_qa:
                pairs.append((
                    question,
                    answer,
                    True  # FROM DESCRIPTION - needs manual verification
                ))
            
            # SECOND: Classification-based questions
            # Define classifications to check
            classifications = [
                ('drug', 'a drug'),
                ('vitamin', 'a vitamin'),
                ('amino acid', 'an amino acid'),
                ('hormone', 'a hormone'),
                ('enzyme', 'an enzyme'),
                ('antibiotic', 'an antibiotic'),
                ('antioxidant', 'an antioxidant'),
            ]
            
            for classification, article_form in classifications:
                result = self.interpret_classification_from_description(desc, classification)
                if result is not None:
                    pairs.append((
                        f"Is {name} {article_form}?",
                        result,
                        True  # FROM DESCRIPTION - needs manual verification
                    ))
            
            # Special case for antibody (check both terms)
            antibody_result = self.interpret_classification_from_description(desc, 'antibody')
            if antibody_result is None:
                antibody_result = self.interpret_classification_from_description(desc, 'immunoglobulin')
            if antibody_result is not None:
                pairs.append((
                    f"Is {name} an antibody?",
                    antibody_result,
                    True  # FROM DESCRIPTION - needs manual verification
                ))
            
            # Check for property-based questions
            desc_lower = desc.lower()
            
            # Anti-inflammatory
            if 'anti-inflammatory' in desc_lower or 'antiinflammatory' in desc_lower:
                result = self.interpret_classification_from_description(desc, 'anti-inflammatory')
                if result is not None:
                    pairs.append((
                        f"Does {name} have anti-inflammatory properties?",
                        result,
                        True  # FROM DESCRIPTION - needs manual verification
                    ))
            
            # Anti-cancer
            for term in ['anticancer', 'anti-cancer', 'anti-tumor', 'antitumor']:
                if term in desc_lower:
                    result = self.interpret_classification_from_description(desc, term)
                    if result is not None:
                        pairs.append((
                            f"Does {name} have anti-cancer properties?",
                            result,
                            True  # FROM DESCRIPTION - needs manual verification
                        ))
                    break
        
        return pairs
    
    def generate_dataset(self, cid_list: List[int]) -> None:
        """Generate Q&A dataset sequentially to respect rate limits"""
        total = len(cid_list)
        processed = 0
        qa_batch = []
        batch_size = 5
        
        print(f"Processing {total} compounds sequentially (conservative rate limiting)...")
        print(f"Saving to: {self.qa_file} and {self.csv_file}")
        print("=" * 50)
        
        for cid in cid_list:
            processed += 1
            compound_data = self.get_compound_data(cid)
            qa_pairs = self.generate_qa_pairs(compound_data) if compound_data else []
            
            if compound_data:
                self.compound_data_list.append(compound_data)
                self.save_compound_to_csv(compound_data)
                
                for qa_tuple in qa_pairs:
                    question, answer, is_from_desc = qa_tuple
                    qa_batch.append({
                        'question': question,
                        'answer': answer,
                        'compound_cid': cid,
                        'compound_name': compound_data['name'],
                        'description_question': is_from_desc
                    })
                    self.qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'compound_cid': cid,
                        'compound_name': compound_data['name'],
                        'description_question': is_from_desc
                    })
                
                if len(qa_batch) >= batch_size:
                    self.save_qa_batch_to_json(qa_batch)
                    qa_batch = []
                
                desc_status = "✓ has desc" if compound_data['description'] else "✗ no desc"
                print(f"[{processed}/{total}] {compound_data['name'][:50]} ({len(qa_pairs)} QA) {desc_status}")
            else:
                print(f"[{processed}/{total}] ✗ CID {cid} failed")
        
        if qa_batch:
            self.save_qa_batch_to_json(qa_batch)
    
    def save_dataset(self) -> None:
        """Print final summary"""
        print(f"\n{'=' * 50}")
        print(f"Total Q&A pairs generated: {len(self.qa_pairs)}")
        print(f"Unique compounds: {len(self.compound_data_list)}")
        print(f"Failed compounds: {len(self.failed_compounds)}")
        print(f"Q&A dataset saved to {self.qa_file}")
        print(f"Properties CSV saved to {self.csv_file}")


if __name__ == "__main__":
    generator = PubChemQAGenerator(output_file="chemistry_qa_dataset.json", max_workers=5)
    
    # Example: Use first 30 compound CIDs for testing
    cid_list = list[int](range(1, 30))
    
    print("Starting PubChem Q&A dataset generation...")
    print("=" * 50)
    
    generator.generate_dataset(cid_list)
    generator.save_dataset()