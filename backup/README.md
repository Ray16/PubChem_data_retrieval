# Molecule dataset construction from PubChem

This repo contains code to construct molecule dataset from PubChem data, which can be either downloaded from [FTP site](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/) or using [PUG REST API](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest#section=URL-based-API).

The dataset consist of microscopic molecular properties such as complexity, and macroscopic descriptions of molecules. In the current implementation, the microscopic molecular properties are parsed from the SDF data, whereas the macroscopic descriptions are retrieved via API.

## Usage
First, the SDF data can be parsed by running `python extract_from_sdf.py`. This code parses `Compound_1_1000.sdf`, a subset of `Compound_000000001_000500000.sdf` downloaded from [https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/).

The resulting csv file is saved to `output/Compound_1_1000.csv`, and contains the following information:

- PUBCHEM_COMPOUND_CID: PubChem Compound ID
- PUBCHEM_CACTVS_COMPLEXITY: complexity of molecule
- PUBCHEM_CACTVS_HBOND_ACCEPTOR: number of hydrogen acceptors
- PUBCHEM_CACTVS_HBOND_DONOR: number of hydrogen donors
- PUBCHEM_CACTVS_ROTATABLE_BOND: number of rotatable bonds
- PUBCHEM_IUPAC_NAME: The IUPAC name
- PUBCHEM_XLOGP3_AA: hydrophobicity of molecule. Higher values indicates more hydrophobic.
- PUBCHEM_MOLECULAR_FORMULA: chemical formula
- PUBCHEM_MOLECULAR_WEIGHT: molecular weight
- PUBCHEM_SMILES: SMILES string of molecule
- PUBCHEM_CACTVS_TPSA: topological polar surface area
- PUBCHEM_TOTAL_CHARGE: total charge
- PUBCHEM_HEAVY_ATOM_COUNT: number of heavy atoms

Next, use `fetch_description.py` to retrieve descritpion of molecules via PUG REST API: https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/[CID]/description/JSON.

The descriptions (when available) are appended to a new column named `description`, and the new file is saved to `output/Compound_1_1000_with_description.csv`.