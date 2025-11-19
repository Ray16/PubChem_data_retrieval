import os
from rdkit.Chem import PandasTools

name_of_sdf = 'Compound_1_1000'

os.makedirs('output',exist_ok=True)

sdf_file_path = f"split/{name_of_sdf}.sdf"
df = PandasTools.LoadSDF(sdf_file_path, smilesName='SMILES', molColName='Molecule', includeFingerprints=False)
# remove redundant columns
columns_to_keep = ['PUBCHEM_COMPOUND_CID','PUBCHEM_CACTVS_COMPLEXITY','PUBCHEM_CACTVS_HBOND_ACCEPTOR','PUBCHEM_CACTVS_HBOND_DONOR','PUBCHEM_CACTVS_ROTATABLE_BOND','PUBCHEM_IUPAC_NAME','PUBCHEM_XLOGP3_AA','PUBCHEM_MOLECULAR_FORMULA','PUBCHEM_MOLECULAR_WEIGHT','PUBCHEM_SMILES','PUBCHEM_CACTVS_TPSA','PUBCHEM_TOTAL_CHARGE','PUBCHEM_HEAVY_ATOM_COUNT']
df = df[columns_to_keep]
df.to_csv(f'output/{name_of_sdf}.csv',index=False)