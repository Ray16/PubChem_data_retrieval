# PubChemQAGenerator

This script generates a Chemistry Question-Answering (QA) Dataset by fetching properties from PubChem and computationally deriving structural features using RDKit (optional).

🚀 Quick Start

Install Dependencies:

```
pip install pubchempy pandas
conda install -c conda-forge rdkit
```

Run the script
```
python chem_qa_gen.py
```

✨ Features

- Data Source: Fetches basic physicochemical data from PubChem.
- Structural Analysis (RDKit): Calculates Murcko Scaffolds, ring counts, Degree of Unsaturation (DoU), and a comprehensive list of functional groups.
- Intelligent Group Filtering: Uses a hierarchical SMARTS logic to prevent misclassification of functional groups (e.g., an ester is filtered from being counted as a simple ether).
- Drug-Likeness: Assesses compounds against Lipinski's Rule of Five and Veber's Rule.
- QA Generation: Deterministically creates structured QA pairs, including:
  - Counting atoms and functional groups.
  - Boolean checks (e.g., "Does X contain a ketone?").
  - Relational queries (e.g., Maximum Common Substructure (MCS) of two compounds).
  - Basic reactivity predictions.


📁 Output Files

The script generates two files
- chemistry_qa_dataset_properties.csv: A CSV file containing all fetched and computed properties for each CID.
- chemistry_qa_dataset_qa.json: A JSON file containing the generated list of Question-Answer pairs.
