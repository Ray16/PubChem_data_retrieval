# PubChemQAGenerator

A simple Python tool that fetches PubChem compound data and generates question–answer (Q&A) pairs from both numerical properties and descriptive text.

## Features
- Retrieves compound data using PubChemPy and PubChem REST API  
- Extracts factual statements from descriptions  
- Generates Q&A pairs from:
  - molecular properties (MW, formula, donors, acceptors, TPSA, LogP, etc.)
  - description text  
- Saves results to:
  - `chemistry_qa_dataset.json`
  - `chemistry_qa_dataset_properties.csv`

## Installation
```
pip install pubchempy pandas requests
```

## Basic Usage
```
python chem_qa_gen.py
```


## Example Q&A pair
```
{
  "question": "What is the molecular weight of L-alanine?",
  "answer": "89.09 g/mol",
  "compound_cid": 5950,
  "compound_name": "L-alanine",
  "description_question": false
}
```