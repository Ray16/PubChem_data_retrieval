import requests
import json
import time
import csv
import pandas as pd
from typing import List, Dict
from pathlib import Path
from datetime import datetime

dataset_name = 'Compound_1_1000'

class PubChemBulkDownloader:
    """Download PubChem descriptions while respecting rate limits."""
    
    def __init__(self, batch_size: int = 100, requests_per_second: float = 2):
        """
        Initialize downloader with rate limiting.
        
        Args:
            batch_size: Number of CIDs per request (100-200 recommended)
            requests_per_second: Conservative rate limit (PubChem allows 5, use 2-3 to be safe)
        """
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"
        self.batch_size = batch_size
        self.delay = 1.0 / requests_per_second
        self.descriptions = {}
        self.failed_batches = []
        self.request_count = 0
        self.start_time = None
        self.current_delay = self.delay  # Dynamic delay adjustment
        
    def download(self, cids: List[int]) -> Dict:
        """
        Download descriptions for all CIDs while respecting rate limits.
        
        Args:
            cids: List of compound IDs
            
        Returns:
            Dictionary with descriptions keyed by CID
        """
        self.start_time = time.time()
        cids = sorted(set(cids))
        batches = [cids[i:i + self.batch_size] for i in range(0, len(cids), self.batch_size)]
        
        print(f"Starting download of {len(cids)} compounds in {len(batches)} batches")
        print(f"Delay between requests: {self.delay:.2f}s (rate limit: {1/self.delay:.1f} req/s)")
        print(f"Estimated time: ~{len(batches) * self.delay / 60:.1f} minutes\n")
        
        for idx, batch in enumerate(batches, 1):
            self._process_batch(batch, idx, len(batches))
            
            if idx < len(batches):
                time.sleep(self.current_delay)
        
        elapsed = time.time() - self.start_time
        print(f"\n✓ Download complete in {elapsed:.1f}s")
        print(f"  Successfully downloaded: {len(self.descriptions)}")
        print(f"  Failed batches: {len(self.failed_batches)}")
        
        return self.descriptions
    
    def _process_batch(self, batch: List[int], batch_num: int, total_batches: int):
        """Process a single batch of CIDs."""
        cid_str = ",".join(map(str, batch))
        url = f"{self.base_url}/{cid_str}/description/JSON"
        
        try:
            print(f"[{batch_num}/{total_batches}] Requesting CIDs {batch[0]}-{batch[-1]} ({len(batch)} total)...", end=" ", flush=True)
            self.request_count += 1
            
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if "InformationList" in data:
                    for info in data["InformationList"].get("Information", []):
                        cid = info.get("CID")
                        if cid:
                            if cid not in self.descriptions:
                                self.descriptions[cid] = {}
                            
                            self.descriptions[cid].update({
                                "Title": info.get("Title", ""),
                                "Description": info.get("Description", ""),
                                "DescriptionSource": info.get("DescriptionSourceName", ""),
                                "DescriptionURL": info.get("DescriptionURL", "")
                            })
                
                print(f"✓ ({len([i for i in data.get('InformationList', {}).get('Information', []) if 'Description' in i])} descriptions)")
                # Reset delay on success
                self.current_delay = self.delay
            
            elif response.status_code == 404:
                print(f"✗ Status 404 (likely rate limit/connection issue)")
                self._retry_batch_with_backoff(batch, url)
            
            else:
                print(f"✗ Status {response.status_code}")
                self._retry_batch_with_backoff(batch, url)
        
        except requests.exceptions.Timeout:
            print(f"✗ Timeout")
            self._retry_batch_with_backoff(batch, url)
        except requests.exceptions.RequestException as e:
            print(f"✗ Error: {str(e)[:30]}")
            self._retry_batch_with_backoff(batch, url)
    
    def _retry_batch(self, batch: List[int], url: str, max_retries: int = 2):
        """Retry a failed batch with exponential backoff, then test individual CIDs."""
        for attempt in range(1, max_retries + 1):
            wait_time = 5 * (2 ** (attempt - 1))
            print(f"  Retrying in {wait_time}s (attempt {attempt}/{max_retries})...", flush=True)
            time.sleep(wait_time)
            
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if "InformationList" in data:
                        for info in data["InformationList"].get("Information", []):
                            cid = info.get("CID")
                            if cid:
                                if cid not in self.descriptions:
                                    self.descriptions[cid] = {}
                                self.descriptions[cid].update({
                                    "Title": info.get("Title", ""),
                                    "Description": info.get("Description", ""),
                                    "DescriptionSource": info.get("DescriptionSourceName", ""),
                                    "DescriptionURL": info.get("DescriptionURL", "")
                                })
                    print(f"  ✓ Retry successful!")
                    return
            except:
                pass
        
        # If batch still fails, test individual CIDs
        print(f"  Testing individual CIDs in batch...")
        self._test_individual_cids(batch)
    
    def _retry_batch_with_backoff(self, batch: List[int], url: str, max_retries: int = 3):
        """Retry with decreased rate limit on errors."""
        for attempt in range(1, max_retries + 1):
            # Increase delay on each retry (progressive backoff)
            wait_time = 5 * (2 ** (attempt - 1))
            # Decrease requests per second dynamically
            self.current_delay = self.delay * (2 ** (attempt - 1))
            
            print(f"  Decreasing rate to {1/self.current_delay:.1f} req/s, retrying in {wait_time}s (attempt {attempt}/{max_retries})...", flush=True)
            time.sleep(wait_time)
            
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if "InformationList" in data:
                        for info in data["InformationList"].get("Information", []):
                            cid = info.get("CID")
                            if cid:
                                if cid not in self.descriptions:
                                    self.descriptions[cid] = {}
                                self.descriptions[cid].update({
                                    "Title": info.get("Title", ""),
                                    "Description": info.get("Description", ""),
                                    "DescriptionSource": info.get("DescriptionSourceName", ""),
                                    "DescriptionURL": info.get("DescriptionURL", "")
                                })
                    print(f"  ✓ Retry successful! Rate restored to {1/self.delay:.1f} req/s")
                    self.current_delay = self.delay  # Reset to original rate
                    return
            except:
                pass
        
        # If batch still fails after retries with backoff, test individual CIDs
        print(f"  Testing individual CIDs with reduced rate...")
        self._test_individual_cids(batch)
    
    def _test_individual_cids(self, batch: List[int]):
        """Test each CID individually to find valid ones."""
        valid_count = 0
        invalid_cids = []
        
        for cid in batch:
            url = f"{self.base_url}/{cid}/description/JSON"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if "InformationList" in data:
                        for info in data["InformationList"].get("Information", []):
                            cid_result = info.get("CID")
                            if cid_result:
                                if cid_result not in self.descriptions:
                                    self.descriptions[cid_result] = {}
                                self.descriptions[cid_result].update({
                                    "Title": info.get("Title", ""),
                                    "Description": info.get("Description", ""),
                                    "DescriptionSource": info.get("DescriptionSourceName", ""),
                                    "DescriptionURL": info.get("DescriptionURL", "")
                                })
                                valid_count += 1
                else:
                    invalid_cids.append(cid)
            except:
                invalid_cids.append(cid)
            
            time.sleep(0.2)  # Small delay between individual requests
        
        if invalid_cids:
            print(f"  ✓ Found {valid_count} valid CIDs, {len(invalid_cids)} invalid/missing")
            print(f"  Invalid CIDs: {invalid_cids}")
        else:
            print(f"  ✓ All {valid_count} CIDs in batch are valid")


def append_descriptions_to_csv(input_csv: str, output_csv: str, cid_column: str = "CID"):
    """
    Read CSV, download descriptions, and append as new column.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to save updated CSV file
        cid_column: Name of the column containing CIDs
    """
    print(f"Reading CSV from: {input_csv}\n")
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Verify CID column exists
    if cid_column not in df.columns:
        print(f"Error: Column '{cid_column}' not found in CSV")
        print(f"Available columns: {list(df.columns)}")
        return
    
    print(f"Found {len(df)} rows in CSV")
    print(f"CID column: '{cid_column}'")
    
    # Extract CIDs and download descriptions
    cids = df[cid_column].tolist()
    
    downloader = PubChemBulkDownloader(batch_size=150, requests_per_second=2)
    descriptions = downloader.download(cids)
    
    # Create description column
    print("\nCreating description column...")
    description_data = []
    found_count = 0
    
    for cid in cids:
        if cid in descriptions and descriptions[cid].get("Description"):
            description_data.append(descriptions[cid]["Description"])
            found_count += 1
        else:
            description_data.append("")
    
    # Add new column to dataframe
    df["Description"] = description_data
    
    # Save updated CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Updated CSV saved to: {output_csv}")
    print(f"  Total rows: {len(df)}")
    print(f"  Rows with descriptions: {found_count}")
    print(f"  Columns: {list(df.columns)}")


# Main execution
if __name__ == "__main__":
    # Paths
    input_file = f"output/{dataset_name}.csv"
    output_file = f"output/{dataset_name}_with_description.csv"
    
    # Run the process
    append_descriptions_to_csv(input_file, output_file, cid_column="PUBCHEM_COMPOUND_CID")