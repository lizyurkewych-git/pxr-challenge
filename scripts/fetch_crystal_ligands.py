"""
Download PXR co-crystal ligand SMILES from the OpenADMET re-refinement repository.

Source: github.com/OpenADMET/pxr_xtal_re-refinement (pxr_pdb_ids.txt)
Format: pdb_id, ligand_id, chain, residue_number, SMILES [, repeat for second ligand]
Output: data/pxr_crystal_ligands.csv  [smiles, pdb_id, ligand_id]

Usage:
    python scripts/fetch_crystal_ligands.py
"""

import csv
import sys
import urllib.request
from pathlib import Path

from rdkit import Chem

URL = "https://raw.githubusercontent.com/OpenADMET/pxr_xtal_re-refinement/main/pxr_pdb_ids.txt"
OUT = Path("data/pxr_crystal_ligands.csv")


def canonicalize(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None


def parse_ligand_file(text: str) -> list[dict]:
    """Parse pxr_pdb_ids.txt into a list of {pdb_id, ligand_id, smiles} dicts."""
    records = []
    for line in text.strip().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 5:
            continue
        pdb_id = parts[0]
        # Each ligand entry is 4 fields: ligand_id, chain, residue_num, SMILES
        i = 1
        while i + 3 < len(parts):
            ligand_id = parts[i]
            smiles_raw = parts[i + 3]
            can = canonicalize(smiles_raw)
            if can:
                records.append({"pdb_id": pdb_id, "ligand_id": ligand_id, "smiles": can})
            i += 4
    return records


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {URL} ...")
    with urllib.request.urlopen(URL) as resp:
        text = resp.read().decode("utf-8")

    records = parse_ligand_file(text)
    print(f"Parsed {len(records)} raw ligand entries")

    # Deduplicate by canonical SMILES
    seen = set()
    unique = []
    for r in records:
        if r["smiles"] not in seen:
            seen.add(r["smiles"])
            unique.append(r)

    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles", "pdb_id", "ligand_id"])
        writer.writeheader()
        writer.writerows(unique)

    print(f"Saved {len(unique)} unique crystal ligands → {OUT}")

    by_lig = {}
    for r in records:
        by_lig[r["ligand_id"]] = by_lig.get(r["ligand_id"], 0) + 1
    most_common = sorted(by_lig.items(), key=lambda x: -x[1])[:5]
    print(f"Most common ligand IDs: {most_common}")


if __name__ == "__main__":
    main()
