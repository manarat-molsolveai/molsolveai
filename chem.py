"""تحويل SMILES إلى رسم بياني (ذرات وروابط) + بصمة Morgan + خصائص RDKit الفيزيائية."""
from typing import Optional
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import config

RDLogger.DisableLog("rdApp.*")

ATOMS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "other"]
HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
       Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]
BONDS = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


def _onehot(v, choices):
    out = [0.0] * (len(choices) + 1)          # آخر خانة = "غير ذلك"
    out[choices.index(v) if v in choices else -1] = 1.0
    return out


def atom_features(a):
    sym = a.GetSymbol() if a.GetSymbol() in ATOMS else "other"
    return (_onehot(sym, ATOMS[:-1]) + _onehot(a.GetDegree(), [0, 1, 2, 3, 4, 5])
            + _onehot(a.GetFormalCharge(), [-1, 0, 1]) + _onehot(a.GetHybridization(), HYB)
            + _onehot(a.GetTotalNumHs(), [0, 1, 2, 3])
            + [float(a.GetIsAromatic()), float(a.IsInRing())])


def bond_features(b):
    return _onehot(b.GetBondType(), BONDS) + [float(b.GetIsConjugated()), float(b.IsInRing())]


ATOM_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_DIM = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


def canonical(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)


def scaffold(smiles: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        return ""




def physchem(smiles: str) -> dict:
    """خصائص تُحسب بدقة مباشرة (بلا ذكاء اصطناعي)."""
    m = Chem.MolFromSmiles(smiles)
    return {
        "الوزن الجزيئي MW": round(Descriptors.MolWt(m), 2),
        "LogP (دهنية)": round(Crippen.MolLogP(m), 2),
        "TPSA (المساحة القطبية)": round(rdMolDescriptors.CalcTPSA(m), 2),
        "مانحات الهيدروجين HBD": Lipinski.NumHDonors(m),
        "مستقبلات الهيدروجين HBA": Lipinski.NumHAcceptors(m),
        "الروابط الدوّارة": rdMolDescriptors.CalcNumRotatableBonds(m),
    }


def lipinski_violations(p: dict) -> int:
    return int(p["الوزن الجزيئي MW"] > 500) + int(p["LogP (دهنية)"] > 5) + \
        int(p["مانحات الهيدروجين HBD"] > 5) + int(p["مستقبلات الهيدروجين HBA"] > 10)
