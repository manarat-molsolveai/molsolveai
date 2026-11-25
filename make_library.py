"""Offline script: builds library.json from PubChem (name -> CID -> SMILES, InChIKey, weight) with automated checks.
Run: python make_library.py   (needs internet). Output is committed; the app never calls this."""
import json, time, datetime, urllib.parse, urllib.request
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")
NAMES = """Aspirin|Paracetamol|Ibuprofen|Naproxen|Diclofenac|Caffeine|Theophylline|Nicotine|Diazepam|Morphine|Codeine|Penicillin G|Amoxicillin|Metformin|Atenolol|Propranolol|Lidocaine|Fluoxetine|Sertraline|Omeprazole|Ranitidine|Warfarin|Chloramphenicol|Ciprofloxacin|Sildenafil|Atorvastatin|Simvastatin|Cholesterol|Glucose|Ethanol|Dopamine|Serotonin|Testosterone|Estradiol|Hydrocortisone|Loratadine|Cetirizine|Salbutamol|Acyclovir|Tamoxifen
Celecoxib|Indomethacin|Ketoprofen|Piroxicam|Meloxicam|Tramadol|Fentanyl|Oxycodone|Methadone|Naloxone|Naltrexone|Buprenorphine|Alprazolam|Lorazepam|Clonazepam|Zolpidem|Phenobarbital|Carbamazepine|Phenytoin|Valproic acid|Lamotrigine|Levetiracetam|Topiramate|Gabapentin|Pregabalin|Haloperidol|Chlorpromazine|Olanzapine|Risperidone|Quetiapine|Clozapine|Aripiprazole|Lithium carbonate|Amitriptyline|Imipramine|Citalopram|Escitalopram|Paroxetine|Venlafaxine|Bupropion|Mirtazapine|Trazodone|Methylphenidate|Amphetamine|Modafinil|Donepezil|Levodopa|Carbidopa|Selegiline|Ropinirole|Sumatriptan|Melatonin|Cimetidine|Famotidine|Lansoprazole|Pantoprazole|Ondansetron|Metoclopramide|Domperidone|Loperamide|Bisacodyl|Lisinopril|Enalapril|Captopril|Losartan|Valsartan|Amlodipine|Nifedipine|Verapamil|Diltiazem|Metoprolol|Bisoprolol|Carvedilol|Hydrochlorothiazide|Furosemide|Spironolactone|Digoxin|Amiodarone|Clopidogrel|Rivaroxaban|Apixaban|Heparin|Nitroglycerin|Rosuvastatin|Pravastatin|Ezetimibe|Fenofibrate|Gemfibrozil|Glibenclamide|Glimepiride|Pioglitazone|Sitagliptin|Empagliflozin|Levothyroxine|Methimazole|Prednisone|Prednisolone|Dexamethasone|Betamethasone|Progesterone|Levonorgestrel|Ethinylestradiol|Finasteride|Tadalafil|Vardenafil|Cephalexin|Cefuroxime|Ceftriaxone|Azithromycin|Erythromycin|Clarithromycin|Doxycycline|Tetracycline|Gentamicin|Levofloxacin|Moxifloxacin|Sulfamethoxazole|Trimethoprim|Metronidazole|Nitrofurantoin|Vancomycin|Linezolid|Clindamycin|Rifampicin|Isoniazid|Ethambutol|Pyrazinamide|Fluconazole|Ketoconazole|Terbinafine|Amphotericin B|Oseltamivir|Zidovudine|Ribavirin|Tenofovir|Chloroquine|Hydroxychloroquine|Artemisinin|Quinine|Primaquine|Ivermectin|Albendazole|Mebendazole|Praziquantel|Methotrexate|Cyclophosphamide|Doxorubicin|Cisplatin|Paclitaxel|Fluorouracil|Imatinib|Gefitinib|Erlotinib|Capecitabine|Azathioprine|Cyclosporine|Tacrolimus|Allopurinol|Colchicine|Alendronic acid|Montelukast|Fexofenadine|Diphenhydramine|Chlorpheniramine|Pseudoephedrine|Ephedrine|Dextromethorphan|Guaifenesin|Theobromine|Atropine|Scopolamine|Neostigmine|Epinephrine|Norepinephrine|Histamine|Acetylcholine|Adenosine|Cortisol|Vitamin C|Folic acid|Thiamine|Riboflavin|Nicotinamide|Pyridoxine|Cyanocobalamin|Retinol|Cholecalciferol|Tocopherol|Urea|Glycine|Creatine|Taurine|Capsaicin|Menthol|Curcumin|Resveratrol|Quercetin|Lactic acid|Citric acid|Salicylic acid|Benzoic acid|Aniline|Phenol|Toluene|Acetone|Glycerol|Sucrose|Lactose|Dapsone|Bortezomib|Lenalidomide|Thalidomide|Sorafenib|Sunitinib|Tofacitinib|Ibrutinib|Naloxegol|Varenicline|Orlistat|Sibutramine|Minoxidil|Isotretinoin|Tretinoin|Clotrimazole|Miconazole|Benzocaine|Bupivacaine|Ketamine|Propofol|Midazolam|Sevoflurane|Succinylcholine|Rocuronium|Neomycin|Streptomycin|Chlorhexidine|Povidone-iodine""".replace("\n", "|").split("|")
SYN = {'Paracetamol', 'Hydrocortisone', 'Vitamin C', 'Cholecalciferol', 'Tretinoin', 'Cyclosporine'}
NAMES = list(dict.fromkeys(n.strip() for n in NAMES if n.strip()))
def get(url):
    for i in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return None
import os, sys
CACHE = '/tmp/libcache.json'
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
T0 = time.time()
out, bad = [], []
today = datetime.date.today().isoformat()
for n in NAMES:
    u = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/property/ConnectivitySMILES,MolecularWeight,Title,InChIKey,MolecularFormula/JSON" % urllib.parse.quote(n)
    if n in cache: j = cache[n]
    else:
        if time.time() - T0 > 85: continue
        j = get(u) or {"missing": True}; cache[n] = j; json.dump(cache, open(CACHE, "w")); time.sleep(0.25)
    if j.get("missing"): bad.append((n, "not found")); continue
    p = j["PropertyTable"]["Properties"][0]
    smi = p.get("ConnectivitySMILES") or p.get("SMILES")
    m = Chem.MolFromSmiles(smi) if smi else None
    chk = {}
    chk["parses"] = m is not None
    if m is not None:
        chk["single_fragment"] = len(Chem.GetMolFrags(m)) == 1
        chk["weight_matches_pubchem"] = abs(Descriptors.MolWt(m) - float(p["MolecularWeight"])) < 0.1
        chk["title_matches_name"] = n.lower().replace("-", " ") in p["Title"].lower().replace("-", " ") or p["Title"].lower() in n.lower()
    if not chk.get("title_matches_name", True) and n in SYN:
        chk["title_matches_name"] = "known synonym (reviewed by the assistant, not by a chemist)"
    ok = all(bool(v) for v in chk.values())
    ok = ok and m is not None
    rec = {"name": n, "smiles": Chem.MolToSmiles(m) if m else smi, "cid": p["CID"], "inchikey": p["InChIKey"], "formula": p["MolecularFormula"],
           "pubchem_title": p["Title"], "checks": chk, "passed": ok, "source": "PubChem PUG REST (name -> CID)", "retrieved": today}
    (out if ok else bad).append(rec if ok else (n, p["Title"], chk))
    print(("OK  " if ok else "BAD ") + n, p["Title"], flush=True)
_seen=set(); _o=[]
for e in out:
    if e["inchikey"] in _seen: print("duplicate dropped:", e["name"]); continue
    _seen.add(e["inchikey"]); _o.append(e)
out = _o
json.dump({"generated": today, "source": "PubChem", "entries": out, "excluded": [{"name": b[0], "reason": b[1] if len(b)==2 else "failed automated check: " + ", ".join(k for k, v in b[2].items() if not v)} for b in bad]}, open("library.json", "w"), ensure_ascii=False, indent=1)
json.dump(bad, open("/tmp/lib_bad.json", "w"), ensure_ascii=False, indent=1, default=str)
print(len(out), "ok;", len(bad), "flagged;", len(cache), "of", len(NAMES), "fetched")
