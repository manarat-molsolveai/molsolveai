"""الإعدادات. القيم تُقرأ من ملف .env إن وُجد، ولا توجد أسرار داخل الكود."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data" / "raw"
CKPT_DIR = BASE_DIR / "checkpoints"
CKPT_PATH = CKPT_DIR / "molsolve.pt"

HIDDEN_DIM = int(os.getenv("MOLSOLVE_HIDDEN_DIM", 128))
NUM_LAYERS = int(os.getenv("MOLSOLVE_NUM_LAYERS", 4))
EPOCHS = int(os.getenv("MOLSOLVE_EPOCHS", 30))
BATCH_SIZE = int(os.getenv("MOLSOLVE_BATCH_SIZE", 64))
LR = float(os.getenv("MOLSOLVE_LR", 1e-3))
SEED = int(os.getenv("MOLSOLVE_SEED", 42))
PATIENCE = 8
DROPOUT = 0.15
FP_BITS = 1024
SECRET_KEY = os.getenv("MOLSOLVE_SECRET_KEY", "")

# المهام المتعلَّمة (من بيانات عامة حقيقية)
TOX21_TASKS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
               "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
REG_TASKS = ["LogS"]                      # الذوبانية (ESOL)
CLS_TASKS = ["BBB"] + TOX21_TASKS         # عبور الحاجز الدموي الدماغي + 12 اختبار سمّية
TASKS = REG_TASKS + CLS_TASKS

DISCLAIMER = ("⚠️ للبحث والتعلم فقط. التنبؤات تقديرية ولا تصلح لقرار علاجي أو دوائي "
              "أو لسلامة الاستخدام. راجع مختصاً وتجارب مخبرية.")
