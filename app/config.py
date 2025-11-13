from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Caminho padrão para o CSV com os dados brutos
DATA_PATH = BASE_DIR / "data" / "AequitasDB.csv"

# Parâmetro de privacidade: mínimo de registros em qualquer grupo/cluster
K_ANONYMITY = 10
