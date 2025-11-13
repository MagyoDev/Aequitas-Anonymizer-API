from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Caminho padrão para o CSV com os dados brutos
DATA_PATH = BASE_DIR / "data" / "AequitasDB.csv"

# Parâmetro de privacidade: mínimo de registros em qualquer grupo/cluster
K_ANONYMITY = 10

# Limite máximo de registros para uma consulta agregada
MAX_RESULTS = 4000

# Colunas consideradas sensíveis e que não podem ser usadas em clusterização/agregação
SENSITIVE_COLUMNS = [
    "CPF",
    "RG",
    "CNH",
    "PASSAPORTE",
    "TELEFONE",
    "END_RESIDENCIAL",
    "NOME",  # não usar NOME como feature de ML
]
