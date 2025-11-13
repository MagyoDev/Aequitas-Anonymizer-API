from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from . import config


def extract_city(address: Any) -> Optional[str]:
    """
    Extrai uma 'cidade/bairro' simplificada a partir do campo END_RESIDENCIAL.
    Heurística simples:
    - se existir '–', pega o texto depois do '–'
    - remove UF e espaços extras
    """
    if not isinstance(address, str) or not address.strip():
        return None

    part = address
    if "–" in address:
        part = address.split("–", 1)[1]

    part = part.strip()

    # remover possível UF no final (ex: ", RJ", ", ES")
    if "," in part:
        fragments = [p.strip() for p in part.split(",")]
        # se o último fragmento for 2 letras (possível UF), remove
        if len(fragments[-1]) == 2 and fragments[-1].isalpha():
            fragments = fragments[:-1]
        part = ", ".join(fragments)

    return part or None


def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Carrega o dataset a partir de um arquivo CSV.
    Enriquecendo com a coluna 'CIDADE' derivada do endereço, se existir.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado em: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("O dataset está vazio.")

    # cria coluna CIDADE a partir de END_RESIDENCIAL, se existir
    if "END_RESIDENCIAL" in df.columns and "CIDADE" not in df.columns:
        df["CIDADE"] = df["END_RESIDENCIAL"].apply(extract_city)

    return df


def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Separa colunas numéricas e categóricas de forma automática.
    """
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    categorical_cols = [
        col for col in df.columns
        if df[col].dtype == "object"
    ]

    return numeric_cols, categorical_cols


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Pré-processa o dataframe:
    - remove colunas sensíveis das features de ML
    - identifica colunas numéricas e categóricas
    - aplica one-hot encoding nas categóricas
    - normaliza os dados numéricos
    """
    # removendo colunas sensíveis do conjunto de features
    df_features = df.copy()
    cols_to_drop = [c for c in config.SENSITIVE_COLUMNS if c in df_features.columns]
    df_features = df_features.drop(columns=cols_to_drop, errors="ignore")

    numeric_cols, categorical_cols = detect_column_types(df_features)

    # Numéricos
    df_numeric = df_features[numeric_cols].copy() if numeric_cols else pd.DataFrame()

    # Categóricos -> one-hot encoding
    if categorical_cols:
        df_cat = pd.get_dummies(
            df_features[categorical_cols].fillna("UNKNOWN"),
            drop_first=False,
        )
    else:
        df_cat = pd.DataFrame()

    # Concatena tudo
    processed = pd.concat([df_numeric, df_cat], axis=1)

    if processed.empty:
        raise ValueError("Não há colunas utilizáveis para clusterização.")

    processed = processed.fillna(0)

    # Normalização (StandardScaler)
    scaler = StandardScaler()
    processed_scaled = pd.DataFrame(
        scaler.fit_transform(processed),
        columns=processed.columns,
        index=processed.index,
    )

    return processed_scaled, numeric_cols, categorical_cols


def choose_n_clusters(n_samples: int, requested: int | None = None) -> int:
    """
    Define um número de clusters razoável caso o usuário não especifique.
    """
    if requested is not None and requested >= 2:
        return requested

    if n_samples <= 20:
        return max(2, n_samples // 2)
    if n_samples <= 200:
        return max(2, n_samples // 10)

    # Regra simples: aproximadamente sqrt(n)
    return max(2, int(n_samples ** 0.5))


def clusterize(
    processed_data: pd.DataFrame,
    requested_clusters: int | None = None,
) -> Tuple[pd.Series, int]:
    """
    Executa KMeans sobre os dados pré-processados.
    Retorna:
    - labels (cluster_id para cada linha)
    - número efetivo de clusters usados
    """
    n_samples = processed_data.shape[0]
    n_clusters = choose_n_clusters(n_samples, requested_clusters)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(processed_data)
    return pd.Series(labels, index=processed_data.index, name="cluster_id"), n_clusters


def aggregate_clusters(
    df_original: pd.DataFrame,
    labels: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    """
    Gera um dataframe de agregados por cluster:
    - tamanho do cluster
    - médias para colunas numéricas
    - moda para colunas categóricas
    Colunas sensíveis já foram removidas da lista de features.
    """
    df = df_original.copy()
    df["cluster_id"] = labels

    groups = df.groupby("cluster_id")
    rows: List[Dict[str, Any]] = []

    for cluster_id, g in groups:
        row: Dict[str, Any] = {
            "cluster_id": int(cluster_id),
            "size": int(len(g)),
        }

        # Médias numéricas
        for col in numeric_cols:
            if col in g.columns:
                row[f"mean_{col}"] = float(g[col].mean(skipna=True))

        # Moda para categóricas
        for col in categorical_cols:
            if col in g.columns:
                mode = g[col].mode(dropna=True)
                row[f"mode_{col}"] = mode.iloc[0] if not mode.empty else None

        rows.append(row)

    agg_df = pd.DataFrame(rows).set_index("cluster_id")
    return agg_df


def count_by_name(df: pd.DataFrame, name: str) -> int:
    """
    Conta quantas pessoas têm o nome informado.
    Pressupõe uma coluna 'NOME' no dataset.
    """
    if "NOME" not in df.columns:
        raise ValueError("Coluna 'NOME' não encontrada no dataset.")
    return int((df["NOME"].astype(str).str.lower() == name.lower()).sum())


def count_by_filters(df: pd.DataFrame, **filters) -> int:
    """
    Conta quantos registros atendem aos filtros cruzados.
    Exemplo de filtros:
    - NOME="Lucas Ferreira"
    - IDADE=45
    - SEXO="F"
    - OCUPACAO="Analista de Sistemas"
    - CIDADE="Laranjeiras"
    """
    query = df.copy()

    for col, val in filters.items():
        if val is None:
            continue
        if col not in query.columns:
            # ignora filtros para colunas inexistentes
            continue

        series = query[col].astype(str).str.lower()
        query = query[series == str(val).lower()]

        if query.empty:
            break

    return len(query)
