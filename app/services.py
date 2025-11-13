from pathlib import Path
from typing import Tuple, List, Dict, Any

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Carrega o dataset a partir de um arquivo CSV.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado em: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("O dataset está vazio.")
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
    - identifica colunas numéricas e categóricas
    - aplica one-hot encoding nas categóricas
    - normaliza os dados numéricos
    """
    numeric_cols, categorical_cols = detect_column_types(df)

    # Numéricos
    df_numeric = df[numeric_cols].copy() if numeric_cols else pd.DataFrame()

    # Categóricos -> one-hot encoding
    if categorical_cols:
        df_cat = pd.get_dummies(
            df[categorical_cols].fillna("UNKNOWN"),
            drop_first=False
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
    if requested is not None:
        return max(2, requested)

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
            row[f"mean_{col}"] = float(g[col].mean(skipna=True))

        # Moda para categóricas
        for col in categorical_cols:
            mode = g[col].mode(dropna=True)
            row[f"mode_{col}"] = mode.iloc[0] if not mode.empty else None

        rows.append(row)

    agg_df = pd.DataFrame(rows).set_index("cluster_id")
    return agg_df

def count_by_name(df: pd.DataFrame, name: str) -> int:
    """
    Conta quantas pessoas têm o nome informado.
    Pressupõe uma coluna 'nome' no dataset.
    """
    possible_cols = ["nome", "NOME", "Name", "NAME", "full_name", "FULL_NAME"]

    for col in possible_cols:
        if col in df.columns:
            return int((df[col].astype(str).str.lower() == name.lower()).sum())

    raise ValueError("Nenhuma coluna de nome foi encontrada no dataset.")
