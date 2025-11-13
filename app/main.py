import os
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import config
from . import services


app = FastAPI(
    title="Aequitas Anonymizer API",
    description=(
        "API para anonimização de dados pessoais via clusterização. "
        "Retorna apenas resultados agregados, com k-anonymity e limites de consulta."
    ),
    version="2.0.0",
)


class FitRequest(BaseModel):
    n_clusters: Optional[int] = None


class FitResponse(BaseModel):
    num_records: int
    num_clusters: int
    k_anonymity: int
    max_results: int


class NameStatsResponse(BaseModel):
    name: str
    count: int
    anonymized: bool
    message: str


class MultiStatsResponse(BaseModel):
    filters: Dict[str, Any]
    count: int
    anonymized: bool
    message: str


class ClusterSummary(BaseModel):
    cluster_id: int
    size: int


class ClusterDetail(BaseModel):
    cluster_id: int
    size: int
    numeric_means: Dict[str, Optional[float]]
    categorical_modes: Dict[str, Optional[str]]


class AnonymizerState:
    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.cluster_labels: Optional[pd.Series] = None
        self.agg_clusters: Optional[pd.DataFrame] = None
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.n_clusters: Optional[int] = None


state = AnonymizerState()


def _ensure_fitted():
    if state.df is None or state.cluster_labels is None or state.agg_clusters is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo ainda não foi treinado. Chame o endpoint /fit primeiro.",
        )


def apply_privacy_rules(count: int) -> Optional[Dict[str, Any]]:
    """
    Aplica regras de privacidade:
    - count < K_ANONYMITY  -> bloqueia (risco de reidentificação)
    - count > MAX_RESULTS  -> bloqueia (consulta ampla demais)
    - caso contrário        -> permitido
    """
    if count == 0:
        # quem chama decide a mensagem de 'nenhum registro'
        return None

    if count < config.K_ANONYMITY:
        return {
            "count": 0,
            "anonymized": True,
            "message": "Consulta bloqueada por privacidade (k-anonymity).",
        }

    if count > config.MAX_RESULTS:
        return {
            "count": 0,
            "anonymized": True,
            "message": "Consulta bloqueada por exceder o limite máximo permitido (mais de "
                       f"{config.MAX_RESULTS} resultados).",
        }

    return None


@app.on_event("startup")
def startup_event():
    """
    Carrega dados e treina automaticamente, EXCETO quando estiver no CI.
    """
    if os.getenv("CI") == "true":
        print("[startup] Modo CI detectado — ignorando treinamento automático.")
        return

    try:
        df = services.load_data(config.DATA_PATH)
        processed, numeric_cols, categorical_cols = services.preprocess(df)
        labels, n_clusters = services.clusterize(processed)

        agg = services.aggregate_clusters(df, labels, numeric_cols, categorical_cols)

        state.df = df
        state.cluster_labels = labels
        state.agg_clusters = agg
        state.numeric_cols = numeric_cols
        state.categorical_cols = categorical_cols
        state.n_clusters = n_clusters
    except Exception as e:
        print(f"[startup] Aviso: não foi possível treinar automaticamente. Motivo: {e}")


@app.post("/fit", response_model=FitResponse)
def fit_model(body: FitRequest):
    """
    Recarrega o CSV e re-treina o modelo de clusterização.
    Pode receber opcionalmente o número de clusters desejado.
    """
    try:
        df = services.load_data(config.DATA_PATH)
        processed, numeric_cols, categorical_cols = services.preprocess(df)
        labels, n_clusters = services.clusterize(
            processed,
            requested_clusters=body.n_clusters,
        )
        agg = services.aggregate_clusters(df, labels, numeric_cols, categorical_cols)

        state.df = df
        state.cluster_labels = labels
        state.agg_clusters = agg
        state.numeric_cols = numeric_cols
        state.categorical_cols = categorical_cols
        state.n_clusters = n_clusters

        return FitResponse(
            num_records=len(df),
            num_clusters=n_clusters,
            k_anonymity=config.K_ANONYMITY,
            max_results=config.MAX_RESULTS,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/nome/{nome}", response_model=NameStatsResponse)
def stats_by_name(nome: str):
    """
    Devolve uma resposta agregada do tipo:
    "existem X pessoas com o nome Juan", respeitando:
    - k-anonymity (mínimo 10)
    - limite máximo de 4000 resultados
    """
    _ensure_fitted()

    try:
        count = services.count_by_name(state.df, nome)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if count == 0:
        return NameStatsResponse(
            name=nome,
            count=0,
            anonymized=True,
            message=f"Não há registros com o nome {nome}.",
        )

    privacy = apply_privacy_rules(count)
    if privacy:
        return NameStatsResponse(
            name=nome,
            count=privacy["count"],
            anonymized=privacy["anonymized"],
            message=privacy["message"],
        )

    return NameStatsResponse(
        name=nome,
        count=count,
        anonymized=True,
        message=f"Existem {count} pessoas com o nome {nome}.",
    )


@app.get("/stats/multi", response_model=MultiStatsResponse)
def stats_multi(
    nome: Optional[str] = None,
    idade: Optional[int] = None,
    sexo: Optional[str] = None,
    ocupacao: Optional[str] = None,
    cidade: Optional[str] = None,
):
    """
    Consulta cruzada com múltiplos atributos:
    - nome (NOME)
    - idade (IDADE)
    - sexo (SEXO)
    - ocupação (OCUPACAO)
    - cidade (CIDADE, derivada de END_RESIDENCIAL)

    Aplica:
    - k-anonymity (mínimo 10)
    - limite máximo de 4000 resultados
    """
    _ensure_fitted()

    filters = {
        "NOME": nome,
        "IDADE": idade,
        "SEXO": sexo,
        "OCUPACAO": ocupacao,
        "CIDADE": cidade,
    }

    count = services.count_by_filters(state.df, **filters)

    if count == 0:
        return MultiStatsResponse(
            filters={k: v for k, v in filters.items() if v is not None},
            count=0,
            anonymized=True,
            message="Nenhum registro encontrado para os filtros informados.",
        )

    privacy = apply_privacy_rules(count)
    if privacy:
        return MultiStatsResponse(
            filters={k: v for k, v in filters.items() if v is not None},
            count=privacy["count"],
            anonymized=privacy["anonymized"],
            message=privacy["message"],
        )

    return MultiStatsResponse(
        filters={k: v for k, v in filters.items() if v is not None},
        count=count,
        anonymized=True,
        message=f"Existem {count} pessoas que atendem a esses atributos.",
    )


@app.get("/clusters", response_model=List[ClusterSummary])
def list_clusters():
    """
    Lista os clusters com seus tamanhos (apenas agregados).
    Não retorna clusters menores que k-anonymity.
    """
    _ensure_fitted()

    summaries: List[ClusterSummary] = []
    for cluster_id, row in state.agg_clusters.iterrows():
        size = int(row["size"])
        if size < config.K_ANONYMITY:
            continue
        summaries.append(
            ClusterSummary(
                cluster_id=int(cluster_id),
                size=size,
            )
        )

    return summaries


@app.get("/clusters/{cluster_id}", response_model=ClusterDetail)
def cluster_detail(cluster_id: int):
    """
    Devolve detalhes agregados de um cluster específico:
    - tamanho
    - médias numéricas
    - modas categóricas
    (respeitando k-anonymity)
    """
    _ensure_fitted()

    if cluster_id not in state.agg_clusters.index:
        raise HTTPException(status_code=404, detail="Cluster não encontrado.")

    row = state.agg_clusters.loc[cluster_id]
    size = int(row["size"])

    if size < config.K_ANONYMITY:
        raise HTTPException(
            status_code=403,
            detail="Cluster muito pequeno para divulgação (k-anonymity).",
        )

    numeric_means: Dict[str, Any] = {}
    categorical_modes: Dict[str, Any] = {}

    for col in state.numeric_cols:
        key = f"mean_{col}"
        if key in row:
            numeric_means[col] = float(row[key])

    for col in state.categorical_cols:
        key = f"mode_{col}"
        if key in row:
            value = row[key]
            categorical_modes[col] = None if pd.isna(value) else str(value)

    return ClusterDetail(
        cluster_id=int(cluster_id),
        size=size,
        numeric_means=numeric_means,
        categorical_modes=categorical_modes,
    )
