# Aequitas Anonymizer API

Este projeto implementa uma API de anonimização de dados pessoais baseada em **clusterização** e **respostas agregadas**.

Em vez de devolver dados individuais (linhas do dataset), a API:
- agrupa registros semelhantes em **clusters**;
- aplica uma política de **k-anonymity** (grupo mínimo de registros);
- responde apenas com informações agregadas, como:

> "Existem 110 pessoas com o nome Juan."

## Objetivo

Demonstrar na prática como aplicar técnicas de Machine Learning (especialmente clusterização) para:
- reduzir o risco de identificação de indivíduos;
- expor apenas estatísticas agregadas;
- apoiar requisitos de privacidade e LGPD.

## Tecnologias

- Python
- FastAPI
- Pandas
- Scikit-learn

## Estrutura do projeto

```text
anonymizer-ml-api/
├── app/
│   ├── __init__.py
│   ├── main.py          # Endpoints da API
│   ├── config.py        # Configurações, K-anonymity, caminho do CSV
│   └── services.py      # Lógica de ML (pré-processamento, clusterização, agregação)
├── data/
│   └── AequitasDB.csv   # Dataset com os dados pessoais (não versionar no Git se forem dados sensíveis)
├── requirements.txt
├── README.md
└── .gitignore
```

## Preparando o ambiente

1. Crie e ative um ambiente virtual:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows (PowerShell ou CMD)
   # ou
   source .venv/bin/activate  # Linux / Mac
   ```

2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Coloque o arquivo `AequitasDB.csv` dentro da pasta `data/`.

   - Ajuste o caminho em `app/config.py` se necessário.
   - O projeto supõe que exista (por exemplo) uma coluna `nome` para o endpoint de estatísticas de nomes.

## Executando a API

Dentro da pasta raiz do projeto (`anonymizer-ml-api`), execute:

```bash
uvicorn app.main:app --reload
```

A API ficará disponível em:

- Documentação interativa (Swagger): http://127.0.0.1:8000/docs
- Documentação alternativa (ReDoc): http://127.0.0.1:8000/redoc

## Endpoints principais

### 1. `POST /fit`

Recarrega o CSV e re-treina o modelo de clusterização.

Request (opcional):

```json
{
  "n_clusters": 12
}
```

Response (exemplo):

```json
{
  "num_records": 1000,
  "num_clusters": 12,
  "k_anonymity": 10
}
```

### 2. `GET /stats/nome/{nome}`

Retorna uma resposta agregada para um nome específico, respeitando k-anonymity.

Exemplo de resposta:

```json
{
  "name": "Juan",
  "count": 110,
  "anonymized": true,
  "message": "Existem 110 pessoas com o nome Juan."
}
```

Se a contagem for muito pequena (menor que `k_anonymity`):

```json
{
  "name": "Alan",
  "count": 0,
  "anonymized": true,
  "message": "Os dados para esse nome existem, mas não podem ser exibidos por questão de privacidade (k-anonymity)."
}
```

### 3. `GET /clusters`

Lista os clusters disponíveis com seus tamanhos (apenas agregados, ocultando clusters menores que `k_anonymity`).

```json
[
  { "cluster_id": 0, "size": 210 },
  { "cluster_id": 1, "size": 180 },
  { "cluster_id": 2, "size": 95 }
]
```

### 4. `GET /clusters/{cluster_id}`

Mostra detalhes agregados de um cluster:

- tamanho
- médias de colunas numéricas
- categorias mais frequentes (moda) de colunas categóricas

Exemplo:

```json
{
  "cluster_id": 0,
  "size": 210,
  "numeric_means": {
    "idade": 29.4,
    "renda": 3200.7
  },
  "categorical_modes": {
    "cidade": "São Paulo",
    "profissao": "Analista"
  }
}
```

Se o cluster tiver menos registros que `k_anonymity`, o acesso é negado.

## Como isso garante anonimização

- Nenhuma chamada retorna uma linha individual do dataset.
- Contagens pequenas são suprimidas (k-anonymity).
- Os valores exibidos são sempre estatísticas agregadas.
- A lógica pode ser estendida com técnicas adicionais (por exemplo, ruído diferencial).

## Possíveis extensões

- Uso de embeddings de texto para atributos como nome, profissão, etc.
- Trocar KMeans por DBSCAN ou HDBSCAN.
- Persistência dos modelos em disco.
- Autenticação na API.
- Auditoria de consultas para evitar ataques de reconstrução.
