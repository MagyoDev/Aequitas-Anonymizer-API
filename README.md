# Aequitas Anonymizer API

A **Aequitas Anonymizer API** implementa um sistema completo de **anonimização de dados pessoais**, baseado em:

* **Clusterização** de registros semelhantes
* **Respostas agregadas** (nunca dados individuais)
* **k-anonymity** (mínimo de 10 pessoas)
* **Supressão automática de atributos sensíveis**
* **Bloqueio de consultas muito pequenas (<10)**
* **Bloqueio de consultas muito grandes (>4000)**
* **Consultas cruzadas multi-atributo**
* **Política de privacidade alinhada à LGPD**

O objetivo é demonstrar como aplicar técnicas de Machine Learning e estatística para reduzir o risco de identificação de pessoas, entregando apenas **informações agregadas e privadas**.

---

# Objetivos do projeto

* Evitar reidentificação de indivíduos mesmo após **cruzamento de atributos**.
* Demonstrar o uso de **k-anonymity** e limites estatísticos.
* Aplicar clusterização para formar **grupos homogêneos** e sumarizar dados.
* Expor apenas estatísticas seguras — nunca dados brutos.
* Atender padrões de privacidade recomendados por órgãos como GDPR, LGPD, NIST e Google DP.

---

# Tecnologias utilizadas

* **Python 3.11**
* **FastAPI**
* **Pandas**
* **Scikit-Learn**
* **Uvicorn**
* **GitHub Actions (CI/CD)**
* **Docker**

---

# Estrutura do projeto

```text
anonymizer-ml-api/
├── app/
│   ├── __init__.py
│   ├── main.py          # Endpoints e regras de privacidade
│   ├── config.py        # Configurações, caminhos, k-anonymity, limites
│   └── services.py      # Lógica de ML (pré-processamento, clusterização, agregação)
├── data/
│   └── AequitasDB.csv   # Dataset (não versionar dados sensíveis)
├── requirements.txt
├── Dockerfile           # Build da imagem da API
├── .github/
│   └── workflows/
│       └── ci-cd.yml    # Pipeline de CI/CD com safe startup
├── README.md
└── .gitignore
```

---

# Instalação e uso

## 1. Criando o ambiente virtual

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# ou
source .venv/bin/activate  # Linux/Mac
```

## 2. Instalando dependências

```bash
pip install -r requirements.txt
```

## 3. Colocando os dados

Coloque o arquivo:

```
data/AequitasDB.csv
```

O projeto extrai automaticamente:

* apenas atributos permitidos
* cria a coluna **CIDADE** extraída de `END_RESIDENCIAL`
* remove todos os atributos sensíveis antes da clusterização

---

# Executando a API

```bash
uvicorn app.main:app --reload
```

Endpoints disponíveis em:

* Swagger → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Redoc → [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

# Política de Privacidade implementada

A API aplica **três camadas de proteção**:

### ✔ 1. k-Anonymity

Nenhuma resposta é exibida se houver **menos de 10 registros**.

### ✔ 2. Limite Máximo de Resultado

Consultas que retornam **mais de 4000 registros** são bloqueadas.

### ✔ 3. Supressão de Atributos Sensíveis

Nunca usados em clusterização nem exibidos:

* CPF
* RG
* CNH
* PASSAPORTE
* TELEFONE
* ENDEREÇO completo
* NOME nas features de clusterização

### ✔ 4. Apenas estatísticas agregadas

Nenhuma linha individual é retornada em nenhum endpoint.

---

# Endpoints principais

---

## 1. `POST /fit`

Recarrega o CSV e re-treina o modelo de clusterização.

Request:

```json
{ "n_clusters": 12 }
```

Response:

```json
{
  "num_records": 100000,
  "num_clusters": 12,
  "k_anonymity": 10,
  "max_results": 4000
}
```

---

## 2. `GET /stats/nome/{nome}`

Retorna contagem agregada de pessoas com determinado nome, respeitando:

* k-anonymity (<10 bloqueia)
* limite máximo (>4000 bloqueia)

Exemplo:

```json
{
  "name": "Juan",
  "count": 110,
  "anonymized": true,
  "message": "Existem 110 pessoas com o nome Juan."
}
```

Se count < 10:

```json
{
  "name": "Alan",
  "count": 0,
  "anonymized": true,
  "message": "Consulta bloqueada por privacidade (k-anonymity)."
}
```

---

## 3. `GET /stats/multi`

Consulta cruzada multi-atributo:

Parâmetros disponíveis:

* `nome`
* `idade`
* `sexo`
* `ocupacao`
* `cidade` (extraída automaticamente)

Exemplo:

```
/stats/multi?idade=45&sexo=F&ocupacao=Professora
```

Response:

```json
{
  "filters": {
    "idade": 45,
    "sexo": "F",
    "ocupacao": "Professora"
  },
  "count": 512,
  "anonymized": true,
  "message": "Existem 512 pessoas que atendem a esses atributos."
}
```

Se count > 4000:

```json
{
  "filters": {...},
  "count": 0,
  "anonymized": true,
  "message": "Consulta bloqueada por exceder o limite máximo permitido (mais de 4000 resultados)."
}
```

---

## 4. `GET /clusters`

Lista clusters com seus tamanhos, ocultando clusters < 10:

```json
[
  { "cluster_id": 0, "size": 9341 },
  { "cluster_id": 1, "size": 8812 }
]
```

---

## 5. `GET /clusters/{cluster_id}`

Retorna agregados de um cluster:

```json
{
  "cluster_id": 0,
  "size": 9341,
  "numeric_means": {
    "IDADE": 39.6
  },
  "categorical_modes": {
    "SEXO": "F",
    "OCUPACAO": "Analista de Marketing",
    "CIDADE": "São Paulo"
  }
}
```

Clustes menores que 10 → bloqueio automático.

---

# Como a API garante anonimização

* Nunca retorna uma linha individual.
* Remove todos os atributos sensíveis.
* Aplica **k-anonymity** (mínimo 10).
* Bloqueia consultas com **>4000** registros.
* Responde com **médias e modas**, não valores individuais.
* Clusterização reforça o agrupamento natural.
* Cidade extraída reduz risco ao usar endereço completo.
* CI/CD evita clusterização no ambiente de testes (rápido e seguro).

---

# Segurança, LGPD e riscos mitigados

Esta API reduz riscos de:

* Reidentificação por cruzamento de dados.
* Inferência reversa de atributos sensíveis.
* Ataques de linkagem.
* Engenharia reversa de perfis individuais.
* Ataques de reconstrução do dataset.

Cumpre princípios de:

* LGPD — anonimização, minimização e finalidade
* GDPR — data minimization, purpose limitation
* NIST — k-anonymity, suppression
* Google DP — result-size limits

---

# CI/CD

Inclui:

* Workflow GitHub Actions completo
* Modo CI com clusterização desativada
* Teste automatizado do servidor
* Lint (flake8)
* Build e push Docker (se configurado)

---

# Executando via Docker

Build:

```bash
docker build -t aequitas-anonymizer .
```

Run:

```bash
docker run -p 8000:8000 aequitas-anonymizer
```
