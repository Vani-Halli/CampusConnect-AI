# CampusConnect AI

**A student-query chatbot for Basaveshwar Engineering College (BEC), Bagalkote**

CampusConnect AI is a React website and FastAPI service that help students ask questions about BEC. The backend searches the project's supplied FAQ records, applies relevance and answerability checks, and returns matching answers with source information where available. Students can also browse categories and the knowledge base, review retrieval evaluation results, and provide answer feedback.

> **Dataset notice:** College information was supplied by the project owner and has not been independently verified. Confirm important or current details with BEC. Public redistribution and display permission for the dataset must be confirmed before publishing its contents or exposing the chatbot publicly.

## Problem statement

Students need a convenient way to find academic and campus information. FAQ information can be incomplete, outdated, or difficult to search. This project provides a searchable question-and-answer interface while making evidence gaps visible instead of intentionally inventing college facts.

## Project objectives

- Let students ask questions in natural language.
- Find relevant information in the supplied BEC FAQ dataset.
- Preserve and display source metadata and verification caveats.
- Offer browsing, categories, feedback, and retrieval evaluation.
- Use a safe fallback when the available evidence is insufficient.

## Features

- Chat interface with TF-IDF, semantic, and hybrid retrieval modes.
- FAQ answerability checks and clarification/fallback responses.
- Related FAQ suggestions, source links, and verification notes when provided.
- Broad FAQ summaries and paginated record browsing.
- Knowledge-base search and category browsing.
- Feedback collection and analytics summary backed by SQLite.
- Retrieval evaluation report view.
- Optional protected PDF upload and indexing API; management requires a backend-only admin key.
- Responsive React interface.

## Technologies

| Area | Technologies |
| --- | --- |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, Lucide React |
| Backend | Python 3.11, FastAPI, Pydantic, Uvicorn |
| Retrieval | scikit-learn TF-IDF, optional Sentence Transformers, hybrid reciprocal-rank fusion |
| Data and storage | JSON FAQ dataset, SQLite/SQLAlchemy, pandas, pypdf |
| Tests | pytest, FastAPI TestClient/httpx |

The standard answer path uses retrieved records and does not require an external LLM. Optional LLM settings exist, but no external LLM API is required to run the application.

## System architecture

```text
Student browser
    │ React / TypeScript / Vite
    │ VITE_API_BASE_URL
    ▼
FastAPI backend
    ├── Loads the project-provided BEC JSON corpus
    ├── Retrieves records with TF-IDF, semantic, or hybrid search
    ├── Checks relevance and topical answerability
    ├── Returns saved FAQ answers, sources, or a safe fallback
    └── Stores feedback and analytics in SQLite
```

The frontend sends chat requests to `POST /api/chat`. Retrieval scores are ranking values, not probabilities or guarantees that an answer is correct. The backend's answerability checks also look at topical evidence. When the evidence does not meet the configured requirements, the API returns a fallback instead of an FAQ answer.

### Retrieval methods

- **TF-IDF:** A keyword-based baseline using term weighting and cosine similarity. It is relatively lightweight and does not need a semantic model download.
- **Semantic:** Uses a Sentence Transformers model to compare query/document embeddings. It can better match paraphrases, but the model must be available to the backend and may need to download on first use.
- **Hybrid:** Combines the TF-IDF and semantic rankings using reciprocal rank fusion. The methods' raw scores are on different scales, so the system fuses ranks rather than treating those scores as directly comparable.

If semantic retrieval cannot be initialized or fails, the retrieval engine can use TF-IDF candidates as a fallback. Model download and cold-start behavior depend on the deployment environment.

## Project structure

```text
CampusConnect-AI/
├── backend/
│   ├── app/
│   │   ├── core/config.py                # Backend settings and paths
│   │   ├── services/                     # Data loading, retrieval, answerability
│   │   ├── data/                          # Local-only legacy demo data (not published)
│   │   ├── database.py                   # SQLite persistence
│   │   └── main.py                       # FastAPI app and API routes
│   ├── data/
│   │   ├── bec_campusconnect_dataset.json # Required BEC FAQ corpus, included in repository
│   │   └── retrieval_evaluation_queries.json
│   ├── reports/                          # Saved evaluation and project reports
│   ├── scripts/                          # Evaluation and data utilities
│   ├── tests/                            # Backend automated tests
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api.ts                        # Typed backend client
│   │   ├── App.tsx                       # Chat and other views
│   │   └── styles.css
│   ├── .env.example                      # Public frontend URL placeholder
│   ├── package.json
│   └── vite.config.ts
├── docs/BEGINNER_GUIDE.md
├── .env.example                          # Backend configuration example
├── .gitignore
├── .python-version
└── README.md
```

The BEC JSON file is included because it is required for the chatbot to start and answer questions. Per the project owner's confirmation, the questions and answers are written in their own words based on information collected from the college's official website; the dataset's source URLs and verification notes are preserved. The local-only legacy demo FAQ file, unrelated downloaded datasets, and its downloader are excluded from Git.

The backend does not silently start with an empty BEC corpus if that file is missing: startup raises a `FileNotFoundError` that identifies the dataset path. If you obtain an updated authorized copy, place it at `backend\data\bec_campusconnect_dataset.json`, or set `BEC_DATASET_PATH` to its path before starting the backend.

## Dataset and limitations

The active BEC FAQ corpus is `backend/data/bec_campusconnect_dataset.json`. It contains 170 records, including questions, answers, categories, and source metadata where supplied. The loader preserves original record fields and generates stable local IDs for records that lack IDs.

- The data was supplied by the project owner, who confirms the question and answer wording is their own paraphrase of material collected from the college's official website. The content is **not independently fact-checked**.
- Source URLs and verification notes are retained, but a source link alone does not establish that information is current or authoritative for a student's decision.
- The dataset includes publicly listed college contact details. Recheck contact information and appropriateness before reusing it outside this project.
- Exact B.E. fee amounts are not available in the supplied records.
- A complete step-by-step B.E. admission procedure is not available.
- Internship information is not supplied.
- Other details can be incomplete, stale, or contradictory.
- FAQ-record counts are not counts of real-world courses, people, offers, or services.
- Students should confirm important, time-sensitive, or conflicting information with the college.

The dataset is included in this repository at the project owner's direction. This project review found no credentials, government-ID-like values, or unusually long answer passages; it is not a legal review or independent verification of every source. Students should confirm important, current information directly with the college.

## Evaluation methodology and results

The evaluation script uses the production retrieval engine and configured thresholds against a manually prepared benchmark of **33 answerable queries and 8 rejection queries**. The report measures Top-1, Top-3, mean reciprocal rank (MRR), answerable acceptance, and rejection accuracy.

Latest saved benchmark results:

| Mode | Top-1 before → after answerability | Top-3 after | MRR after | Rejection accuracy after |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF | 87.9% → 87.9% | 93.9% | 90.4% | 100.0% |
| Semantic | 72.7% → 87.9% | 97.0% | 92.4% | 100.0% |
| Hybrid | 87.9% → **93.9%** | 97.0% | 95.5% | **100.0%** |

These are regression measurements on a small, manually authored benchmark, not a representative sample of student questions and not a guarantee of real-world accuracy. The answerable examples include questions close to FAQ wording, and the rejection sample is small. Results should not be read as a probability that any individual response is correct. See `backend/reports/retrieval_evaluation.md` for the detailed report.

The latest recorded complete backend test run passed **106 tests**. It emitted two dependency deprecation warnings (Starlette and Pydantic); these were warnings, not test failures.

## Requirements

- Windows PowerShell for the commands below (equivalent commands can be used on other systems).
- Python 3.11, as declared in `.python-version`.
- Node.js and npm.
- The included BEC dataset file at the configured path.
- Internet access may be needed to install dependencies and retrieve the optional semantic model.

## Backend setup

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
Copy-Item .env.example .env
```

If PowerShell prevents environment activation, run the virtual environment's Python directly as shown below. The required `backend\data\bec_campusconnect_dataset.json` is included in the repository checkout.

Start the backend from a PowerShell terminal at the project root:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The API health check is `http://127.0.0.1:8000/api/health`. Interactive API documentation is at `http://127.0.0.1:8000/docs`.

## Frontend setup and local run

In a second PowerShell terminal at the project root:

```powershell
Copy-Item frontend\.env.example frontend\.env.local
Set-Location frontend
npm.cmd ci
npm.cmd run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`). The frontend reads the backend origin from `VITE_API_BASE_URL`; its example points to `http://localhost:8000`.

### Environment variables

The root `.env.example` contains backend settings. Important values include:

| Variable | Purpose |
| --- | --- |
| `APP_ENV`, `DEBUG` | Backend environment and debug mode. |
| `BACKEND_HOST`, `BACKEND_PORT` | Local server bind settings. |
| `DATABASE_URL` | SQLite database URL used for feedback and analytics by default. |
| `ALLOWED_ORIGINS` | Comma-separated browser origins allowed by CORS. |
| `BEC_DATASET_PATH` | Dataset path, relative to the backend directory unless absolute. |
| `ANSWER_TFIDF_MIN_SCORE`, `ANSWER_SEMANTIC_MIN_SCORE` | Answerability score thresholds. |
| `ADMIN_API_KEY` | Optional backend-only key for protected document routes. |
| `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL` | Optional LLM integration settings. |

The frontend `.env.example` contains `VITE_API_BASE_URL`. Vite variables are delivered to the browser: never put credentials, API keys, or other secrets in frontend variables. Set `VITE_API_BASE_URL` to the deployed backend origin when one exists. Configure backend `ALLOWED_ORIGINS` with the exact deployed frontend origin; do not use a production wildcard.

## API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Health and corpus status |
| `GET` | `/api/faqs` | Loaded FAQ records |
| `GET` | `/api/categories` | Available categories |
| `GET` | `/api/category-counts` | FAQ record counts by category |
| `GET` | `/api/knowledge` | Search and browse knowledge records |
| `GET` | `/api/evaluation` | Read the saved compatible evaluation report |
| `POST` | `/api/chat` | Retrieve an answer or fallback |
| `POST` | `/api/feedback` | Save answer feedback |
| `GET` | `/api/analytics/summary` | Feedback/query analytics summary |
| `GET` | `/api/documents` | List uploaded documents (protected) |
| `POST` | `/api/documents/upload` | Upload and index a PDF (protected) |
| `DELETE` | `/api/documents/{document_id}` | Delete an uploaded PDF (protected) |

Chat example:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/chat `
  -ContentType 'application/json' `
  -Body '{"question":"How long is the B.E. programme at BEC?","mode":"tfidf"}'
```

The response includes the answer, status, retrieved document IDs, related questions, and source metadata when available. Document-management endpoints require `X-Admin-Key` matching the backend's `ADMIN_API_KEY`. Do not put that key in the frontend.

## Testing and evaluation

Run the full backend suite from the project root:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m pytest -q
```

Build the frontend (includes TypeScript project checks) from the project root:

```powershell
Set-Location frontend
npm.cmd run build
```

Refresh the retrieval evaluation report from the project root:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m scripts.evaluate_retrieval
```

Evaluation writes `backend\reports\retrieval_evaluation.md` and `.json`. Its output describes the loaded model state; semantic results are unavailable if the optional dependency/model cannot load.

## GitHub publication and deployment readiness

The repository includes the active BEC dataset so the backend can run after checkout. The `.gitignore` excludes virtual environments, caches, local environment files, databases, Node dependencies/build output, uploaded files, generated legacy submission output, and unrelated datasets. It preserves `.env.example` files.

Do not commit `.env`, `.env.local`, credentials, the local SQLite database, virtual environments, generated build/cache files, or unapproved data. Keep source code, tests, dependency lock/manifest files, configuration examples, documentation, and useful evaluation reports.

The project is **not deployed**. Deployment preparation is not proof of a public website. Before deployment, configure exact CORS origins and the public frontend backend URL, and consider persistence for SQLite feedback/analytics and uploaded documents. No hosting account or public URL is configured here.

## Known limitations

- The FAQ dataset is small, incomplete, unverified, and may be outdated or contradictory.
- Exact B.E. fees, a full B.E. admission process, and internship details are missing.
- Some source or verification metadata may be absent.
- Similarity and evaluation scores are not confidence probabilities.
- Semantic model downloads can increase startup time or fail; TF-IDF fallback may be used.
- SQLite and uploaded file persistence depend on the hosting filesystem.
- Optional LLM settings are not required for the standard retrieval answer path.

## Future improvements

- Independently verify FAQ records with BEC and update source references as official pages change.
- Expand the benchmark with representative, independently labeled student questions.
- Add persistent production storage if feedback or uploaded documents must survive restarts.
- Monitor retrieval failures and user feedback, then refine only with verified data and regression tests.
- Deploy after reviewing secrets, dataset permissions, environment settings, CORS, and storage requirements.

## Acknowledgements

The BEC FAQ records were supplied by the project owner. No independent dataset license or third-party BEC data attribution is claimed. The application uses the open-source libraries listed above; consult their project licenses when redistributing the software.
