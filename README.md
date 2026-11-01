# CampusConnect AI

**An AI-powered college FAQ chatbot that helps students access campus information through natural language queries.**

## Overview

CampusConnect AI is a web-based chatbot developed to help students find information about Basaveshwar Engineering College (BEC), Bagalkote.

Students can ask questions related to admissions, courses, fees, placements, facilities, examinations, and other college-related topics. The chatbot retrieves relevant answers from a structured FAQ dataset using information retrieval techniques.

The project aims to make college information easier to access through a simple, interactive, and user-friendly interface.

## Live Demo

- **Frontend:** https://campus-connect-ai-six-rust.vercel.app
- **Backend API:** https://campusconnect-ai-ehev.onrender.com
- **API Documentation:** https://campusconnect-ai-ehev.onrender.com/docs

## Key Features

- **Natural Language Queries:** Ask college-related questions in simple language.
- **FAQ-Based Retrieval:** Retrieves relevant answers using TF-IDF and other supported retrieval methods.
- **Answerability Checking:** Identifies unsupported or unrelated questions to reduce incorrect responses.
- **Chat History:** Saves and restores previous conversations using browser local storage.
- **Category-Based Information:** Organizes college FAQs into different categories.
- **Responsive Interface:** Supports desktop and mobile screens.
- **Feedback and Analytics:** Supports collecting user feedback and basic usage information.

## Screenshots

<img width="1342" height="626" alt="image" src="https://github.com/user-attachments/assets/805749f9-5d26-484c-bae2-d0cf1db48984" />

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite, Tailwind CSS |
| Backend | Python, FastAPI |
| Information Retrieval | Scikit-learn, TF-IDF |
| Semantic Retrieval | Sentence Transformers (optional) |
| Database | SQLite |
| Data Format | JSON |
| Deployment | Vercel, Render |
| Development Tools | VS Code, Git, GitHub |

## System Architecture

```text
           Student
              |
              v
       React Frontend
              |
              v
         FastAPI Backend
              |
              v
       FAQ Retrieval System
              |
       +------+------+
       |             |
       v             v
   TF-IDF       Answerability
   Retrieval       Check
       |             |
       +------+------+
              |
              v
       Relevant Answer
              |
              v
        Student Interface
```

## How It Works

1. The student enters a question through the chatbot interface.
2. The frontend sends the question to the FastAPI backend.
3. The backend processes the question and searches the structured FAQ dataset.
4. The retrieval system identifies relevant FAQ records.
5. Answerability checks help determine whether the retrieved information is sufficient to answer the question.
6. The chatbot displays the retrieved answer or requests clarification when necessary.

## Dataset

The project uses a structured JSON dataset containing **170 FAQ records across 11 categories**.

Each record contains fields such as:

- Question
- Answer
- Category
- Institution
- Source URL
- Academic year
- Verification notes

The dataset is maintained in:

`backend/data/bec_campusconnect_dataset.json`

The information is intended for academic project use. Some entries may be incomplete or require verification against official college sources.

## Retrieval Evaluation

The retrieval system was evaluated using a manually prepared benchmark of 33 answerable queries and 8 unanswerable queries.

| Metric | TF-IDF | Semantic | Hybrid |
|---|---:|---:|---:|
| Top-1 Accuracy | 87.9% | 87.9% | 93.9% |
| Top-3 Accuracy | 93.9% | 97.0% | 97.0% |
| Mean Reciprocal Rank (MRR) | 90.4% | 92.4% | 95.5% |
| Rejection Accuracy | 100% | 100% | 100% |

**Note:** These results are based on a small manually prepared evaluation dataset and should not be considered a guarantee of real-world performance. The deployed free Render service currently uses TF-IDF fallback when the semantic model is unavailable.

## Installation and Setup

### Prerequisites

- Python 3.11
- Node.js and npm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Vani-Halli/CampusConnect-AI.git
cd CampusConnect-AI
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
```

Activate the virtual environment.

**Windows:**

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
uvicorn app.main:app --reload
```

Backend runs at:

`http://127.0.0.1:8000`

### 3. Frontend Setup

Open another terminal:

```bash
cd frontend
npm install
```

Create a `.env` file in the frontend directory:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Start the frontend:

```bash
npm run dev
```

Open the local URL displayed in the terminal.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Backend health check |
| POST | `/api/chat` | Submit a chatbot question |
| GET | `/api/categories` | Retrieve FAQ categories |
| GET | `/api/category-counts` | Retrieve category counts |
| GET | `/api/knowledge` | Access knowledge information |
| POST | `/api/feedback` | Submit user feedback |

Interactive API documentation is available at `/docs`.

## Testing

The backend test suite contains **106 passing tests**.

Testing covers:

- FAQ dataset loading and validation.
- Question retrieval and answerability checks.
- Chat API functionality.
- Feedback and analytics functionality.
- Error handling and application behavior.

## Limitations

1. **Dataset Coverage:** The chatbot can answer only questions covered by the available FAQ dataset.
2. **Information Verification:** Some college information may be incomplete, outdated, or require official verification.
3. **Retrieval Accuracy:** Similar or ambiguous questions may sometimes return less relevant answers.
4. **Chat History:** Conversation history is stored locally in the browser and is not synchronized across devices.
5. **Free Hosting Constraints:** The backend may take time to respond after inactivity, and temporary server storage may not retain feedback or uploaded data after restarts.

## Future Improvements

1. **Dataset Expansion:** Add more verified FAQs covering additional student queries.
2. **Improved Retrieval:** Improve retrieval accuracy using larger datasets and more diverse evaluation queries.
3. **Regular Data Updates:** Introduce a process to keep college information accurate and up to date.
4. **Cloud-Based Chat History:** Add optional student accounts and synchronized chat history across devices.
5. **Admin Dashboard:** Provide an authorized interface to manage and update FAQ records.

## Project Structure

```text
CampusConnect-AI/
│
├── backend/
│   ├── app/
│   ├── data/
│   │   └── bec_campusconnect_dataset.json
│   ├── tests/
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
│
└── README.md
```

## Author

**Vani Nagappa Halli**

Computer Science and Engineering

- GitHub: https://github.com/Vani-Halli
- Portfolio: https://vani-halli.github.io/profile/

---

*Developed as an academic minor project to make college information more accessible through AI-based information retrieval.*
