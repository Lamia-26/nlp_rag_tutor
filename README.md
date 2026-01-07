# NLP RAG Tutor 🤖

An intelligent tutor based on a **RAG (Retrieval-Augmented Generation)** architecture, designed to answer natural language questions from NLP course materials.

This project was developed as part of the **Natural Language Processing** course and implements techniques studied during the labs: embeddings, retrieval, FAISS, LLMs, and evaluation.

---

##  Project Objective

The goal of this project is to build a text-based AI service capable of:

- ingesting pedagogical documents (PDFs),
- retrieving relevant passages for a given question,
- generating clear, pedagogical, and **source-grounded** answers,
- quantitatively evaluating the quality of the retrieval component.

The system acts as an **NLP tutor**, helping students better understand key concepts of the field.

---

##  Corpus

- **Single source**: *Speech and Language Processing* — Jurafsky & Martin  
- Language: English  
- Format: PDF  
- Size: ~600 pages  

Using a single, dense, and authoritative source helps reduce hallucinations and ensures answer reliability.

---

##  Architecture (RAG)

```text
                ┌───────────────┐
                │     PDF       │
                │  (course)     │
                └───────┬───────┘
                        │
                 Ingestion & Cleaning
                        │
                ┌───────▼───────┐
                │     Pages     │
                └───────┬───────┘
                        │
                  Chunking
           (overlap + max length)
                        │
                ┌───────▼───────┐
                │    Chunks     │
                └───────┬───────┘
                        │
                  Embeddings
          (Sentence-Transformers)
                        │
                ┌───────▼───────┐
                │     FAISS     │
                │  Vector Index │
                └───────┬───────┘
                        │
User Question ───────────┘
                        │
                  Semantic Search
                        │
                ┌───────▼───────┐
                │  Retrieved    │
                │   Context     │
                └───────┬───────┘
                        │
                    Prompting
                        │
                ┌───────▼───────┐
                │      LLM      │
                │    (Groq)     │
                └───────┬───────┘
                        │
                   Final Answer
                 (with sources)

---

## Main components

- PDF ingestion and text cleaning

- Text chunking with overlap

- Semantic embeddings (Sentence-Transformers)

- Vector indexing with FAISS

- Answer generation using an LLM (Groq)

- Quantitative evaluation of retrieval quality

---

##  Installation

1. Clone the repository
```bash
git clone https://github.com/Lamia-26/nlp_rag_tutor/
cd nlp-rag-tutor
```

2. Install dependencies

Dependencies are defined in `pyproject.toml`.

```bash
python -m pip install -e .
```

---

##  Usage

1. Ingest the PDF
```bash
python -m src.main ingest
```

2. Chunk the documents
```bash
python -m src.main chunk --max_chars 2500 --overlap_chars 300
```

3. Build the vector index
```bash
python -m src.main index --embed_model sentence-transformers/all-MiniLM-L6-v2
```

4. Ask a question
```bash
python -m src.main ask "Explain TF-IDF simply, with an example."
```

---

##  Evaluation

The retrieval system is evaluated on a set of 40 NLP-related questions.

```bash
python -m src.main evaluate --top_k 8 --out_dir data/eval/run_001
```

### Results

- Recall@8: 0.95
- MRR: 0.88

These metrics show that relevant passages are almost always retrieved among the top results.

---

##  Iterative Approach

An iterative experimentation process was followed and is documented in:

`notebooks/iterative_rag_notebook.ipynb`

This notebook demonstrates:

- the impact of chunk size and overlap,

- comparison of embedding models,

- prompt refinement to reduce hallucinations,

- quantitative and qualitative analysis of improvements.

- small to big tested 

---

##  Limitations and Future Improvements

### Current limitations

- Sensitivity to question phrasing

- Imperfect PDF text extraction

- No cross-encoder re-ranking

### Potential improvements

- Chapter-aware chunking

- Query expansion

- Neural re-ranking

- Web-based interface

---

##  Conclusion

This project demonstrates that a well-designed RAG system, even when based on a single document, can provide reliable, pedagogical, and source-grounded answers.
It fully meets the objectives of the NLP course and serves as a strong foundation for intelligent tutoring systems.
