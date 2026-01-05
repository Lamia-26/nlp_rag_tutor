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


