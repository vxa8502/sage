# Sage

RAG-powered product recommendation system with explainable AI. Retrieves relevant products from customer reviews, generates natural language explanations grounded in evidence, and verifies faithfulness using hallucination detection.

## Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Recommendation Quality (NDCG@10) | 0.30 | **0.46** |
| Explanation Faithfulness (Claim-Level) | 90% | **97%** |
| Human Evaluation (50 samples) | 3.5/5.0 | **4.19/5.0** |

## Architecture

```
Query → Semantic Search (Qdrant) → Rank Products → Generate Explanation (LLM)
                                                           ↓
                                   Verify Citations ← Retrieve Evidence
                                           ↓
                          Check Faithfulness (HHEM) → Response + Confidence
```

## Tech Stack

- **Embeddings:** E5-small (384-dim, 100% Top-5 accuracy on product reviews)
- **Vector DB:** Qdrant with semantic caching
- **LLM:** Claude Sonnet / GPT-4o-mini
- **Faithfulness:** HHEM (Vectara hallucination detector) + quote verification
- **API:** FastAPI with streaming support

## Quick Start

```bash
# Setup
make setup
source venv/bin/activate

# Start Qdrant and load data
make qdrant-up
make data

# Run demo
make demo

# Start API
make serve
```

## API Example

```bash
curl "http://localhost:8000/recommend?q=wireless+earbuds+for+running&k=3&explain=true"
```

```json
{
  "query": "wireless earbuds for running",
  "recommendations": [{
    "product_id": "B07HKFG85D",
    "score": 0.847,
    "explanation": "Customers praise the secure fit during workouts...",
    "hhem_confidence": 0.94,
    "evidence": [{"id": "review_127", "text": "..."}]
  }]
}
```

## Evaluation

```bash
make eval          # Standard: NDCG, faithfulness, spot-checks
make eval-deep     # Full: ablations, baselines, failure analysis
make human-eval    # Interactive 50-sample evaluation
```

## License

Academic research only (uses Amazon Reviews 2023 dataset).
