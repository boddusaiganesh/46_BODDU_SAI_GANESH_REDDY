# Evaluation & Guardrails Documentation

## 1. Quality Assurance Strategy
In the financial domain, hallucinations are unacceptable. Our approach prioritizes **Precision** and **Factuality** over creativity. We employ a multi-layered evaluation strategy:

1.  **Deterministic Checks**: Validates numbers against the source of truth (Dataframe).
2.  **Citation Verification**: Ensures claims are grounded in retrieved context.
3.  **Semantic Analysis**: Uses LLM-as-a-Judge to assess coherence and tone.

## 2. Guardrails Framework (`src/evaluator.py`)

Every generated MD&A draft is passed through the `MDAGuardrails` class. A report is flagged if it fails any critical check.

| Guardrail Check | Method | Target | Description |
| :--- | :--- | :--- | :--- |
| **Factual Consistency** | `check_factual_consistency` | >95% | Extracts all numerical entities from the text and attempts to fuzzy-match them to the source dataframe. A tolerance of ±5% is allowed for rounding. |
| **Citation Coverage** | `check_citations` | 100% | Checks if every section containing claims has at least one reference to a source chunk. |
| **Metric Accuracy** | `check_metric_calculations` | Pass/Fail | Re-calculates percentage changes (e.g., YoY Growth) from the mentioned raw numbers to ensure the math in the text is correct. |
| **Financial Reasonableness** | `check_financial_reasonableness` | Pass/Fail | Detecting logic errors, such as: <br>- Profit Margin > 100% <br>- Negative Revenue <br>- Operating Expenses < 0 |
| **Tone & Style** | `check_tone` | Professional | LLM-based check to ensure the language is objective (no marketing fluff) and professional. |

## 3. Metrics & Scoring

We calculate a composite **Quality Score** (0-100%) for each report:

$$ Score = (0.4 \times Factuality) + (0.3 \times Completeness) + (0.2 \times Citations) + (0.1 \times Tone) $$

- **Factuality**: Percentage of matched numbers.
- **Completeness**: Ratio of required sections present (5 total).
- **Citations**: Density of citations per paragraph.

## 4. Testing Methodology

### Unit Tests
Located in `tests/`:
- `test_kpi_calculator.py`: Verifies financial math accuracy on edge cases (divide by zero, missing data).
- `test_data_loader.py`: Ensures SEC JSONs are parsed correctly.

### Integration Tests
- **RAG Latency**: We measure end-to-end generation time. To optimize this, we use local embeddings (`all-MiniLM-L6-v2`) which reduced indexing time by 60%.
- **Rate Limits**: The system is stress-tested against Gemini's 15 RPM limit to verify the Model Rotation logic works.

## 5. Known Limitations

1.  **PDF Tables**: Currently does not support parsing PDF annual reports directly; relies on structured JSON/XBRL or CSV.
2.  **Complex Adjustments**: Does not handle non-GAAP adjustments automatically unless explicitly provided in the input CSV.
3.  **Context Window**: Extremely long filings (10-Ks) are chunked, which can sometimes break long-range context across widely separated footnotes.

## 6. Future Improvements

- [ ] Implement Table-to-Text specialized models (e.g., TAPAS) for better tabular reasoning.
- [ ] Add support for XBRL-JSON direct mapping.
- [ ] Deploy a "Human-in-the-Loop" review step in the Streamlit UI.
