# Austrian Tax QA with a Small LLM: Inference vs. Fine-Tuning vs. RAG

How well does a small open model answer questions on Austrian tax law, and does fine-tuning or retrieval help? This project compares three pipelines built on the same 0.5B backbone, all running locally without any API.

Built for the course *4805 Applications of Data Science: Large Language Models* at WU Vienna (summer term 2026). The original submission is part of the course repository [svakulenk0/wu-llms-ss26](https://github.com/svakulenk0/wu-llms-ss26).

## Results

All three models answered the same 643 German questions. Answers were scored against reference solutions with BERTScore (`bert-base-multilingual-cased`).

| Model | Precision | Recall | F1 |
|---|---|---|---|
| 1 — Plain inference | 0.6381 | **0.6795** | **0.6574** |
| 2 — Fine-tuned | 0.6178 | 0.6619 | 0.6385 |
| 3 — RAG | **0.6475** | 0.6590 | 0.6525 |

- **Plain inference wins on F1**, but only narrowly ahead of RAG.
- **RAG has the best precision.** When retrieval finds the right provision, answers stay closer to the reference. When it misses, answers get short or incoherent.
- **Fine-tuning performed worst.** The model drifted into generic template phrasing ("Grundsätzlich gilt …"), so the synthetic training data did not transfer well to the test questions.
- **Common failure modes** across all models: legal hallucination (including switching from Austrian to Swiss law), prompt leakage such as "Antwort:" in the output, and truncated answers.

The full write-up is in [`report/report.pdf`](report/report.pdf).

## Approach

**Backbone:** [`ipst/Qwen2.5-0.5B-Instruct-SLDS`](https://huggingface.co/ipst/Qwen2.5-0.5B-Instruct-SLDS), a Qwen2.5-0.5B-Instruct variant fine-tuned on Swiss court decision summaries. It knows legal language but is not specific to Austrian tax law. All pipelines decode greedily (`temperature = 0`).

**Test set:** 643 questions across ten areas, including income tax (EStG), corporate tax, domestic and international VAT, employment income, self-employment, real estate transfer tax and international tax.

| Pipeline | What it does | Key settings |
|---|---|---|
| [Model 1 — Inference](model_1_inference/) | Uses the backbone as is | 256 input tokens, 160 new tokens |
| [Model 2 — Fine-tuning](model_2_fine_tune/) | Supervised fine-tuning on 2,700 synthetic Austrian tax Q&A pairs (8 focus areas, 90/10 train/validation split) | LoRA r=8, α=16 planned; 2 epochs, lr 2e-4 |
| [Model 3 — RAG](model_3_rag/) | Fetches 33 provisions of EStG, UStG and other laws from the official Austrian legal database (RIS) and retrieves them by lexical token overlap | top-k = 4, ~5,200 characters of context, 1,024 input tokens |

**Note on Model 2:** `peft` was not installed during the recorded run, so the notebook fell back to training the base model directly instead of a LoRA adapter (batch size 1, gradient accumulation 16).

## Repository structure

```
model_1_inference/    Notebook, test questions and generated answers
model_2_fine_tune/    Notebook, training data, test questions and generated answers
model_3_rag/          Notebook, RIS source list, test questions and generated answers
evaluation/           BERTScore notebook, summary and per-question scores
report/               Final report (PDF)
```

Each model folder is self-contained: open the notebook from inside its folder and it finds its input files there. `submission.csv` holds the final answers and `submission_checkpoint.csv` the intermediate checkpoint used to resume interrupted runs.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook
```

- Models 1 and 3 run on CPU, but a GPU makes them much faster. Model 3 needs internet access to download the legal texts from RIS.
- Model 2 can require a GPU for training. Install `peft` to train a LoRA adapter as intended.
- The evaluation notebook needs the reference answers (`solutions.csv`) from the course, which are not included here.

## Data

- `dataset_clean.csv`: test questions provided by the course.
- `train_tax_qa.csv`: 2,700 synthetic question–answer pairs with focus area, legal basis and source.
- `rag_corpus.csv`: list of RIS sources (law, paragraph, URL) used for retrieval.
