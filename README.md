# Cross-Lingual Prompt-Tuned LLMs for NER in Indigenous Indian Languages

A modular, production-ready system for Named Entity Recognition (NER) in low-resource Indigenous Indian languages (focusing on tribal/regional languages like **Santali, Maithili, and Bhojpuri**) using **Disentangled Soft-Prompt Tuning** ($P_{task} + P_{lang}$) on multilingual transformer backbones (e.g. `google/muril-base-cased` / `xlm-roberta-base` / `ai4bharat/IndicBERTv2-MLM-only`).

---

## 🌟 Key Innovations & Architecture

```
                       +-----------------------------------+
                       |        Input Text Sequence        |
                       | ("बिरसा मुंडा से पटनामें भेंट") |
                       +-----------------------------------+
                                         |
                                         v
                       +-----------------------------------+
                       | Script Normalizer (Unicode/NFKC)  |
                       | Agglutinative Postposition Strip |
                       +-----------------------------------+
                                         |
                                         v
+------------------------+------------------------+-----------------------------------+
|  Task Prompt (P_task)  | Language Prompt (P_lang)|      Word Token Embeddings        |
|  (Anchor NER tuned)    | (Target lang adapted)  |      [E(w_1), E(w_2), ... ]       |
+------------------------+------------------------+-----------------------------------+
                         \           |            /
                          v          v           v
                       +-----------------------------------+
                       | Pretrained Multilingual Backbone  |
                       |  (MuRIL / XLM-R / IndicBERTv2)    |
                       +-----------------------------------+
                                         |
                                         v
                       +-----------------------------------+
                       |  Token Classification Head (IOB)  |
                       |   (PER, ORG, LOC, MISC, DATE)     |
                       +-----------------------------------+
```

### 1. Disentangled Soft-Prompt Formulation
Instead of full fine-tuning, continuous virtual prompt embeddings are prepended to the word embedding sequence:
$$P_{combined} = [P_{task} \,;\, P_{lang}]$$
- **Task Prompt ($P_{task}$)**: Captures generic NER boundary and entity extraction instructions. Tuned on high-resource anchor languages (Hindi/English).
- **Language Prompt ($P_{lang}$)**: Captures language-specific syntax, morphology, and script characteristics of low-resource target languages (Bhojpuri, Maithili, Santali).

### 2. Script Normalization & Agglutinative Postposition Stripping
- **Script Unification**: NFKC Unicode normalization, zero-width space removal, and transliteration mapping (Ol Chiki for Santali, Tirhuta for Maithili to Devanagari/Latin).
- **Agglutinative Postposition Stripping**: Detaches case clitics (`-ने`, `-को`, `-से`, `-का`, `-में`, `-ते`, `-सों`, `-ला`, etc.) attached to nouns so that postpositions do not corrupt entity span boundaries.

---

## 📁 Codebase Directory Structure

```
d:\NLP project\
├── data/
│   ├── __init__.py
│   ├── loader.py         # NER Dataset PyTorch loaders & subword label index shifting
│   ├── normalizer.py     # Unicode script normalizer & agglutinative postposition stripper
│   └── synthetic.py      # Synthetic NER data generator for Bhojpuri, Maithili, Santali, Hindi
├── models/
│   ├── __init__.py
│   ├── soft_prompt.py    # Disentangled soft prompt injection module (P_task + P_lang)
│   └── ner_model.py      # Prompt-tuned Transformer backbone with NER token classification head
├── training/
│   ├── __init__.py
│   └── trainer.py        # Multi-stage prompt training runner (Stage 1: P_task, Stage 2: P_lang)
├── evaluation/
│   ├── __init__.py
│   └── metrics.py        # Span-level seqeval metrics evaluation & boundary alignment checks
├── tests/
│   └── sanity_check.py   # Full test suite verifying forward pass, shapes, loss, and post-processing
├── app.py                # Rich interactive Streamlit visualizer & inference dashboard
├── requirements.txt      # Clean, pinned dependencies
└── README.md             # System documentation & execution guide
```

---

## ⚡ Execution Instructions

### 1. Run Unit Tests & Sanity Checks
Verify forward pass execution, prompt matrix shape concatenation, and span evaluation without errors:
```bash
python tests/sanity_check.py
```

### 2. Launch Interactive Streamlit Web Dashboard
Run the web application to test interactive entity extraction, script normalization, and prompt controls:
```bash
streamlit run app.py
```

---

## 🧪 Multi-Stage Training Protocol

```python
from data.synthetic import SyntheticNERGenerator
from data.loader import create_dataloaders
from models.ner_model import DisentangledPromptNERModel
from training.trainer import MultiStagePromptTrainer
from transformers import AutoTokenizer

# 1. Initialize dataset & model
generator = SyntheticNERGenerator()
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
train_samples = generator.generate_dataset(num_samples=200)

train_loader, val_loader = create_dataloaders(train_samples, tokenizer, batch_size=8, prompt_length=16)

model = DisentangledPromptNERModel(
    model_name_or_path="xlm-roberta-base",
    task_prompt_len=8,
    lang_prompt_len=8
)

# 2. Multi-Stage Trainer
trainer = MultiStagePromptTrainer(model=model, train_loader=train_loader, val_loader=val_loader)

# Stage 1: Tune P_task on anchor dataset
trainer.train_stage(stage=1, epochs=3)

# Stage 2: Tune P_lang on target low-resource dataset
trainer.train_stage(stage=2, epochs=3)

# Save soft prompt checkpoint
trainer.save_prompt_checkpoint("./checkpoints")
```
