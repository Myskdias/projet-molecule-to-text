# Molecule-to-Text

This repository contains our work for the **ALTEGRAD / MVA 2026 Kaggle project** on molecular graph captioning.

**Full report:** [view the report in the repository](./Altegrad.pdf) or [download the PDF](https://raw.githubusercontent.com/Myskdias/projet-molecule-to-text/Best_so_far/Altegrad.pdf).

The goal of the project was to generate a natural language description of a molecule from its molecular graph representation. Each molecule is represented as a graph: nodes correspond to atoms and edges correspond to chemical bonds. The task is therefore a multimodal graph-to-text problem, where the model must understand a structured molecular representation and produce a chemically meaningful textual caption.

## Project context

The dataset contains approximately **33,000 molecule-text pairs**:

- around **32,000** training and validation samples, with both the molecular graph and the reference textual description;
- around **1,000** test molecules, provided only as graphs and evaluated through the Kaggle leaderboard.

The task is difficult because the dataset is small compared to the complexity of the problem. Molecular captioning requires both structural chemical understanding and fluent language generation. Training a large generative model end-to-end on such limited data can easily lead to overfitting, while a small model trained from scratch lacks the linguistic knowledge needed to produce fluent captions.

The molecular graphs are also rich: atoms are described by several categorical attributes, such as atomic number, chirality, degree, formal charge, number of hydrogens, hybridization, aromaticity, and ring membership. Bonds are also described by categorical features such as bond type, stereochemistry, and conjugation.

## Evaluation

The competition used a composite evaluation based on two complementary NLP metrics:

- **BLEU-4**, which measures n-gram overlap between the generated caption and the reference description;
- **BERTScore**, based on contextual embeddings, which measures semantic similarity between the generated and reference captions.

Good predictions therefore need to be both lexically close to the reference caption and semantically correct.

## Main approaches

We explored several directions, including a lightweight Transformer trained from scratch, a naive CLIP + RAG + language-model approach, and rule-based editing strategies. The final work focused on two stronger approaches:

1. **End-to-end graph-to-sequence generation with GINE + T5**
2. **Retrieval-based graph-to-text generation with reranking and soft editing**

---

## Method I: End-to-End Graph-to-Sequence Generation

The first main approach treats the task as a translation problem: the source representation is the molecular graph, and the target sequence is the molecular description.

### Graph encoder: GINE

We use a deep **Graph Isomorphism Network with Edge Features (GINE)** as the structural encoder. GINE is well suited for molecular graphs because it explicitly incorporates edge attributes during message passing.

Atoms and bonds are encoded through multi-feature embeddings. Instead of representing each atom or bond with a single feature vector, we embed each categorical property separately and combine these embeddings. This allows the model to use detailed chemical information without creating extremely large sparse input vectors.

The graph encoder uses residual connections and normalization to preserve structural information and stabilize training across several message-passing layers.

### T5 decoder with graph conditioning

The graph encoder produces node-level molecular embeddings. These embeddings are projected into the hidden dimension expected by **T5-Small** and used to condition a pretrained T5 decoder.

This approach is inspired by soft prompting: the molecular representation is injected into the language model as continuous embeddings. T5 can therefore use its pretrained linguistic knowledge while still being guided by the target molecular graph.

### Position-aware weighted loss

A key issue in molecular captioning is the molecule name. The beginning of the caption often contains the IUPAC name or a highly specific molecular identifier. If the model generates the wrong name, the rest of the caption may still sound plausible while describing the wrong molecule.

To address this, we used a **position-aware weighted loss** that gives more importance to the first tokens of the generated sequence. This encourages the model to focus on correctly generating the most chemically specific part of the caption.

### Metric-based checkpointing

Instead of selecting the best model only based on training loss, we used validation generation metrics. At the end of each epoch, captions are generated on the validation set and the checkpoint is selected based on BLEU performance. This is more aligned with the Kaggle objective than raw language-modeling loss.

---

## Method II: Retrieval-Based Graph-to-Text with Soft Editing

The second main approach frames molecular captioning as a retrieval problem. This turned out to be particularly effective in the limited-data setting.

### Graph-text contrastive retrieval

We train a graph-text alignment model inspired by CLIP. Molecular graphs and textual captions are embedded into a shared latent space. The model is trained with a contrastive objective so that each molecular graph is close to its matching caption and far from other captions in the batch.

For graph encoding, we again use a GINE-based architecture with rich atom and bond features. For text encoding, we use a pretrained sentence encoder. The text encoder is kept frozen, which makes training much faster and reduces overfitting.

### Multi-layer graph pooling

Rather than using only the final graph representation, we use a multi-layer pooling strategy. Graph-level embeddings are extracted from multiple GINE layers and combined. Early layers capture local chemical environments, while deeper layers capture more global molecular structure.

This avoids explicit graph coarsening methods that may discard atom- and bond-level details, which are important for chemistry.

### Retrieval and reranking

At inference time, the query molecule is embedded into the shared graph-text space. The system retrieves the nearest candidate captions from the training set and then reranks them.

Several reranking strategies were explored:

- graph-text similarity reranking;
- lexical centroid reranking;
- hybrid reranking combining graph-based and text-based scores;
- MBR-style reranking;
- distance-based pruning of weak candidates.

The strongest variant was a hybrid reranking method combined with pruning.

### Graph-guided soft editing

Retrieved captions are often globally correct but may contain local inconsistencies, especially for chemical properties such as charge, ion status, or conjugate acid/base terminology.

To correct these errors, we apply a conservative **soft editing** step. This step uses information from the target graph, such as formal charges, to fix specific high-impact terms like:

- `anion`
- `cation`
- `zwitterion`
- `conjugate acid`
- `conjugate base`

This editing is deliberately limited: it does not rewrite the whole caption, but only corrects targeted lexical inconsistencies. This helps improve BLEU while preserving the fluency of retrieved human-written captions.

## Results

The strongest approaches were the two final methods:

| Method | Kaggle Score |
| --- | ---: |
| Rule-based editing | ≈ 0.576 |
| Simple CLIP + RAG + LLM | ≈ 0.583 |
| From-scratch Transformer | ≈ 0.601 |
| GINE + T5 | ≈ 0.618 |
| RAG + Soft Editing | ≈ 0.619 |

The retrieval-based method with soft editing achieved the best score. This suggests that, in a limited-data regime, retrieval can be more robust than fully generative modeling. The retrieved captions provide fluent and realistic molecular descriptions, while graph-guided soft editing corrects important local chemical errors.

The GINE + T5 approach also performed well. It benefited from the pretrained linguistic knowledge of T5 and avoided the grammatical issues observed with the from-scratch Transformer.

## Main challenges

The main challenge was the small size of the dataset relative to the complexity of the task.

Molecular captioning requires:

- understanding detailed graph structure;
- capturing atom and bond attributes;
- generating fluent English;
- correctly naming molecules;
- preserving chemical consistency;
- optimizing both lexical overlap and semantic similarity.

Because the dataset is limited, purely end-to-end training is unstable. Large language models can overfit, while smaller models trained from scratch often lack fluency. This motivated our use of pretrained models, contrastive graph-text alignment, retrieval, reranking, and lightweight graph-guided editing.

## Repository structure

```text
.
├── architecture.py          # Model architectures
├── data_utils.py            # Dataset and graph batching utilities
├── retrieval.py             # Retrieval index and nearest-neighbor search
├── training_pipeline.py     # Training pipeline for graph-text alignment and generation
├── test.ipynb               # Experiments and analysis
└── Altegrad.pdf             # Full project report
```

## Conclusion

This project explores molecular graph captioning under strong data constraints. We found that combining graph neural networks with pretrained language models improves generation quality, while retrieval-based methods are especially effective when training data is limited.

The best-performing system uses a GINE graph encoder, contrastive graph-text retrieval, hybrid reranking, and graph-guided soft editing. This approach combines the fluency of existing molecular captions with targeted corrections based on the target molecular graph.
