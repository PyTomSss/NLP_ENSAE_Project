# A Multi-Label Dataset of French Fake News

This NLP project was developed as part of the *Machine Learning for NLP* course during the second year of the Master’s program at ENSAE. It builds upon the work of Icard et al. (2024), *A Multi-Label Dataset of French Fake News: Human and Machine Insights*, and leverages the accompanying [OBSINFOX GitHub repository](https://github.com/icard-obsinfox/obsinfox), which provides both the dataset and extensive documentation.

## Project Overview

The primary objective of this project is to explore how interpretable, manually constructed features—alongside outputs from large pre-trained language models—can be used to detect and characterize manipulative news content in French. This hybrid approach seeks to bridge the gap between human annotation and machine perception by leveraging the informative richness and multidimensional nature of the OBSINFOX dataset.

We propose a pipeline that combines:

* Structural feature engineering (e.g., word counts, lexical diversity),
* Contextual embeddings and classification scores from models pre-trained on related tasks (e.g., sentiment analysis, NLI),
* Analytical comparisons between human annotator judgments and model predictions.

This dialogue between human insights and automated analysis aims to identify textual patterns suggestive of factual manipulation or editorial bias.

---

## Dataset — OBSINFOX

The OBSINFOX dataset includes:

* **100 French-language news articles** from 17 sources flagged as unreliable by organizations like *NewsGuard* and *Conspiracy Watch*.
* **11 multi-label annotations** per article, covering a wide range of stylistic, rhetorical, and factual phenomena (e.g., exaggeration, conspiracy, emotional tone, decontextualization).
* **Metadata** such as article title, annotator ID, and source URL.

These labels were assigned by a panel of **8 expert annotators**, ensuring a high level of annotation quality and intersubjective reliability.

---

## Why This Dataset Matters

Fake news is rarely defined by outright falsehoods alone. Instead, it often operates through more subtle mechanisms: exaggeration, selective framing, decontextualization, or emotional manipulation. OBSINFOX is uniquely positioned to support research in this direction thanks to:

* **Multi-dimensional annotations** that go beyond binary "true/false" labeling,
* **Expert human annotation**, which enables nuanced and linguistically-informed analysis,
* **Multi-label structure**, suitable for both supervised learning and exploratory studies.

This makes OBSINFOX a valuable resource for developing models capable of capturing **weak signals of misinformation**, especially in low-resource languages such as French.

---

## Methodology

The notebook details our methodology, including:

* Preprocessing and normalization steps,
* Feature extraction (manual and model-based),
* Exploratory data analysis,
* First insights on classification and dimensionality reduction.

By combining human and machine perspectives, we investigate whether linguistic and semantic cues can help uncover manipulative editorial strategies in online news.

---

##References

* Icard, T., et al. (2024). *A Multi-Label Dataset of French Fake News: Human and Machine Insights*.
* OBSINFOX GitHub Repository: [https://github.com/icard-obsinfox/obsinfox](https://github.com/icard-obsinfox/obsinfox)
