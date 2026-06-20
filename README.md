# ESG Risk Pattern Detection in Indian Banking Sustainability Disclosures

> An end-to-end NLP + ML pipeline for automatically detecting, classifying, and temporally tracking ESG risk themes in Indian banking annual and sustainability reports (2020–2024).

---

## 📌 Overview

Indian banks are mandated by SEBI to publish Business Responsibility and Sustainability Reports (BRSR), generating large volumes of unstructured ESG disclosure text. Existing ESG evaluation frameworks rely on static scores and pre-coded categories, making them unable to capture how risk narratives evolve year over year.

This project addresses that gap with a fully automated pipeline that:
- Extracts paragraph-level text from ESG PDF reports
- Filters risk-relevant content using a domain-specific lexicon
- Discovers latent ESG risk themes using BERTopic neural topic modeling
- Quantifies how theme intensity shifts across five fiscal years (2020–2024)
- Visualizes findings through an interactive Plotly dashboard

---

## 🏦 Dataset

- **24 Indian banks** (public and private sector), including ICICI, HDFC, SBI, Axis, Kotak Mahindra, Canara, IndusInd, Yes Bank, PNB, and Bank of Baroda
- **121 PDF documents** spanning FY2020–FY2024 (Sustainability Reports, Integrated Annual Reports, BRSR filings)
- **1,134 ESG risk-relevant paragraphs** identified after lexicon filtering
- Reports sourced directly from official bank websites and corporate disclosure portals

---

## 🏗️ Architecture

```
Raw ESG PDFs
     │
     ▼
PDF Text Extraction (PyMuPDF / fitz)
     │  paragraph-level segmentation, metadata tagging (bank, year, page)
     ▼
ESG Risk Lexicon Filtering
     │  5 categories · regex patterns · precision filtering
     ▼
BERTopic Neural Topic Modeling
     │  sentence-transformers → UMAP → HDBSCAN → c-TF-IDF
     ▼
Temporal Intensity Computation
     │  normalized proportion metric per theme per year
     ▼
Linear Regression Trend Analysis
     │  classify themes as Emerging / Declining / Stable
     ▼
Interactive Plotly Dashboard
     (line chart · bank-wise heatmap · topic frequency bar chart · intertopic distance map)
```

---

## 🗂️ ESG Risk Lexicon Categories

| Category | Sample Keywords |
|---|---|
| **Climate Risk** | carbon footprint, GHG emissions, net zero, TCFD, transition risk, renewable energy |
| **Governance Risk** | board diversity, anti-corruption, whistleblower, regulatory compliance, internal controls |
| **Cyber Risk** | cybersecurity, data breach, ransomware, phishing, cyber resilience, IT risk |
| **Credit / Portfolio Risk** | NPA, loan defaults, capital adequacy, stressed assets, liquidity risk |
| **Social / Reputation Risk** | human rights, diversity & inclusion, community engagement, responsible banking |

---

## 🤖 BERTopic Model Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `embedding_model` | `all-MiniLM-L6-v2` | Lightweight; semantic precision suited to ESG paragraph length |
| `min_topic_size` | `2` | Accommodates sparse corpus; prevents over-noise-classification |
| `umap n_neighbors` | `15` | Balances local vs. global structure in embedding space |
| `umap n_components` | `5` | Sufficient dimensionality for HDBSCAN while removing noise |
| `hdbscan metric` | `euclidean` | Standard metric for post-UMAP space |
| `vectorizer_model` | `CountVectorizer(stop_words='english')` | Removes stop words from c-TF-IDF keyword generation |

---

## 📊 Key Results

### Discovered Themes

| Topic | Theme | Defining Keywords |
|---|---|---|
| 0 | **Strategic ESG Integration** | sustainability strategy, ESG governance, green finance, stakeholder engagement, climate risk management |
| 1 | **Regulatory & Compliance Reporting** | BRSR compliance, SEBI disclosure, mandatory reporting, audit committee, board-level risk governance |

### Temporal Trend Analysis (2020–2024)

| Theme | Slope (β) | Trend | Peak Year |
|---|---|---|---|
| Strategic ESG Integration | +0.1095 | 🟢 Emerging | 2024 |
| Regulatory & Compliance Reporting | −0.0842 | 🔴 Declining | 2020–2021 |

**Key Insight:** Banks initially devoted significant disclosure space to explaining BRSR compliance processes (2020–2021), then internalized those requirements and shifted toward more substantive, proactive ESG strategy narratives (2022–2024).

---

## ⚙️ Tech Stack

| Layer | Tools |
|---|---|
| PDF Extraction | `PyMuPDF (fitz)` |
| Text Processing | `re`, `NLTK`, `pandas` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Dimensionality Reduction | `UMAP` |
| Clustering | `HDBSCAN` |
| Topic Modeling | `BERTopic` |
| Trend Analysis | `scikit-learn` (LinearRegression) |
| Visualization | `Plotly`, `Matplotlib`, `Seaborn` |
| Notebook Environment | `Jupyter` / `Google Colab` |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/esg-risk-detection.git
cd esg-risk-detection
```

### 2. Install dependencies

```bash
pip install pandas numpy PyMuPDF scikit-learn nltk matplotlib seaborn plotly \
            bertopic sentence-transformers umap-learn hdbscan tqdm
```

Or in a Colab/Jupyter environment:

```python
%pip install pandas numpy pdfplumber PyMuPDF scikit-learn nltk \
             matplotlib seaborn plotly bertopic sentence-transformers \
             umap-learn hdbscan tqdm ipywidgets
```

### 3. Organize your PDFs

```
ESG_Project/
├── pdfs/
│   ├── ICICI_Bank/
│   │   ├── ICICI_2020.pdf
│   │   ├── ICICI_2021.pdf
│   │   └── ...
│   ├── HDFC_Bank/
│   └── ...
├── json_data/      # auto-generated
└── output/         # auto-generated
```

### 4. Run the pipeline

Open and run `Code.ipynb` sequentially, or execute the modules in order:

```
1. PDF Text Extraction      → json_data/*.json
2. Lexicon Filtering        → filtered corpus with Risk_Categories
3. BERTopic Training        → esg_bertopic_model/
4. Temporal Intensity       → output/risk_theme_intensity.csv
5. Trend Analysis           → output/temporal_trend_analysis.csv
6. Visualization            → output/*.html
```

---

## 📁 Output Files

| File | Description |
|---|---|
| `json_data/*.json` | Extracted paragraphs with bank, year, page, text, and matched risk categories |
| `risk_theme_intensity.csv` | Normalized intensity scores per theme per year |
| `temporal_trend_analysis.csv` | Linear regression slopes and Emerging/Declining/Stable labels |
| `final_esg_analysis_summary.csv` | Executive insights per theme |
| `temporal_intensity.html` | Interactive Plotly line chart of theme evolution |
| `bankwise_heatmap.html` | Interactive Plotly heatmap — bank × theme paragraph counts |
| `esg_bertopic_model/` | Saved BERTopic model for inference on new reports |

---

## 🗺️ Project Roadmap (Agile Sprints)

| Sprint | Weeks | Deliverables |
|---|---|---|
| **Sprint I** | 1–4 | PDF collection, PyMuPDF extraction, JSON corpus construction |
| **Sprint II** | 5–8 | ESG risk lexicon, BERTopic training, theme mapping |
| **Sprint III** | 9–12 | Temporal intensity, regression trend analysis, visualizations, summary report |

---

## 🔭 Future Enhancements

- **OCR Integration** — Tesseract / AWS Textract fallback for scanned image-based PDFs
- **Expanded Dataset** — Include mid-sized banks, RRBs, small finance banks, and NBFCs
- **Domain-specific Embeddings** — Replace `all-MiniLM-L6-v2` with `FinBERT` or `ClimateBERT`
- **Extended Temporal Window** — Pre-2020 and post-2024 data for richer trend estimation
- **Multi-lingual Analysis** — Hindi-language BRSR annexures via multilingual sentence transformers
- **Real-time Dashboard** — Streamlit/Dash app that auto-ingests newly published ESG reports

---

## 🌐 SDG Alignment

| SDG | Relevance |
|---|---|
| **SDG 8** – Decent Work & Economic Growth | Monitors credit/portfolio risk disclosures for early financial instability signals |
| **SDG 13** – Climate Action | Tracks climate risk theme intensity in banking disclosures |
| **SDG 16** – Strong Institutions | Monitors governance and compliance risk themes for institutional transparency |
| **SDG 17** – Partnerships for the Goals | Open, replicable methodology for regulators, ESG agencies, and academia |

---

## 📚 Key References

- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv:2203.05794*
- Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv:1908.10063*
- Webersinke et al. (2021). ClimateBERT: A pretrained language model for climate-related text. *arXiv:2110.12010*
- Schimanski et al. (2024). Bridging the gap in ESG measurement using NLP. *Finance Research Letters, 61*, 104979

---

## 📜 License

This project was developed as a B.Tech Major Project (21CSP401L) at SRM Institute of Science and Technology. For academic use only.
