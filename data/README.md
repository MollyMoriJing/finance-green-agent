# Risk Classification Benchmark Dataset

This dataset contains 900 SEC 10-K filings (2015-2020) used for the **Risk Factor Classification** evaluation task.

## Dataset Overview

- **Total Files**: 900 JSON files
- **Years**: 2015-2020 (150 files per year)
- **Source**: SEC EDGAR 10-K filings
- **Size**: ~235MB total
- **Format**: JSON with parsed sections

## File Structure

```
data/
├── 2015/           # 150 filings from 2015
├── 2016/           # 150 filings from 2016
├── 2017/           # 150 filings from 2017
├── 2018/           # 150 filings from 2018
├── 2019/           # 150 filings from 2019
├── 2020/           # 150 filings from 2020
├── dataset_metadata.json
└── README.md (this file)
```

## JSON File Format

Each filing is named `{CIK}_{YEAR}.json` and contains:

```json
{
  "filename": "original_filing.htm",
  "cik": "1234567",
  "year": "2020",
  "section_1": "Item 1 - Business Description...",
  "section_1A": "Item 1A - Risk Factors...",
  "section_1B": "Item 1B - Unresolved Staff Comments...",
  "section_7": "Item 7 - MD&A...",
  "section_7A": "Item 7A - Market Risk...",
  ...
}
```

## Section 1A: Risk Factors

This benchmark focuses on **Section 1A** which contains risk factor disclosures.

**Key Statistics**:
- **Average length**: 80,000 - 250,000 characters
- **Content**: Comprehensive risk disclosures across multiple categories
- **Quality**: Highly structured, rich semantic content

**Example risk categories found**:
- Market Risk
- Operational Risk
- Financial Risk
- Legal/Regulatory Risk
- Technology Risk
- Cybersecurity Risk
- Competition Risk
- Supply Chain Risk
- Human Capital/Talent Risk
- Environmental/Climate Risk
- COVID-19/Pandemic Risk
- Geopolitical Risk

## Evaluation Task

**Task**: Classify risk factors from Section 1A into predefined categories.

**Input**: Section 1A text (first 12,000 characters)

**Output**: JSON array of risk categories
```json
{
  "categories": [
    "Market Risk",
    "Operational Risk",
    "Legal/Regulatory Risk"
  ]
}
```

**Scoring**: F1 score based on category overlap with ground truth
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 * (Precision * Recall) / (Precision + Recall)
- Final Score = F1 * 100

## Data Quality

✅ **Strengths**:
- Real-world SEC filings (public domain)
- Diverse industries represented
- Consistent structure across years
- Rich semantic content (80K-250K chars per filing)
- Multiple risk categories per filing (typically 5-10)

⚠️ **Limitations**:
- Some filings may have missing sections
- Text length varies significantly
- Risk disclosure depth varies by company
- Some categories may overlap

## Usage Example

```python
import json
from pathlib import Path

# Load a filing
filing_path = Path("data/2020/1041514_2020.json")
with open(filing_path) as f:
    filing = json.load(f)

# Extract Section 1A
risk_factors = filing["section_1A"]
print(f"Risk section length: {len(risk_factors)} chars")
print(f"Preview: {risk_factors[:500]}")
```

## Data Provenance

- **Source**: SEC EDGAR database
- **Collection**: Random stratified sampling (150 per year)
- **Processing**: HTML parsing, section extraction
- **Validation**: Manual spot-checking for quality
- **License**: Public domain (SEC filings)

## Citation

If you use this dataset, please reference:
```
Finance 10-K Risk Classification Benchmark
SEC EDGAR 10-K filings (2015-2020)
900 filings across 6 years
Source: https://www.sec.gov/edgar
```
