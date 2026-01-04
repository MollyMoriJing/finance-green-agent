# Sample Dataset for Testing

This is a minimal sample dataset (3 files) for quick testing of the Finance Multi-Task Green Agent.

## Contents

- **2018/101199_2018.json** - Sample filing from 2018
- **2019/1000045_2019.json** - Sample filing from 2019
- **2020/1041514_2020.json** - Sample filing from 2020 (used in default test scenario)

## Size

- 3 files (~3MB total)
- Compared to full dataset: 900 files (235MB)

## Usage

### Quick Test with Sample Data

```bash
# Edit .env to use sample dataset
FINANCE_DATA_PATH=data_sample

# Start green agent
uv run python src/server.py --port 9009

# In another terminal, start a purple agent
cd ../finance-purple-agent
uv run python analyst.py --port 9020

# Run evaluation (uses CIK 1041514 from 2020)
cd ../agentbeats-tutorial
uv run python -m agentbeats.client_cli ../finance-green-agent/scenario.toml
```

### Download Full Dataset

For production evaluation, use the full dataset (900 files, 2015-2020):

```bash
# The full dataset is included in the repository under data/
# To use full dataset, set in .env:
FINANCE_DATA_PATH=data
```

## File Structure

Each JSON file contains parsed SEC 10-K filing sections:
- `section_1` - Business Description
- `section_1A` - Risk Factors
- `section_7` - Management Discussion & Analysis (MD&A)
- `section_7A` - Quantitative and Qualitative Disclosures About Market Risk
- And other sections...

## Notes

- Sample dataset is for **testing only**
- For meaningful evaluation results, use the full dataset
- Sample files were randomly selected from each year
- All files follow the same structure as the full dataset
