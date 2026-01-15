# Finance Multi-Task Green Agent

A comprehensive benchmark agent for the [AgentBeats](https://agentbeats.org) platform that evaluates AI agents on their ability to analyze SEC 10-K filings across multiple financial analysis tasks.

## Overview

This green agent tests purple agents on **three distinct financial analysis tasks** using 900 real SEC 10-K filings from 2015-2020:

1. **Risk Factor Classification** (40% weight) - Identify and categorize risk factors from Section 1A
2. **Business Summary Generation** (30% weight) - Extract key business information from Section 1
3. **Cross-Section Consistency Check** (30% weight) - Verify risks mentioned in Section 1A are discussed in Section 7

**Overall Score**: Weighted average (0-100) across all three tasks, providing comprehensive evaluation of financial document understanding.

## Quick Start

### Prerequisites

- Python 3.11+
- `uv` package manager
- OpenRouter API key (or compatible LLM API)

### Installation

```bash
# Install dependencies
uv sync

# Configure API key
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# (Optional) Validate dataset and environment
uv run python src/data_utils.py
```

> [!IMPORTANT]
> **Ground Truth Caching**: This agent uses a caching system to ensure reproducible evaluation results. The first evaluation of each 10-K filing will generate and cache the ground truth answers. Subsequent evaluations of the same filing will use the cached results, ensuring consistent benchmark performance.

### Running Locally

```bash
# Start the green agent
uv run python src/server.py --host 127.0.0.1 --port 9009
```

The agent will be available at `http://127.0.0.1:9009`

### Testing with Example Purple Agent

```bash
# Terminal 1: Start green agent
uv run python src/server.py --port 9009

# Terminal 2: Start example purple agent (from parent directory)
cd ../finance-purple-agent
uv run python src/analyst.py --port 9020

# Terminal 3: Run evaluation
cd ../agentbeats-tutorial
uv run python -m agentbeats.client_cli ../finance-green-agent/scenario.toml
```

**Cache Behavior**:
- **First run**: Generates ground truth using LLM (~15-25 seconds)
- **Subsequent runs**: Loads from cache (~instant, <100ms)
- **Cache location**: `data/ground_truth_cache.json`

## Evaluation Tasks

### Input Format

Purple agents receive an `EvalRequest` with:

```json
{
  "participants": {
    "analyst": "http://purple-agent-url:port"
  },
  "config": {
    "year": "2020",
    "cik": "1041514"  // optional, random if not specified
  }
}
```

### Task 1: Risk Factor Classification (40% weight)

**Input**: Section 1A text (first 12,000 characters of Risk Factors)

**Task**: Classify risk factors into predefined categories

**Predefined Categories**:
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

**Expected Output**:
```json
{
  "task": "risk_classification",
  "categories": [
    "Market Risk",
    "Operational Risk",
    "Legal/Regulatory Risk"
  ]
}
```

**Scoring**: F1 score (precision & recall) × 100

### Task 2: Business Summary Generation (30% weight)

**Input**: Section 1 text (first 10,000 characters of Business Description)

**Task**: Extract key business information including industry, products/services, and geographic markets

**Expected Output**:
```json
{
  "task": "business_summary",
  "industry": "Financial Technology (Fintech)",
  "products": "Universal Electronic Payment System (UEPS)",
  "geography": "South Africa, emerging markets"
}
```

**Scoring**: Percentage of key elements correctly identified (industry, products, geography) × 100

### Task 3: Cross-Section Consistency Check (30% weight)

**Input**:
- Section 1A text (Risk Factors - first 12,000 chars)
- Section 7 text (MD&A - first 15,000 chars)

**Task**: Identify which risks mentioned in Section 1A are actually discussed in Section 7 (Management Discussion & Analysis)

**Expected Output**:
```json
{
  "task": "consistency_check",
  "risks_found_in_1a": [
    "COVID-19 pandemic impact",
    "Operational challenges",
    "Financial risks"
  ],
  "risks_discussed_in_7": [
    "COVID-19 pandemic impact",
    "Financial risks"
  ]
}
```

**Scoring**: Consistency rate (correctly identified discussions / total risks) × 100

### Overall Scoring

**Final Score** = (Task1_Score × 0.4) + (Task2_Score × 0.3) + (Task3_Score × 0.3)

**Example**:
```
Task 1 (Risk Classification): 82.4/100
Task 2 (Business Summary): 100.0/100
Task 3 (Consistency Check): 80.0/100

Overall Score = (82.4 × 0.4) + (100.0 × 0.3) + (80.0 × 0.3) = 86.9/100
```

## Dataset

- **900 SEC 10-K filings** (2015-2020)
- **150 files per year**
- **Source**: SEC EDGAR database
- **Format**: JSON with parsed sections
- **Size**: ~235MB total

See `data/README.md` for detailed dataset documentation.

## Project Structure

```
finance-green-agent/
├── data/                   # 900 10-K filings (2015-2020)
│   ├── 2015/ ... 2020/
│   ├── dataset_metadata.json
│   └── README.md
├── src/
│   ├── agent.py            # Core evaluation logic
│   ├── executor.py         # A2A request handler
│   ├── messenger.py        # Agent communication
│   └── server.py           # Server & agent card
├── tests/
│   └── test_agent.py       # Test suite
├── .env                    # API configuration
├── scenario.toml           # Test scenario
├── Dockerfile              # Container image
├── pyproject.toml          # Dependencies
└── README.md               # This file
```

## Example Results

**Test with CIK 1041514 (2020)**:

```
Overall Score: 86.9/100

Task Breakdown:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Task 1: Risk Classification (40% weight)
Score: 82.4/100
Precision: 70.00%, Recall: 100.00%, F1: 82.35%

Ground Truth (7 categories):
✓ COVID-19/Pandemic Risk
✓ Operational Risk
✓ Financial Risk
✓ Legal/Regulatory Risk
✓ Competition Risk
✓ Human Capital/Talent Risk
✓ Market Risk

Agent Found (10 categories):
✓ All 7 ground truth categories identified
✗ 3 false positives: Supply Chain Risk, Reputational Risk, Strategic Risk

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Task 2: Business Summary (30% weight)
Score: 100.0/100

✓ industry: Financial Technology (Fintech)
✓ products: Universal Electronic Payment System (UEPS)
✓ geography: South Africa (primary market), emerging economies

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Task 3: Consistency Check (30% weight)
Score: 80.0/100
Consistency Rate: 80.00%

Risks in Section 1A (5 total):
✓ COVID-19 pandemic impact on operations
✓ Government restrictions affecting business
✓ Dependence on face-to-face interactions
✓ Potential increase in loan defaults
✗ Remote work and cybersecurity challenges (not discussed in Section 7)

Correctly identified discussions: 4/5
```

## Docker Deployment

### Building

```bash
docker build -t finance-multi-task-agent .
```

### Running

```bash
docker run -p 9009:9009 \
  -e OPENROUTER_API_KEY=your-key \
  -e MODEL_ID=deepseek/deepseek-v3.2 \
  finance-multi-task-agent
```

### Environment Variables

- `OPENROUTER_API_KEY`: API key for LLM provider
- `MODEL_ID`: Model to use (default: `deepseek/deepseek-v3.2`)
- `FINANCE_DATA_PATH`: Path to dataset (default: `data`)

## Testing

```bash
# Install test dependencies
uv sync --extra test

# Run A2A conformance tests
uv run pytest tests/ --agent-url http://localhost:9009
```

## API Documentation

### Agent Card

```bash
curl http://localhost:9009/.well-known/agent-card.json
```

Returns agent metadata including:
- Name: `finance-multi-task-analyst`
- Skills: Multi-task financial analysis (risk classification, business summary, consistency checking)
- Capabilities: Streaming support

### Health Check

The agent automatically handles A2A protocol health checks.

## Performance

**Cost per Evaluation** (using DeepSeek V3.2):
- ~40,000 tokens per evaluation (3 tasks)
- Cost: ~$0.0006 USD per evaluation
- 90% cheaper than Google Gemini

**Evaluation Time**:
- Task 1 (Risk Classification): ~5-8 seconds
- Task 2 (Business Summary): ~3-5 seconds
- Task 3 (Consistency Check): ~5-7 seconds
- Total: ~15-25 seconds per filing

## Benchmark Quality

**Strengths**:
- ✅ Real-world data (900 SEC filings)
- ✅ Multi-task evaluation (3 distinct analytical capabilities)
- ✅ Rich semantic content (80K-250K chars per filing)
- ✅ Clear evaluation metrics (F1, percentage match, consistency rate)
- ✅ Weighted scoring system balances task importance
- ✅ Reproducible results
- ✅ Tests both classification and generation capabilities

**Limitations**:
- Some filings may have missing sections
- Risk categories can overlap
- Ground truth generated by LLM (not human-verified)
- Text truncated to fit token limits
- Business summary evaluation is keyword-based

## What Makes This Benchmark Unique

**Multi-Task Comprehension**: Unlike single-task benchmarks, this evaluates:
- **Classification skills** (Task 1: Risk categorization)
- **Extraction skills** (Task 2: Business information)
- **Reasoning skills** (Task 3: Cross-document consistency)

**Real-World Relevance**: SEC 10-K analysis is a critical financial task requiring deep document understanding, making this benchmark practical and valuable.

**Scalability**: 900 filings × 3 tasks = 2,700 potential evaluation points, enabling robust statistical analysis of agent capabilities.

## Contributing

Contributions welcome! Areas for improvement:
- Human-verified ground truth labels
- Additional risk categories
- More sophisticated business summary evaluation (semantic similarity)
- Additional consistency checks (e.g., Section 1A vs 1, Section 7 vs 7A)
- Expanded dataset (more years, more companies)
- Multi-year trend analysis tasks

## License

- Code: MIT License
- Dataset: Public domain (SEC filings)

## Citation

```bibtex
@misc{finance-multi-task-2026,
  title={Finance 10-K Multi-Task Analysis Benchmark},
  author={AgentX-AgentBeats Competition},
  year={2026},
  note={900 SEC 10-K filings (2015-2020) for multi-task financial analysis: risk classification, business summary, consistency checking}
}
```

## Resources

- **AgentBeats Platform**: https://agentbeats.org
- **A2A Protocol**: https://a2a-protocol.org/latest/
- **Competition**: https://ape.agentbeats.org/
- **SEC EDGAR**: https://www.sec.gov/edgar

## Support

For issues or questions:
1. Check `data/README.md` for dataset details
2. Review evaluation examples in test files
3. See A2A protocol documentation for integration help

---

## 📋 Phase 1 提交 TODO 清单

> **截止日期**: 2026-01-16 | **剩余时间**: ~1天

### ✅ 已完成

**核心功能**:
- [x] Ground Truth 缓存系统（确保可复现性）
- [x] 三项评估任务实现
- [x] A2A 协议兼容
- [x] 单元测试（17个）
- [x] Dockerfile 配置
- [x] Baseline Purple Agent

**文档**:
- [x] README 完整
- [x] 代码注释
- [x] 数据验证工具

### ⏳ 待办事项（需队友协助）

#### 1. 🎬 录制 Demo Video（最高优先级）
**负责人**: ___________  
**时长**: ≤3分钟  
**预计**: 30分钟

**脚本**:
```
0:00-0:30  项目介绍 + 三项任务说明
0:30-1:30  启动 Green + Purple Agent
1:30-2:30  ⭐ 演示缓存机制（第1次慢，第2次快）+ 评分
2:30-3:00  总结亮点（可复现性、900数据集、多任务）
```

#### 2. 📤 GitHub 推送（高优先级）
**负责人**: ___________  
**预计**: 15分钟

```bash
# Green Agent
cd finance-green-agent
git init
git add .
git commit -m "Phase 1: Finance Green Agent with caching"
git remote add origin YOUR_REPO_URL
git push -u origin main

# Purple Agent  
cd finance-purple-agent
git init
git add .
git commit -m "Phase 1: Baseline Purple Agent"
git remote add origin YOUR_PURPLE_REPO_URL
git push -u origin main
```

**检查**: 
- [ ] 仓库设为 Public
- [ ] `.env` 未被推送

#### 3. 🌐 AgentBeats 注册（高优先级）
**负责人**: ___________  
**预计**: 20分钟

**步骤**:
1. 访问 https://agentbeats.dev
2. 注册 Green Agent
3. 注册 Purple Agent (baseline)
4. 上传 Demo Video
5. 提交 Abstract

### 📊 提交材料

- [x] Source Code
- [ ] Demo Video
- [x] README
- [x] Baseline Purple Agent
- [x] Dockerfile
- [ ] AgentBeats Registration

### 📞 队友分工

- **队友 A**: Demo Video 录制
- **队友 B**: GitHub 推送
- **队友 C**: 平台注册

**预计完成**: 1-2小时

🎯 冲刺 Phase 1！
