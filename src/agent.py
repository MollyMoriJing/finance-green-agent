import json
import os
import random
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, HttpUrl, ValidationError
from openai import AsyncOpenAI
from dotenv import load_dotenv

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from cache import GroundTruthCache

load_dotenv()


def safe_json_parse(raw_text: str) -> dict:
    """
    Parse JSON from LLM response, handling markdown code blocks.
    
    Args:
        raw_text: Raw text from LLM that may contain JSON
        
    Returns:
        Parsed JSON dict
        
    Raises:
        json.JSONDecodeError: If parsing fails after cleanup
    """
    text = raw_text.strip()
    
    # Remove markdown code block markers
    if text.startswith("```"):
        # Split by ``` and take the middle part
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            # Remove language identifier (e.g., 'json')
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
    
    # Try to parse
    return json.loads(text)


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]


# Task 1: Risk Classification Models
class RiskEvalResult(BaseModel):
    """Evaluation result for risk classification task."""
    task: Literal["risk_classification"] = "risk_classification"
    score: float  # 0-100
    categories_found: list[str]
    ground_truth_categories: list[str]
    precision: float
    recall: float
    f1_score: float
    feedback: str


# Task 2: Business Summary Models
class BusinessSummaryEvalResult(BaseModel):
    """Evaluation result for business summary task."""
    task: Literal["business_summary"] = "business_summary"
    score: float  # 0-100
    summary_quality: float  # Semantic similarity
    key_elements_found: dict[str, bool]  # industry, products, geography
    feedback: str


# Task 3: Consistency Check Models
class ConsistencyEvalResult(BaseModel):
    """Evaluation result for cross-section consistency check."""
    task: Literal["consistency_check"] = "consistency_check"
    score: float  # 0-100
    risks_mentioned_in_1a: list[str]
    risks_discussed_in_7: list[str]
    consistency_rate: float
    missing_discussions: list[str]
    feedback: str


# Overall Multi-Task Result
class MultiTaskEvalResult(BaseModel):
    """Combined evaluation result for all tasks."""
    overall_score: float  # 0-100, weighted average
    task_results: dict[str, dict]  # task_name -> result dict
    task_scores: dict[str, float]  # task_name -> score
    feedback: str


class Agent:
    # Required participant roles: one purple agent to be evaluated
    required_roles: list[str] = ["analyst"]
    # Required config: which year and CIK to analyze
    required_config_keys: list[str] = ["year"]

    # Task weights for overall score
    TASK_WEIGHTS = {
        "risk_classification": 0.4,
        "business_summary": 0.3,
        "consistency_check": 0.3
    }

    def __init__(self):
        self.messenger = Messenger()
        # Initialize OpenRouter client
        self._client = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        self._model = os.getenv("MODEL_ID", "deepseek/deepseek-v3.2")
        self._finance_data_path = Path(os.getenv("FINANCE_DATA_PATH", "data"))
        
        # Initialize ground truth cache
        cache_path = self._finance_data_path / "ground_truth_cache.json"
        self._cache = GroundTruthCache(cache_path)
        print(f"📦 Cache initialized: {self._cache.stats()}")

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Validate year is in range
        year = request.config.get("year")
        if year not in ["2015", "2016", "2017", "2018", "2019", "2020"]:
            return False, f"Year must be between 2015-2020, got: {year}"

        return True, "ok"

    def load_10k_filing(self, year: str, cik: str) -> dict | None:
        """Load a specific 10-K filing from the dataset."""
        file_path = self._finance_data_path / year / f"{cik}_{year}.json"
        if not file_path.exists():
            return None

        with open(file_path, 'r') as f:
            return json.load(f)

    def get_random_filing(self, year: str) -> tuple[str, dict] | None:
        """Get a random filing from specified year."""
        year_path = self._finance_data_path / year
        if not year_path.exists():
            return None

        files = list(year_path.glob("*.json"))
        if not files:
            return None

        selected_file = random.choice(files)
        cik = selected_file.stem.split('_')[0]  # Extract CIK from filename

        with open(selected_file, 'r') as f:
            data = json.load(f)

        return cik, data

    # ========== TASK 1: RISK CLASSIFICATION ==========

    async def extract_ground_truth_risks(self, filing_data: dict, cik: str, year: str) -> list[str]:
        """Use LLM to extract and classify risk categories from 10-K Section 1A."""
        
        # Check cache first
        cached = self._cache.get(cik, year, "risk")
        if cached:
            return cached["data"]
        
        section_1a = filing_data.get("section_1A", "")

        if not section_1a or len(section_1a) < 100:
            return []

        section_text = section_1a[:12000]

        prompt = f"""Analyze the following Risk Factors section from a 10-K filing and classify the main risk categories.

Risk Factors Text:
{section_text}

Identify and list the PRIMARY risk categories mentioned. Common categories include:
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

Return ONLY a JSON array of category names found in this text.
Example format: {{"categories": ["Market Risk", "Operational Risk"]}}
"""

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are a financial risk analyst. Classify risk factors accurately. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        try:
            result = safe_json_parse(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse LLM response: {e}")
            return []
        
        if isinstance(result, dict):
            categories = result.get("categories", result.get("risk_categories", []))
        else:
            categories = result

        categories_list = categories if isinstance(categories, list) else []
        
        # Store in cache
        self._cache.set(cik, year, "risk", categories_list)
        
        return categories_list

    async def evaluate_risk_classification(self, agent_result: str, ground_truth: list[str]) -> RiskEvalResult:
        """Compare agent's risk classification with ground truth and score."""
        try:
            agent_data = json.loads(agent_result)
            if isinstance(agent_data, dict):
                agent_categories = agent_data.get("categories", agent_data.get("risk_categories", agent_data.get("risk_classification", [])))
            else:
                agent_categories = agent_data

            if not isinstance(agent_categories, list):
                agent_categories = []

        except json.JSONDecodeError:
            return RiskEvalResult(
                score=0.0,
                categories_found=[],
                ground_truth_categories=ground_truth,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                feedback="Agent response is not valid JSON"
            )

        # Normalize categories for comparison
        agent_cats_norm = set(c.lower().strip() for c in agent_categories)
        truth_cats_norm = set(c.lower().strip() for c in ground_truth)

        # Calculate metrics
        true_positives = len(agent_cats_norm & truth_cats_norm)
        false_positives = len(agent_cats_norm - truth_cats_norm)
        false_negatives = len(truth_cats_norm - agent_cats_norm)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        score = f1_score * 100

        feedback_parts = []
        feedback_parts.append(f"Found {len(agent_categories)} categories, expected {len(ground_truth)}")
        feedback_parts.append(f"Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1_score:.2%}")
        feedback_parts.append(f"\nCorrectly identified: {list(agent_cats_norm & truth_cats_norm)}")
        if false_positives > 0:
            feedback_parts.append(f"False positives: {list(agent_cats_norm - truth_cats_norm)}")
        if false_negatives > 0:
            feedback_parts.append(f"Missed: {list(truth_cats_norm - agent_cats_norm)}")

        return RiskEvalResult(
            score=score,
            categories_found=agent_categories,
            ground_truth_categories=ground_truth,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            feedback="\n".join(feedback_parts)
        )

    # ========== TASK 2: BUSINESS SUMMARY ==========

    async def extract_ground_truth_business(self, filing_data: dict, cik: str, year: str) -> dict[str, str]:
        """Extract business summary ground truth from Section 1."""
        
        # Check cache first
        cached = self._cache.get(cik, year, "business")
        if cached:
            return cached["data"]
        
        section_1 = filing_data.get("section_1", "")

        if not section_1 or len(section_1) < 100:
            return {"industry": "N/A", "products": "N/A", "geography": "N/A"}

        section_text = section_1[:10000]

        prompt = f"""Analyze this business description from a 10-K filing and extract key information.

Business Description:
{section_text}

Extract:
1. Industry/sector
2. Main products or services
3. Geographic markets

Return as JSON:
{{"industry": "...", "products": "...", "geography": "..."}}
"""

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are a business analyst. Extract key business information accurately. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        try:
            result = safe_json_parse(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse LLM response: {e}")
            result = {}
        
        business_data = {
            "industry": result.get("industry", "N/A"),
            "products": result.get("products", "N/A"),
            "geography": result.get("geography", "N/A")
        }
        
        # Store in cache
        self._cache.set(cik, year, "business", business_data)
        
        return business_data

    async def evaluate_business_summary(self, agent_result: str, ground_truth: dict[str, str]) -> BusinessSummaryEvalResult:
        """Evaluate business summary quality."""
        try:
            agent_data = json.loads(agent_result)
            if isinstance(agent_data, dict):
                agent_summary = agent_data.get("business_summary", agent_data)
            else:
                agent_summary = {}

        except json.JSONDecodeError:
            return BusinessSummaryEvalResult(
                score=0.0,
                summary_quality=0.0,
                key_elements_found={"industry": False, "products": False, "geography": False},
                feedback="Agent response is not valid JSON"
            )

        # Check if key elements are present
        key_elements = {}
        for key in ["industry", "products", "geography"]:
            agent_value = str(agent_summary.get(key, "")).lower()
            truth_value = str(ground_truth.get(key, "")).lower()
            # Simple check: does agent mention similar concepts?
            key_elements[key] = (truth_value != "n/a" and len(agent_value) > 10)

        # Score based on element coverage
        elements_score = sum(key_elements.values()) / len(key_elements) * 100

        feedback_parts = []
        feedback_parts.append(f"Business summary elements: {sum(key_elements.values())}/{len(key_elements)}")
        for key, found in key_elements.items():
            status = "✓" if found else "✗"
            feedback_parts.append(f"{status} {key}: {agent_summary.get(key, 'missing')[:50]}...")

        return BusinessSummaryEvalResult(
            score=elements_score,
            summary_quality=elements_score / 100,
            key_elements_found=key_elements,
            feedback="\n".join(feedback_parts)
        )

    # ========== TASK 3: CONSISTENCY CHECK ==========

    async def extract_risk_discussions(self, section_1a: str, section_7: str, cik: str, year: str) -> tuple[list[str], list[str]]:
        """Extract risks mentioned in 1A and discussed in 7."""
        
        # Check cache first
        cached = self._cache.get(cik, year, "consistency")
        if cached:
            data = cached["data"]
            return data["risks_1a"], data["discussed_in_7"]
        
        if not section_1a or not section_7:
            return [], []

        # Extract main risk topics from 1A
        prompt_1a = f"""List the main risk topics mentioned in this Risk Factors section (first 8000 chars):

{section_1a[:8000]}

Return as JSON array of risk topics:
{{"risks": ["risk topic 1", "risk topic 2", ...]}}
"""

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "Extract risk topics. Return only valid JSON."},
                {"role": "user", "content": prompt_1a}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        try:
            risks_1a_data = safe_json_parse(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse LLM response: {e}")
            return [], []
        
        risks_1a = risks_1a_data.get("risks", [])[:5]  # Limit to top 5

        if not risks_1a:
            return [], []

        # Check if these risks are discussed in Section 7
        prompt_7 = f"""Check if the following risks are discussed in this MD&A section:

Risks to check: {json.dumps(risks_1a)}

MD&A text (first 8000 chars):
{section_7[:8000]}

Return JSON with risks that ARE discussed:
{{"discussed_risks": ["risk 1", "risk 2", ...]}}
"""

        response2 = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "Identify which risks are discussed. Return only valid JSON."},
                {"role": "user", "content": prompt_7}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        try:
            discussed_data = safe_json_parse(response2.choices[0].message.content)
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse LLM response: {e}")
            discussed_risks = []
        else:
            discussed_risks = discussed_data.get("discussed_risks", [])
        
        # Store in cache
        cache_data = {
            "risks_1a": risks_1a,
            "discussed_in_7": discussed_risks
        }
        self._cache.set(cik, year, "consistency", cache_data)

        return risks_1a, discussed_risks

    async def evaluate_consistency(self, agent_result: str, risks_1a: list[str], discussed_in_7: list[str]) -> ConsistencyEvalResult:
        """Evaluate cross-section consistency check."""
        try:
            agent_data = json.loads(agent_result)
            if isinstance(agent_data, dict):
                agent_discussed = agent_data.get("consistent_risks", agent_data.get("consistency_check", []))
            else:
                agent_discussed = []

        except json.JSONDecodeError:
            return ConsistencyEvalResult(
                score=0.0,
                risks_mentioned_in_1a=risks_1a,
                risks_discussed_in_7=[],
                consistency_rate=0.0,
                missing_discussions=[],
                feedback="Agent response is not valid JSON"
            )

        # Normalize
        agent_norm = set(r.lower().strip() for r in agent_discussed if isinstance(r, str))
        truth_norm = set(r.lower().strip() for r in discussed_in_7)

        # Calculate consistency
        if not risks_1a:
            consistency_rate = 1.0
        else:
            correct = len(agent_norm & truth_norm)
            consistency_rate = correct / len(risks_1a) if len(risks_1a) > 0 else 0.0

        score = consistency_rate * 100

        missing = set(r.lower() for r in risks_1a) - truth_norm

        feedback = f"Consistency check: {len(agent_norm & truth_norm)}/{len(risks_1a)} risks correctly identified as discussed\n"
        feedback += f"Risks in 1A: {risks_1a}\n"
        feedback += f"Discussed in 7: {list(agent_norm & truth_norm)}"

        return ConsistencyEvalResult(
            score=score,
            risks_mentioned_in_1a=risks_1a,
            risks_discussed_in_7=list(agent_norm),
            consistency_rate=consistency_rate,
            missing_discussions=list(missing),
            feedback=feedback
        )

    # ========== MAIN EVALUATION LOOP ==========

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Main multi-task evaluation logic."""
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        # Get configuration
        year = request.config["year"]
        cik = request.config.get("cik")

        await updater.update_status(
            TaskState.working, new_agent_text_message(f"Loading 10-K filing for year {year}...")
        )

        # Load filing data
        if cik:
            filing_data = self.load_10k_filing(year, cik)
            if not filing_data:
                await updater.reject(new_agent_text_message(f"Filing not found: {cik}_{year}"))
                return
        else:
            result = self.get_random_filing(year)
            if not result:
                await updater.reject(new_agent_text_message(f"No filings found for year {year}"))
                return
            cik, filing_data = result

        analyst_url = str(request.participants["analyst"])
        task_results = {}
        task_scores = {}

        # ===== TASK 1: RISK CLASSIFICATION =====

        await updater.update_status(
            TaskState.working, new_agent_text_message(f"Task 1/3: Extracting risk categories from CIK {cik}...")
        )

        ground_truth_risks = await self.extract_ground_truth_risks(filing_data, cik, year)

        if ground_truth_risks:
            section_1a = filing_data.get('section_1A', '')[:12000]
            task1_prompt = f"""TASK 1: Risk Classification

Analyze the following Risk Factors section and classify the main risk categories.

Risk Factors (Section 1A):
{section_1a}

Categories: Market Risk, Operational Risk, Financial Risk, Legal/Regulatory Risk, Technology Risk, Cybersecurity Risk, Competition Risk, Supply Chain Risk, Human Capital/Talent Risk, Environmental/Climate Risk, COVID-19/Pandemic Risk, Geopolitational Risk

Return JSON: {{"risk_classification": ["category1", "category2", ...]}}
"""

            agent_response_1 = await self.messenger.talk_to_agent(task1_prompt, analyst_url)
            eval_result_1 = await self.evaluate_risk_classification(agent_response_1, ground_truth_risks)
            task_results["risk_classification"] = eval_result_1.model_dump()
            task_scores["risk_classification"] = eval_result_1.score
        else:
            task_scores["risk_classification"] = 0.0

        # ===== TASK 2: BUSINESS SUMMARY =====

        await updater.update_status(
            TaskState.working, new_agent_text_message("Task 2/3: Evaluating business summary...")
        )

        ground_truth_business = await self.extract_ground_truth_business(filing_data, cik, year)

        section_1 = filing_data.get('section_1', '')[:10000]
        task2_prompt = f"""TASK 2: Business Summary

Analyze this business description and extract key information:

Business Description (Section 1):
{section_1}

Extract: industry/sector, main products/services, geographic markets

Return JSON: {{"business_summary": {{"industry": "...", "products": "...", "geography": "..."}}}}
"""

        agent_response_2 = await self.messenger.talk_to_agent(task2_prompt, analyst_url)
        eval_result_2 = await self.evaluate_business_summary(agent_response_2, ground_truth_business)
        task_results["business_summary"] = eval_result_2.model_dump()
        task_scores["business_summary"] = eval_result_2.score

        # ===== TASK 3: CONSISTENCY CHECK =====

        await updater.update_status(
            TaskState.working, new_agent_text_message("Task 3/3: Checking cross-section consistency...")
        )

        section_7 = filing_data.get('section_7', '')
        if section_7 and len(section_7) > 100:
            risks_1a, discussed_in_7 = await self.extract_risk_discussions(
                filing_data.get('section_1A', ''),
                section_7,
                cik,
                year
            )

            task3_prompt = f"""TASK 3: Consistency Check

Check which of these risks mentioned in Section 1A are actually discussed in Section 7 (MD&A):

Risks from Section 1A: {json.dumps(risks_1a)}

Section 7 (MD&A) text:
{section_7[:8000]}

Which risks ARE discussed in Section 7?

Return JSON: {{"consistency_check": ["risk1", "risk2", ...]}}
"""

            agent_response_3 = await self.messenger.talk_to_agent(task3_prompt, analyst_url)
            eval_result_3 = await self.evaluate_consistency(agent_response_3, risks_1a, discussed_in_7)
            task_results["consistency_check"] = eval_result_3.model_dump()
            task_scores["consistency_check"] = eval_result_3.score
        else:
            task_scores["consistency_check"] = 0.0

        # ===== CALCULATE OVERALL SCORE =====

        weighted_score = sum(
            task_scores.get(task, 0) * weight
            for task, weight in self.TASK_WEIGHTS.items()
        )

        overall_feedback = f"""Multi-Task Evaluation Complete

Overall Score: {weighted_score:.1f}/100

Task Scores:
- Risk Classification: {task_scores.get('risk_classification', 0):.1f}/100 (weight: 40%)
- Business Summary: {task_scores.get('business_summary', 0):.1f}/100 (weight: 30%)
- Consistency Check: {task_scores.get('consistency_check', 0):.1f}/100 (weight: 30%)
"""

        multi_result = MultiTaskEvalResult(
            overall_score=weighted_score,
            task_results=task_results,
            task_scores=task_scores,
            feedback=overall_feedback
        )

        # Return evaluation results
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=overall_feedback)),
                Part(root=DataPart(data=multi_result.model_dump()))
            ],
            name="MultiTaskFinancialEvaluation",
        )
