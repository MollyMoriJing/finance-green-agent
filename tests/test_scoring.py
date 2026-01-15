"""
Unit tests for scoring logic in Finance Green Agent.

Tests the evaluation functions for risk classification, business summary,
and consistency checks to ensure correct metrics calculation.
"""

import pytest
from src.agent import Agent


@pytest.fixture
def agent():
    """Create an Agent instance for testing."""
    return Agent()


class TestRiskClassificationScoring:
    """Test risk classification F1 score calculation."""
    
    @pytest.mark.asyncio
    async def test_perfect_match(self, agent):
        """Test F1 score with perfect match."""
        ground_truth = ["Market Risk", "Operational Risk", "Financial Risk"]
        agent_response = '{"categories": ["Market Risk", "Operational Risk", "Financial Risk"]}'
        
        result = await agent.evaluate_risk_classification(agent_response, ground_truth)
        
        assert result.score == 100.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1_score == 1.0
        assert len(result.categories_found) == 3
    
    @pytest.mark.asyncio
    async def test_partial_match(self, agent):
        """Test F1 score with partial match."""
        ground_truth = ["Market Risk", "Operational Risk", "Financial Risk"]
        # Agent found 2 correct + 1 false positive, missed 1
        agent_response = '{"categories": ["Market Risk", "Financial Risk", "Technology Risk"]}'
        
        result = await agent.evaluate_risk_classification(agent_response, ground_truth)
        
        # Precision: 2/3 = 0.667
        # Recall: 2/3 = 0.667
        # F1: 2 * (0.667 * 0.667) / (0.667 + 0.667) = 0.667
        assert result.precision == pytest.approx(2/3, rel=0.01)
        assert result.recall == pytest.approx(2/3, rel=0.01)
        assert result.f1_score == pytest.approx(2/3, rel=0.01)
        assert result.score == pytest.approx(66.67, rel=0.1)
    
    @pytest.mark.asyncio
    async def test_case_insensitive(self, agent):
        """Test that scoring is case-insensitive."""
        ground_truth = ["Market Risk", "Operational Risk"]
        agent_response = '{"categories": ["market risk", "OPERATIONAL RISK"]}'
        
        result = await agent.evaluate_risk_classification(agent_response, ground_truth)
        
        assert result.score == 100.0
        assert result.f1_score == 1.0
    
    @pytest.mark.asyncio
    async def test_no_match(self, agent):
        """Test F1 score with no matches."""
        ground_truth = ["Market Risk", "Operational Risk"]
        agent_response = '{"categories": ["Technology Risk", "Cybersecurity Risk"]}'
        
        result = await agent.evaluate_risk_classification(agent_response, ground_truth)
        
        assert result.score == 0.0
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1_score == 0.0
    
    @pytest.mark.asyncio
    async def test_invalid_json(self, agent):
        """Test handling of invalid JSON response."""
        ground_truth = ["Market Risk"]
        agent_response = "This is not JSON"
        
        result = await agent.evaluate_risk_classification(agent_response, ground_truth)
        
        assert result.score == 0.0
        assert "not valid JSON" in result.feedback
    
    @pytest.mark.asyncio
    async def test_empty_agent_response(self, agent):
        """Test handling of empty agent response."""
        ground_truth = ["Market Risk", "Operational Risk"]
        agent_response = '{"categories": []}'
        
        result = await agent.evaluate_risk_classification(agent_response, ground_truth)
        
        assert result.score == 0.0
        assert result.recall == 0.0


class TestBusinessSummaryScoring:
    """Test business summary evaluation."""
    
    @pytest.mark.asyncio
    async def test_all_elements_found(self, agent):
        """Test scoring when all key elements are present."""
        ground_truth = {
            "industry": "Financial Technology",
            "products": "Payment processing",
            "geography": "United States"
        }
        agent_response = '''{
            "business_summary": {
                "industry": "Financial Technology (FinTech)",
                "products": "Online payment processing and merchant services",
                "geography": "Primarily United States with some international presence"
            }
        }'''
        
        result = await agent.evaluate_business_summary(agent_response, ground_truth)
        
        assert result.score == 100.0
        assert result.summary_quality == 1.0
        assert all(result.key_elements_found.values())
    
    @pytest.mark.asyncio
    async def test_partial_elements(self, agent):
        """Test scoring when some elements are missing."""
        ground_truth = {
            "industry": "Healthcare",
            "products": "Medical devices",
            "geography": "Global"
        }
        agent_response = '''{
            "business_summary": {
                "industry": "Healthcare equipment and supplies",
                "products": "Short",
                "geography": ""
            }
        }'''
        
        result = await agent.evaluate_business_summary(agent_response, ground_truth)
        
        # Only industry has > 10 chars
        assert result.key_elements_found["industry"] is True
        assert result.key_elements_found["products"] is False
        assert result.key_elements_found["geography"] is False
        assert result.score == pytest.approx(33.33, rel=0.1)
    
    @pytest.mark.asyncio
    async def test_invalid_json(self, agent):
        """Test handling of invalid JSON."""
        ground_truth = {
            "industry": "Technology",
            "products": "Software",
            "geography": "USA"
        }
        agent_response = "Not JSON"
        
        result = await agent.evaluate_business_summary(agent_response, ground_truth)
        
        assert result.score == 0.0
        assert "not valid JSON" in result.feedback


class TestConsistencyCheckScoring:
    """Test consistency check evaluation."""
    
    @pytest.mark.asyncio
    async def test_all_risks_discussed(self, agent):
        """Test when all risks are correctly identified as discussed."""
        risks_1a = ["COVID-19 impact", "Market volatility", "Regulatory changes"]
        discussed_in_7 = ["COVID-19 impact", "Market volatility", "Regulatory changes"]
        agent_response = '''{
            "consistency_check": ["COVID-19 impact", "Market volatility", "Regulatory changes"]
        }'''
        
        result = await agent.evaluate_consistency(agent_response, risks_1a, discussed_in_7)
        
        assert result.score == 100.0
        assert result.consistency_rate == 1.0
    
    @pytest.mark.asyncio
    async def test_partial_consistency(self, agent):
        """Test partial consistency identification."""
        risks_1a = ["Risk A", "Risk B", "Risk C", "Risk D"]
        discussed_in_7 = ["Risk A", "Risk C"]  # Only 2 out of 4
        agent_response = '{"consistency_check": ["Risk A", "Risk C"]}'
        
        result = await agent.evaluate_consistency(agent_response, risks_1a, discussed_in_7)
        
        # Correctly identified 2 out of 4 = 50%
        assert result.score == 50.0
        assert result.consistency_rate == 0.5
    
    @pytest.mark.asyncio
    async def test_no_risks(self, agent):
        """Test edge case with no risks."""
        risks_1a = []
        discussed_in_7 = []
        agent_response = '{"consistency_check": []}'
        
        result = await agent.evaluate_consistency(agent_response, risks_1a, discussed_in_7)
        
        # Should not divide by zero
        assert result.score == 100.0
        assert result.consistency_rate == 1.0
    
    @pytest.mark.asyncio
    async def test_case_insensitive_consistency(self, agent):
        """Test that consistency check is case-insensitive."""
        risks_1a = ["COVID-19 Impact", "Market Volatility"]
        discussed_in_7 = ["covid-19 impact", "MARKET VOLATILITY"]
        agent_response = '{"consistency_check": ["Covid-19 Impact", "market volatility"]}'
        
        result = await agent.evaluate_consistency(agent_response, risks_1a, discussed_in_7)
        
        assert result.score == 100.0


class TestSafeJsonParse:
    """Test safe JSON parsing utility."""
    
    def test_parse_clean_json(self):
        """Test parsing clean JSON."""
        from src.agent import safe_json_parse
        
        json_str = '{"key": "value"}'
        result = safe_json_parse(json_str)
        
        assert result == {"key": "value"}
    
    def test_parse_markdown_json(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        from src.agent import safe_json_parse
        
        json_str = '''```json
{
  "key": "value",
  "number": 42
}
```'''
        result = safe_json_parse(json_str)
        
        assert result == {"key": "value", "number": 42}
    
    def test_parse_generic_code_block(self):
        """Test parsing JSON in generic code block."""
        from src.agent import safe_json_parse
        
        json_str = '''```
{"key": "value"}
```'''
        result = safe_json_parse(json_str)
        
        assert result == {"key": "value"}
    
    def test_parse_invalid_json_raises(self):
        """Test that invalid JSON raises an error."""
        from src.agent import safe_json_parse
        import json
        
        with pytest.raises(json.JSONDecodeError):
            safe_json_parse("This is not JSON")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
