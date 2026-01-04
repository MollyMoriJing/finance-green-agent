import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the Multi-Task Finance Green Agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    # Agent skill definition
    skill = AgentSkill(
        id="multi-task-financial-analysis",
        name="10-K Multi-Task Financial Analysis",
        description="Evaluates AI agents on three financial analysis tasks: (1) Risk factor classification from Section 1A, "
                    "(2) Business summary generation from Section 1, (3) Cross-section consistency checking between Section 1A and Section 7",
        tags=["finance", "multi-task", "risk-analysis", "business-summary", "consistency-check", "10-K", "benchmark"],
        examples=[
            "Evaluate agent on multi-task analysis for 2020 filings",
            "Test comprehensive 10-K document understanding across risk, business, and consistency tasks"
        ]
    )

    agent_card = AgentCard(
        name="finance-multi-task-analyst",
        description="Green agent that evaluates purple agents on comprehensive SEC 10-K analysis across three tasks: "
                    "(1) Risk Classification (40% weight) - Identify and categorize risk factors from Section 1A, "
                    "(2) Business Summary (30% weight) - Extract key business information from Section 1, "
                    "(3) Consistency Check (30% weight) - Verify risks mentioned in Section 1A are discussed in Section 7. "
                    "Uses 900 real 10-K filings (2015-2020). Overall score is weighted average across all tasks.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='2.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print(f"Starting Finance Multi-Task Green Agent on {args.host}:{args.port}")
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
