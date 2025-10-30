"""
LongMemEval Benchmark Evaluation

This script evaluates the Entity-Aware Memory System on the LongMemEval benchmark,
which tests five core long-term memory abilities:
1. Information extraction
2. Multi-session reasoning
3. Temporal reasoning
4. Knowledge updates
5. Abstention

Dataset: LongMemEval-S (~115k tokens, ~40 sessions per instance, 500 questions)
Source: https://github.com/xiaowu0162/LongMemEval
"""

import json
import os
import sys
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Any
from pathlib import Path
import time
import asyncio
import subprocess
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memory import TemporalSemanticMemory
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark")
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Limit number of instances to evaluate (default: all 500)"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions per instance (for quick testing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=100,
        help="Thinking budget for spreading activation search"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of memory units to retrieve per query"
    )
    return parser.parse_args()


def download_dataset(dataset_path: Path) -> bool:
    """
    Download the LongMemEval dataset if it doesn't exist.

    Returns:
        True if successful, False otherwise
    """
    url = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"

    console.print(f"[yellow]Dataset not found. Downloading from HuggingFace...[/yellow]")
    console.print(f"[dim]URL: {url}[/dim]")
    console.print(f"[dim]Destination: {dataset_path}[/dim]")

    try:
        # Use curl to download with progress
        result = subprocess.run(
            ["curl", "-L", "-o", str(dataset_path), url],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0 and dataset_path.exists():
            console.print(f"[green]✓ Dataset downloaded successfully[/green]")
            return True
        else:
            console.print(f"[red]✗ Download failed: {result.stderr}[/red]")
            return False

    except subprocess.TimeoutExpired:
        console.print(f"[red]✗ Download timed out after 5 minutes[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ Download error: {e}[/red]")
        return False


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load LongMemEval dataset from JSON file."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime object."""
    try:
        # LongMemEval format: "2023/05/20 (Sat) 02:21"
        # Try to parse the main part before the day name
        date_str_cleaned = date_str.split('(')[0].strip() if '(' in date_str else date_str

        # Try multiple formats
        for fmt in ["%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"]:
            try:
                dt = datetime.strptime(date_str_cleaned, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        # Fallback: try ISO format
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to parse date '{date_str}': {e}[/yellow]")
        return datetime.now(timezone.utc)


async def ingest_conversation(memory: TemporalSemanticMemory, agent_id: str, instance: Dict[str, Any]) -> None:
    """
    Ingest conversation history into memory system.

    Args:
        memory: Memory system instance
        agent_id: Unique agent ID for this conversation
        instance: LongMemEval instance containing haystack_sessions
    """
    # LongMemEval format: list of sessions, each session is a list of turn dicts
    sessions = instance.get("haystack_sessions", [])
    dates = instance.get("haystack_dates", [])
    session_ids = instance.get("haystack_session_ids", [])

    # Ensure all lists have same length
    if not (len(sessions) == len(dates) == len(session_ids)):
        console.print(f"[yellow]Warning: Mismatched lengths - sessions:{len(sessions)}, dates:{len(dates)}, ids:{len(session_ids)}[/yellow]")
        min_len = min(len(sessions), len(dates), len(session_ids))
        sessions = sessions[:min_len]
        dates = dates[:min_len]
        session_ids = session_ids[:min_len]

    # Process each session - combine all turns into one put_async call
    for session_turns, date_str, session_id in zip(sessions, dates, session_ids):
        # Parse session date
        session_date = parse_date(date_str) if date_str else datetime.now(timezone.utc)

        # Combine all turns in the session into one content string
        session_content_parts = []
        for turn_dict in session_turns:
            role = turn_dict.get("role", "")
            content = turn_dict.get("content", "")

            if not content.strip():
                continue

            # Format as "role: content" for clarity
            session_content_parts.append(f"{role}: {content}")

        # Ingest entire session as one chunk
        if session_content_parts:
            session_content = "\n".join(session_content_parts)
            context = f"Session {session_id}"

            try:
                await memory.put_async(
                    agent_id=agent_id,
                    content=session_content,
                    context=context,
                    event_date=session_date
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to ingest session {session_id}: {e}[/yellow]")


async def retrieve_memories(
    memory: TemporalSemanticMemory,
    agent_id: str,
    query: str,
    thinking_budget: int,
    top_k: int
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant memories for a query.

    Args:
        memory: Memory system instance
        agent_id: Agent ID
        query: Query text
        thinking_budget: Thinking budget for search
        top_k: Number of results to return

    Returns:
        List of retrieved memory units
    """
    try:
        results = await memory.search_async(
            agent_id=agent_id,
            query=query,
            thinking_budget=thinking_budget,
            top_k=top_k
        )
        return results
    except Exception as e:
        console.print(f"[yellow]Warning: Search failed: {e}[/yellow]")
        return []


def generate_answer(
    client: OpenAI,
    question: str,
    memories: List[Dict[str, Any]],
    model: str = "gpt-4o-mini"
) -> str:
    """
    Generate answer to question using retrieved memories.

    Args:
        client: OpenAI client
        question: Question text
        memories: Retrieved memory units
        model: OpenAI model to use

    Returns:
        Generated answer
    """
    # Format memories as context
    context_parts = []
    for i, mem in enumerate(memories, 1):
        context_parts.append(f"[Memory {i}] {mem['text']}")

    context = "\n".join(context_parts) if context_parts else "No relevant memories found."

    prompt = f"""You are a helpful assistant. Based on the following memories from past conversations, answer the question.

Memories:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided memories
- If the memories don't contain the answer, say "I don't have enough information to answer this question"
- Be concise and direct
- If asked to abstain (e.g., for unanswerable questions), explicitly say you cannot answer

Answer:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        console.print(f"[yellow]Warning: Answer generation failed: {e}[/yellow]")
        return "Error generating answer"


def evaluate_answer(
    client: OpenAI,
    question: str,
    predicted_answer: str,
    gold_answer: str,
    model: str = "gpt-4o"
) -> Dict[str, Any]:
    """
    Evaluate predicted answer against gold answer using LLM-as-judge.

    Args:
        client: OpenAI client
        question: Question text
        predicted_answer: Predicted answer
        gold_answer: Gold answer
        model: OpenAI model to use for evaluation

    Returns:
        Evaluation result with score and explanation
    """
    prompt = f"""You are an expert evaluator. Evaluate if the predicted answer is semantically equivalent to the gold answer.

Question: {question}

Gold Answer: {gold_answer}

Predicted Answer: {predicted_answer}

Instructions:
- Score 1 if the predicted answer is semantically equivalent (same meaning, different wording is OK)
- Score 1 if the predicted answer correctly abstains when the gold answer indicates the question is unanswerable
- Score 0 if the predicted answer is incorrect or contradicts the gold answer
- Score 0 if the predicted answer provides an answer when it should abstain
- Provide a brief explanation

Output format:
Score: [0 or 1]
Explanation: [brief explanation]"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200
        )

        content = response.choices[0].message.content.strip()

        # Parse score and explanation
        lines = content.split('\n')
        score = 0
        explanation = ""

        for line in lines:
            if line.startswith("Score:"):
                score_str = line.replace("Score:", "").strip()
                score = int(score_str) if score_str.isdigit() else 0
            elif line.startswith("Explanation:"):
                explanation = line.replace("Explanation:", "").strip()

        return {
            "score": score,
            "explanation": explanation,
            "raw_output": content
        }
    except Exception as e:
        console.print(f"[yellow]Warning: Evaluation failed: {e}[/yellow]")
        return {
            "score": 0,
            "explanation": f"Evaluation error: {str(e)}",
            "raw_output": ""
        }


def run_benchmark(args):
    """Run the LongMemEval benchmark evaluation."""
    console.print("\n[bold cyan]LongMemEval Benchmark Evaluation[/bold cyan]\n")

    # Load dataset - download if needed
    dataset_path = Path(__file__).parent / "longmemeval_s_cleaned.json"
    if not dataset_path.exists():
        if not download_dataset(dataset_path):
            console.print(f"[red]Failed to download dataset. Please download manually:[/red]")
            console.print("[yellow]curl -L 'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json' -o benchmarks/longmemeval/longmemeval_s_cleaned.json[/yellow]")
            return

    console.print(f"[green]Loading dataset from {dataset_path}[/green]")
    dataset = load_dataset(dataset_path)

    if args.max_instances:
        dataset = dataset[:args.max_instances]
        console.print(f"[yellow]Limited to {args.max_instances} instances[/yellow]")

    console.print(f"Dataset size: {len(dataset)} instances\n")

    # Initialize memory system
    console.print("[cyan]Initializing memory system...[/cyan]")
    memory = TemporalSemanticMemory()

    # Initialize OpenAI client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        console.print("[red]Error: OPENAI_API_KEY not set[/red]")
        return

    client = OpenAI(api_key=openai_api_key)

    # Results storage
    results = []

    # Process each instance
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:

        instance_task = progress.add_task("[cyan]Processing instances...", total=len(dataset))

        for idx, instance in enumerate(dataset):
            question_id = instance.get("question_id", f"q_{idx}")
            question = instance.get("question", "")
            gold_answer = instance.get("answer", "")
            question_type = instance.get("question_type", "unknown")

            progress.update(instance_task, description=f"[cyan]Instance {idx+1}/{len(dataset)}: {question_id}")

            # Use single agent for all LongMemEval data (cleared per question for isolation)
            agent_id = "longmemeval"

            # Clear agent data for this question (each question needs fresh isolated context)
            memory.delete_agent(agent_id)

            # Ingest conversation history
            try:
                asyncio.run(ingest_conversation(memory, agent_id, instance))
            except Exception as e:
                console.print(f"[red]Error ingesting instance {question_id}: {e}[/red]")
                continue

            # Retrieve memories
            memories = asyncio.run(retrieve_memories(
                memory,
                agent_id,
                question,
                args.thinking_budget,
                args.top_k
            ))

            # Generate answer
            predicted_answer = generate_answer(client, question, memories)

            # Evaluate answer
            evaluation = evaluate_answer(client, question, predicted_answer, gold_answer)

            # Store result
            result = {
                "question_id": question_id,
                "question_type": question_type,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "score": evaluation["score"],
                "explanation": evaluation["explanation"],
                "num_memories_retrieved": len(memories),
                "memory_texts": [m["text"] for m in memories[:5]]  # Store top 5 for debugging
            }
            results.append(result)

            progress.update(instance_task, advance=1)

            # Save intermediate results
            if (idx + 1) % 10 == 0:
                save_results(results, args.output)

    # Save final results
    save_results(results, args.output)

    # Display summary
    display_summary(results)


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSON file."""
    output_file = Path(__file__).parent / output_path
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]Results saved to {output_file}[/green]")


def display_summary(results: List[Dict[str, Any]]):
    """Display benchmark summary."""
    console.print("\n[bold cyan]Benchmark Summary[/bold cyan]\n")

    # Overall accuracy
    total = len(results)
    correct = sum(1 for r in results if r["score"] == 1)
    accuracy = (correct / total * 100) if total > 0 else 0

    table = Table(title="Overall Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Questions", str(total))
    table.add_row("Correct", str(correct))
    table.add_row("Incorrect", str(total - correct))
    table.add_row("Accuracy", f"{accuracy:.2f}%")

    console.print(table)

    # Accuracy by question type
    type_stats = {}
    for result in results:
        qtype = result["question_type"]
        if qtype not in type_stats:
            type_stats[qtype] = {"total": 0, "correct": 0}
        type_stats[qtype]["total"] += 1
        type_stats[qtype]["correct"] += result["score"]

    type_table = Table(title="Performance by Question Type")
    type_table.add_column("Question Type", style="cyan")
    type_table.add_column("Total", style="yellow")
    type_table.add_column("Correct", style="green")
    type_table.add_column("Accuracy", style="green")

    for qtype, stats in sorted(type_stats.items()):
        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        type_table.add_row(
            qtype,
            str(stats["total"]),
            str(stats["correct"]),
            f"{acc:.2f}%"
        )

    console.print("\n")
    console.print(type_table)


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
