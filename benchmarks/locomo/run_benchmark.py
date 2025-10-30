"""
LoComo Benchmark Runner for Entity-Aware Memory System

Evaluates the memory system on the LoComo (Long-term Conversational Memory) benchmark.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from datetime import datetime, timezone, timedelta
from memory import TemporalSemanticMemory
from typing import List, Dict
from openai import AsyncOpenAI
import openai
from dotenv import load_dotenv
import os
import asyncio
import pydantic
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich import box

load_dotenv()

console = Console()


def get_groq_client() -> AsyncOpenAI:
    """
    Get configured async Groq client for LLM judge.

    Returns:
        Configured AsyncOpenAI client pointing to Groq
    """
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")

    base_url = os.getenv('GROQ_BASE_URL', 'https://api.groq.com/openai/v1')
    return AsyncOpenAI(
        api_key=groq_api_key,
        base_url=base_url
    )


def parse_date(date_string: str) -> datetime:
    """Parse LoComo date format to datetime."""
    # Format: "1:56 pm on 8 May, 2023"
    try:
        dt = datetime.strptime(date_string, "%I:%M %p on %d %B, %Y")
        return dt.replace(tzinfo=timezone.utc)
    except:
        return datetime.now(timezone.utc)


async def ingest_conversation(memory: TemporalSemanticMemory, conversation_data: Dict, agent_id: str):
    """
    Ingest a LoComo conversation into the memory system (ASYNC version).

    Ingests ALL sessions in ONE batch for maximum efficiency.

    Args:
        memory: Memory system instance
        conversation_data: Conversation data from LoComo
        agent_id: Agent ID to use
    """
    conv = conversation_data['conversation']
    speaker_a = conv['speaker_a']
    speaker_b = conv['speaker_b']

    # Get all session keys sorted
    session_keys = sorted([k for k in conv.keys() if k.startswith('session_') and not k.endswith('_date_time')])

    # Collect all sessions as batch items
    batch_contents = []
    total_turns = 0

    for session_key in session_keys:
        if session_key not in conv or not isinstance(conv[session_key], list):
            continue

        session_data = conv[session_key]

        # Build session content from all turns
        session_parts = []
        for turn in session_data:
            speaker = turn['speaker']
            text = turn['text']
            session_parts.append(f"{speaker}: {text}")
            total_turns += 1

        if not session_parts:
            continue

        # Get session date
        date_key = f"{session_key}_date_time"
        session_date = parse_date(conv.get(date_key, "1:00 pm on 1 January, 2023"))

        # Add to batch
        session_content = "\n".join(session_parts)
        batch_contents.append({
            "content": session_content,
            "context": f"Conversation session between {speaker_a} and {speaker_b}",
            "event_date": session_date
        })

    # Ingest ALL sessions in ONE batch call (MUCH faster!)
    if batch_contents:
        await memory.put_batch_async(
            agent_id=agent_id,
            contents=batch_contents
        )

    return total_turns

class QuestionAnswer(pydantic.BaseModel):
    answer: str
    reasoning: str

async def answer_question(memory: TemporalSemanticMemory, agent_id: str, question: str, thinking_budget: int = 500) -> tuple[str, str, List[Dict]]:
    """
    Answer a question using the memory system (ASYNC version).

    Args:
        memory: Memory system instance
        agent_id: Agent ID
        question: Question to answer
        thinking_budget: How many memory units to explore

    Returns:
        Tuple of (answer string, reasoning string, retrieved memories list)
    """
    # Search memory
    results = await memory.search_async(
        agent_id=agent_id,
        query=question,
        thinking_budget=thinking_budget,
        top_k=20  # Get more results for better context
    )
    if not results:
        return "I don't have enough information to answer that question.", "No relevant memories found.", []

    context_parts = []
    for i, result in enumerate(results):
        context_parts.append(f"{i}. {result['text']}")

    context = "\n".join(context_parts)

    # Use AsyncOpenAI to generate answer from context
    try:
        client = AsyncOpenAI()
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer the question based ONLY on the provided context. If the context doesn't contain the answer, say 'I don't know'. In the reasoning, explain why you choose or not choose the context items for the answer."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                }
            ],
            temperature=0,
            max_tokens=8000,
            response_format=QuestionAnswer
        )
        answer = response.choices[0].message.parsed
        return answer.answer, answer.reasoning, results
    except Exception as e:
        return f"Error generating answer: {str(e)}", "Error occurred during answer generation.", results


async def evaluate_qa_task(
    memory: TemporalSemanticMemory,
    agent_id: str,
    qa_pairs: List[Dict],
    sample_id: str,
    max_questions: int = None
) -> Dict:
    """
    Evaluate the QA task (ASYNC version - processes questions in parallel).

    Returns:
        Dict with evaluation metrics
    """
    questions_to_eval = qa_pairs[:max_questions] if max_questions else qa_pairs

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]Evaluating QA for sample {sample_id} (parallel)...", total=len(questions_to_eval))

        # Create tasks for all questions
        async def process_question(qa):
            question = qa['question']
            correct_answer = qa['answer']
            category = qa.get('category', 0)

            # Get predicted answer, reasoning, and retrieved memories
            predicted_answer, reasoning, retrieved_memories = await answer_question(memory, agent_id, question)

            return {
                'question': question,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'reasoning': reasoning,
                'category': category,
                'retrieved_memories': retrieved_memories
            }

        # Process all questions in parallel
        question_tasks = [process_question(qa) for qa in questions_to_eval]

        # Use as_completed to update progress as results come in
        results = []
        for coro in asyncio.as_completed(question_tasks):
            result = await coro
            results.append(result)
            progress.update(task, advance=1)

    return results

class JudgeResponse(pydantic.BaseModel):
    correct: bool
    reasoning: str

async def judge_single_answer(client: AsyncOpenAI, result: Dict, semaphore: asyncio.Semaphore) -> Dict:
    """
    Judge a single answer using LLM (with concurrency control).

    Args:
        client: Async OpenAI client (Groq)
        result: Result dict with question, correct_answer, predicted_answer, category
        semaphore: Semaphore to limit concurrent requests

    Returns:
        Updated result dict with is_correct field
    """
    async with semaphore:
        try:
            response = await client.beta.chat.completions.parse(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an objective judge. Determine if the predicted answer contains the correct answer or they are the same content (with different form is fine)."
                    },
                    {
                        "role": "user",
                        "content": f"Question: {result['question']}\nCorrect answer: {result['correct_answer']}\nPredicted answer: {result['predicted_answer']}\n\nAre they equivalent?"
                    }
                ],
                temperature=0,
                max_tokens=512,
                response_format=JudgeResponse

            )

            judgement = response.choices[0].message.parsed
            result['is_correct'] = judgement.correct
            result['correctness_reasoning'] = judgement.reasoning

        except Exception as e:
            console.print(f"[red]Error judging answer: {e}[/red]")
            result['is_correct'] = False

    return result


async def calculate_metrics(results: List[Dict]) -> Dict:
    """
    Calculate evaluation metrics using parallel LLM-as-judge.

    Processes up to 8 judgments concurrently for speed.
    """
    total = len(results)
    client = get_groq_client()

    # Semaphore to limit to 8 concurrent requests
    semaphore = asyncio.Semaphore(8)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("[yellow]Judging answers with LLM (parallel, max 8)...", total=total)

        # Create all judgment tasks
        judgment_tasks = []
        for result in results:
            judgment_task = judge_single_answer(client, result, semaphore)
            judgment_tasks.append(judgment_task)

        # Process in parallel with progress updates
        judged_results = []
        for coro in asyncio.as_completed(judgment_tasks):
            judged_result = await coro
            judged_results.append(judged_result)
            progress.update(task, advance=1)

    # Calculate stats
    correct = sum(1 for r in judged_results if r.get('is_correct', False))
    category_stats = {}

    for result in judged_results:
        category = result['category']
        if category not in category_stats:
            category_stats[category] = {'correct': 0, 'total': 0}
        category_stats[category]['total'] += 1
        if result.get('is_correct', False):
            category_stats[category]['correct'] += 1

    accuracy = (correct / total * 100) if total > 0 else 0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'category_stats': category_stats,
        'detailed_results': judged_results
    }


async def process_single_conversation(
    memory: TemporalSemanticMemory,
    conv_data: Dict,
    i: int,
    total_convs: int,
    max_questions_per_conv: int,
    skip_ingestion: bool
) -> Dict:
    """
    Process a single conversation (ingest + evaluate).

    Args:
        memory: Memory system instance
        conv_data: Conversation data
        i: Conversation index (1-based)
        total_convs: Total number of conversations
        max_questions_per_conv: Max questions to evaluate per conversation
        skip_ingestion: Whether to skip ingestion

    Returns:
        Result dict with sample_id, metrics, total_turns
    """
    sample_id = conv_data['sample_id']
    agent_id = "locomo"  # Single agent for all Locomo benchmark data

    console.print(f"\n[bold blue]Conversation {i}/{total_convs}[/bold blue] (Sample ID: {sample_id})")

    if not skip_ingestion:
        # Clear previous locomo agent data only (multi-tenant safe)
        if i == 1:  # Only cleanup on first conversation
            console.print("  [2] Clearing previous 'locomo' agent data...")
            memory.delete_agent(agent_id)
            console.print(f"      [green]✓[/green] Cleared 'locomo' agent data")

        # Ingest conversation (sessions processed in parallel)
        console.print("  [3] Ingesting conversation (sessions in parallel)...")
        total_turns = await ingest_conversation(memory, conv_data, agent_id)
        console.print(f"      [green]✓[/green] Ingested {total_turns} turns across multiple sessions")
    else:
        total_turns = -1

    # Evaluate QA (async - questions processed in parallel)
    console.print(f"  [4] Evaluating {len(conv_data['qa'])} QA pairs (parallel)...")
    qa_results = await evaluate_qa_task(
        memory,
        agent_id,
        conv_data['qa'],
        sample_id,
        max_questions=max_questions_per_conv
    )

    # Calculate metrics (async with parallel LLM judging)
    console.print("  [5] Calculating metrics...")
    metrics = await calculate_metrics(qa_results)

    console.print(f"      [green]✓[/green] Accuracy: {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['total']})")

    return {
        'sample_id': sample_id,
        'metrics': metrics,
        'total_turns': total_turns
    }


def run_benchmark(max_conversations: int = None, max_questions_per_conv: int = None, skip_ingestion: bool = False):
    """
    Run the LoComo benchmark.

    Args:
        max_conversations: Maximum number of conversations to evaluate (None for all)
        max_questions_per_conv: Maximum questions per conversation (None for all)
        skip_ingestion: Whether to skip ingestion and use existing data
    """
    console.print("\n[bold cyan]LoComo Benchmark - Entity-Aware Memory System[/bold cyan]")
    console.print("=" * 80)

    # Load dataset
    console.print("\n[1] Loading LoComo dataset...")
    with open('locomo10.json', 'r') as f:
        dataset = json.load(f)

    conversations_to_eval = dataset[:max_conversations] if max_conversations else dataset
    console.print(f"    [green]✓[/green] Loaded {len(conversations_to_eval)} conversations")

    # Initialize memory system
    console.print("\n[2] Initializing memory system...")
    memory = TemporalSemanticMemory()
    console.print("    [green]✓[/green] Memory system initialized")

    # Run evaluation (conversations sequential, sessions within each conversation parallel)
    all_results = []

    for i, conv_data in enumerate(conversations_to_eval, 1):
        result = asyncio.run(
            process_single_conversation(
                memory, conv_data, i, len(conversations_to_eval),
                max_questions_per_conv, skip_ingestion
            )
        )
        all_results.append(result)

    # Overall results
    console.print("\n[bold green]✓ Benchmark Complete![/bold green]\n")

    # Calculate overall metrics
    total_correct = sum(r['metrics']['correct'] for r in all_results)
    total_questions = sum(r['metrics']['total'] for r in all_results)
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0

    # Display results table
    table = Table(title="LoComo Benchmark Results", box=box.ROUNDED)
    table.add_column("Sample ID", style="cyan")
    table.add_column("Turns", justify="right", style="yellow")
    table.add_column("Questions", justify="right", style="blue")
    table.add_column("Correct", justify="right", style="green")
    table.add_column("Accuracy", justify="right", style="magenta")

    for result in all_results:
        metrics = result['metrics']
        table.add_row(
            result['sample_id'],
            str(result['total_turns']),
            str(metrics['total']),
            str(metrics['correct']),
            f"{metrics['accuracy']:.1f}%"
        )

    table.add_row(
        "[bold]OVERALL[/bold]",
        "-",
        f"[bold]{total_questions}[/bold]",
        f"[bold]{total_correct}[/bold]",
        f"[bold]{overall_accuracy:.1f}%[/bold]"
    )

    console.print(table)

    return {
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_questions': total_questions,
        'conversation_results': all_results
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run LoComo benchmark')
    parser.add_argument('--max-conversations', type=int, default=None, help='Maximum conversations to evaluate')
    parser.add_argument('--max-questions', type=int, default=None, help='Maximum questions per conversation')
    parser.add_argument('--skip-ingestion', action='store_true', help='Skip ingestion and use existing data')

    args = parser.parse_args()

    results = run_benchmark(
        max_conversations=args.max_conversations,
        max_questions_per_conv=args.max_questions,
        skip_ingestion=args.skip_ingestion
    )

    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]✓[/green] Results saved to benchmark_results.json")
