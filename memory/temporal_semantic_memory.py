"""
Temporal + Semantic + Entity Memory System for AI Agents.

This implements a sophisticated memory architecture that combines:
1. Temporal links: Memories connected by time proximity
2. Semantic links: Memories connected by meaning/similarity
3. Entity links: Memories connected by shared entities (PERSON, ORG, etc.)
4. Spreading activation: Search through the graph with activation decay
5. Dynamic weighting: Recency and frequency-based importance
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import asyncio
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from .utils import (
    extract_facts,
    calculate_recency_weight,
    calculate_frequency_weight,
)
from .entity_resolver import EntityResolver


def utcnow():
    """Get current UTC time with timezone info."""
    return datetime.now(timezone.utc)


# Global process pool for parallel embedding generation
# Each process loads its own copy of the embedding model
# This provides TRUE parallelism for CPU-bound embedding operations
_PROCESS_POOL = None
_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Process-local model cache (one per worker process)
_worker_model = None


def _get_worker_model():
    """Get or load the embedding model in worker process."""
    global _worker_model
    if _worker_model is None:
        _worker_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _worker_model


def _encode_batch_worker(texts: List[str]) -> List[List[float]]:
    """
    Worker function for process pool - encodes texts to embeddings.

    This function runs in a separate process and loads its own model.
    """
    model = _get_worker_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


def _get_process_pool():
    """Get or create the global process pool."""
    global _PROCESS_POOL
    if _PROCESS_POOL is None:
        # Use 4 worker processes for true parallelism
        # Adjust based on your CPU cores (each process loads ~500MB model)
        _PROCESS_POOL = ProcessPoolExecutor(max_workers=4)
    return _PROCESS_POOL


class TemporalSemanticMemory:
    """
    Advanced memory system using temporal and semantic linking with PostgreSQL.
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        """
        Initialize the temporal + semantic memory system.

        Args:
            db_url: PostgreSQL connection URL (postgresql://user:pass@host:port/dbname)
            embedding_model: Name of the SentenceTransformer model to use
        """
        load_dotenv()

        # Initialize PostgreSQL connection
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError(
                "Database URL not found. "
                "Set DATABASE_URL environment variable."
            )

        self.conn = psycopg2.connect(self.db_url)
        register_vector(self.conn)

        # Initialize entity resolver
        self.entity_resolver = EntityResolver(self.conn)

        # Initialize local embedding model (384 dimensions)
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"✓ Model loaded (embedding dim: {self.embedding_model.get_sentence_embedding_dimension()})")

    def __del__(self):
        """Clean up database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using local SentenceTransformer model.

        Args:
            text: Text to embed

        Returns:
            384-dimensional embedding vector (bge-small-en-v1.5)
        """
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            return embedding.tolist()
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")

    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using local model in parallel.

        Uses a ProcessPoolExecutor to achieve TRUE parallelism for CPU-bound
        embedding generation. Each worker process loads its own model copy.

        When multiple put_async calls run in parallel, each can generate
        embeddings concurrently in separate processes (no GIL contention).

        Args:
            texts: List of texts to embed

        Returns:
            List of 384-dimensional embeddings in same order as input texts
        """
        try:
            # Run in process pool for true parallelism
            loop = asyncio.get_event_loop()
            pool = _get_process_pool()
            embeddings = await loop.run_in_executor(
                pool,
                _encode_batch_worker,
                texts
            )
            return embeddings
        except Exception as e:
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")

    def _find_duplicate_facts_batch(
        self,
        cursor,
        agent_id: str,
        texts: List[str],
        embeddings: List[List[float]],
        event_date: datetime,
        time_window_hours: int = 24,
        similarity_threshold: float = 0.95
    ) -> List[bool]:
        """
        Check which facts are duplicates using semantic similarity + temporal window.

        For each new fact, checks if a semantically similar fact already exists
        within the time window. Uses pgvector cosine similarity for efficiency.

        Args:
            cursor: Database cursor
            agent_id: Agent identifier
            texts: List of fact texts to check
            embeddings: Corresponding embeddings
            event_date: Event date for temporal filtering
            time_window_hours: Hours before/after event_date to search (default: 24)
            similarity_threshold: Minimum cosine similarity to consider duplicate (default: 0.95)

        Returns:
            List of booleans - True if fact is a duplicate (should skip), False if new
        """
        is_duplicate = []

        time_lower = event_date - timedelta(hours=time_window_hours)
        time_upper = event_date + timedelta(hours=time_window_hours)

        for text, embedding in zip(texts, embeddings):
            # Query for similar facts within time window
            cursor.execute(
                """
                SELECT id, text, 1 - (embedding <=> %s::vector) AS similarity
                FROM memory_units
                WHERE agent_id = %s
                  AND event_date BETWEEN %s AND %s
                  AND 1 - (embedding <=> %s::vector) > %s
                ORDER BY similarity DESC
                LIMIT 1
                """,
                (embedding, agent_id, time_lower, time_upper, embedding, similarity_threshold)
            )

            result = cursor.fetchone()
            if result:
                is_duplicate.append(True)
            else:
                is_duplicate.append(False)

        return is_duplicate

    def put(
        self,
        agent_id: str,
        content: str,
        context: str = "",
        event_date: Optional[datetime] = None,
    ) -> List[str]:
        """
        Store content as memory units (synchronous wrapper).

        This is a synchronous wrapper around put_async() for convenience.
        For best performance, use put_async() directly.

        Args:
            agent_id: Unique identifier for the agent
            content: Text content to store
            context: Context about when/why this memory was formed
            event_date: When the event occurred (defaults to now)

        Returns:
            List of created unit IDs
        """
        # Run async version synchronously
        return asyncio.run(self.put_async(agent_id, content, context, event_date))

    async def put_async(
        self,
        agent_id: str,
        content: str,
        context: str = "",
        event_date: Optional[datetime] = None,
    ) -> List[str]:
        """
        Store content as memory units with temporal and semantic links (ASYNC version).

        This is a convenience wrapper around put_batch_async for a single content item.

        Args:
            agent_id: Unique identifier for the agent
            content: Text content to store
            context: Context about when/why this memory was formed
            event_date: When the event occurred (defaults to now)

        Returns:
            List of created unit IDs
        """
        # Use put_batch_async with a single item (avoids code duplication)
        result = await self.put_batch_async(
            agent_id=agent_id,
            contents=[{
                "content": content,
                "context": context,
                "event_date": event_date
            }]
        )

        # Return the first (and only) list of unit IDs
        return result[0] if result else []

    async def put_batch_async(
        self,
        agent_id: str,
        contents: List[Dict[str, Any]],
    ) -> List[List[str]]:
        """
        Store multiple content items as memory units in ONE batch operation.

        This is MUCH more efficient than calling put_async multiple times:
        - Extracts facts from all contents in parallel
        - Generates ALL embeddings in ONE batch
        - Does ALL database operations in ONE transaction

        Args:
            agent_id: Unique identifier for the agent
            contents: List of dicts with keys:
                - "content" (required): Text content to store
                - "context" (optional): Context about the memory
                - "event_date" (optional): When the event occurred

        Returns:
            List of lists of unit IDs (one list per content item)

        Example:
            unit_ids = await memory.put_batch_async(
                agent_id="user123",
                contents=[
                    {"content": "Alice works at Google", "context": "conversation"},
                    {"content": "Bob loves Python", "context": "conversation"},
                ]
            )
            # Returns: [["unit-id-1"], ["unit-id-2"]]
        """
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"PUT_BATCH_ASYNC START: {agent_id}")
        print(f"Batch size: {len(contents)} content items")
        print(f"{'='*60}")

        if not contents:
            return []

        # Step 1: Extract facts from ALL contents in parallel
        step_start = time.time()

        # Create tasks for parallel fact extraction
        fact_extraction_tasks = []
        for item in contents:
            content = item["content"]
            context = item.get("context", "")
            event_date = item.get("event_date") or utcnow()

            task = extract_facts(content, event_date, context)
            fact_extraction_tasks.append((task, event_date, context))

        # Wait for all fact extractions to complete
        all_fact_results = await asyncio.gather(*[task for task, _, _ in fact_extraction_tasks])

        # Flatten and track which facts belong to which content
        all_fact_texts = []
        all_fact_dates = []
        all_contexts = []
        content_boundaries = []  # [(start_idx, end_idx), ...]

        current_idx = 0
        for i, ((_, event_date, context), fact_dicts) in enumerate(zip(fact_extraction_tasks, all_fact_results)):
            start_idx = current_idx

            for fact_dict in fact_dicts:
                all_fact_texts.append(fact_dict['fact'])
                try:
                    from dateutil import parser as date_parser
                    fact_date = date_parser.isoparse(fact_dict['date'])
                    all_fact_dates.append(fact_date)
                except Exception:
                    all_fact_dates.append(event_date)
                all_contexts.append(context)

            end_idx = current_idx + len(fact_dicts)
            content_boundaries.append((start_idx, end_idx))
            current_idx = end_idx

        total_facts = len(all_fact_texts)

        if total_facts == 0:
            return [[] for _ in contents]

        # Step 2: Generate ALL embeddings in ONE batch (HUGE speedup!)
        step_start = time.time()
        all_embeddings = await self._generate_embeddings_batch(all_fact_texts)
        print(f"[2] Generate embeddings (parallel): {len(all_embeddings)} embeddings in {time.time() - step_start:.3f}s")

        # Step 3: Process everything in ONE database transaction
        cursor = self.conn.cursor()
        try:
            # Deduplication check for all facts
            step_start = time.time()
            all_is_duplicate = []
            for sentence, embedding, fact_date in zip(all_fact_texts, all_embeddings, all_fact_dates):
                dup_flags = self._find_duplicate_facts_batch(
                    cursor, agent_id, [sentence], [embedding], fact_date
                )
                all_is_duplicate.extend(dup_flags)

            duplicates_filtered = sum(all_is_duplicate)
            new_facts = total_facts - duplicates_filtered
            print(f"[3] Deduplication check: {duplicates_filtered} duplicates filtered, {new_facts} new facts in {time.time() - step_start:.3f}s")

            # Filter out duplicates
            filtered_sentences = [s for s, is_dup in zip(all_fact_texts, all_is_duplicate) if not is_dup]
            filtered_embeddings = [e for e, is_dup in zip(all_embeddings, all_is_duplicate) if not is_dup]
            filtered_dates = [d for d, is_dup in zip(all_fact_dates, all_is_duplicate) if not is_dup]
            filtered_contexts = [c for c, is_dup in zip(all_contexts, all_is_duplicate) if not is_dup]

            if not filtered_sentences:
                print(f"[PUT_BATCH_ASYNC] All facts were duplicates, returning empty")
                return [[] for _ in contents]

            # Batch insert ALL units
            step_start = time.time()
            from psycopg2.extras import execute_values
            unit_data = [
                (agent_id, sentence, context, embedding, date, 0)  # access_count starts at 0
                for sentence, context, embedding, date in zip(
                    filtered_sentences, filtered_contexts, filtered_embeddings, filtered_dates
                )
            ]

            results = execute_values(
                cursor,
                """
                INSERT INTO memory_units (agent_id, text, context, embedding, event_date, access_count)
                VALUES %s
                RETURNING id
                """,
                unit_data,
                fetch=True
            )

            created_unit_ids = [str(row[0]) for row in results]
            print(f"[5] Batch insert units: {len(created_unit_ids)} units in {time.time() - step_start:.3f}s")

            # Process entities for ALL units
            step_start = time.time()
            all_entity_links = self._extract_entities_batch_optimized(
                cursor, agent_id, created_unit_ids, filtered_sentences, "", filtered_dates
            )
            print(f"[6] Extract entities (batched): {time.time() - step_start:.3f}s")

            # Create temporal links
            step_start = time.time()
            self._create_temporal_links_batch_per_fact(cursor, agent_id, created_unit_ids)
            print(f"[7] Batch create temporal links: {time.time() - step_start:.3f}s")

            # Create semantic links
            step_start = time.time()
            self._create_semantic_links_batch(cursor, agent_id, created_unit_ids, filtered_embeddings)
            print(f"[8] Batch create semantic links: {time.time() - step_start:.3f}s")

            # Insert entity links
            step_start = time.time()
            if all_entity_links:
                self._insert_entity_links_batch(cursor, all_entity_links)
            print(f"[9] Batch insert entity links: {time.time() - step_start:.3f}s")

            # Commit everything
            commit_start = time.time()
            self.conn.commit()
            print(f"[10] Commit: {time.time() - commit_start:.3f}s")

            # Map created unit IDs back to original content items
            # Account for duplicates when mapping back
            result_unit_ids = []
            filtered_idx = 0

            for start_idx, end_idx in content_boundaries:
                content_unit_ids = []
                for i in range(start_idx, end_idx):
                    if not all_is_duplicate[i]:
                        content_unit_ids.append(created_unit_ids[filtered_idx])
                        filtered_idx += 1
                result_unit_ids.append(content_unit_ids)

            total_time = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"PUT_BATCH_ASYNC COMPLETE: {len(created_unit_ids)} units from {len(contents)} contents in {total_time:.3f}s")
            print(f"{'='*60}\n")

            return result_unit_ids

        except Exception as e:
            self.conn.rollback()
            raise Exception(f"Failed to store batch memory: {str(e)}")
        finally:
            cursor.close()

    def _create_temporal_links(
        self,
        cursor,
        agent_id: str,
        unit_id: str,
        event_date: datetime,
        time_window_hours: int = 24,
    ):
        """
        Create temporal links to recent memories.

        Links this unit to other units that occurred within a time window.

        Args:
            cursor: Database cursor
            agent_id: Agent ID
            unit_id: ID of the current unit
            event_date: When this event occurred
            time_window_hours: Size of the temporal window
        """
        try:
            # Get recent units within time window
            cursor.execute(
                """
                SELECT id, event_date
                FROM memory_units
                WHERE agent_id = %s
                  AND id != %s
                  AND event_date >= %s
                ORDER BY event_date DESC
                LIMIT 10
                """,
                (agent_id, unit_id, event_date - timedelta(hours=time_window_hours))
            )

            recent_units = cursor.fetchall()

            # Create links to recent units
            links = []
            for recent_id, recent_event_date in recent_units:
                # Calculate temporal proximity weight
                time_diff_hours = abs((event_date - recent_event_date).total_seconds() / 3600)
                weight = max(0.3, 1.0 - (time_diff_hours / time_window_hours))

                links.append((unit_id, recent_id, 'temporal', weight, None))

            if links:
                execute_values(
                    cursor,
                    """
                    INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, entity_id)
                    VALUES %s
                    ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                    """,
                    links
                )

        except Exception as e:
            print(f"ERROR: Failed to create temporal links: {str(e)}")
            import traceback
            traceback.print_exc()
            # Re-raise to trigger rollback at put_async level
            raise

    def _create_semantic_links(
        self,
        cursor,
        agent_id: str,
        unit_id: str,
        embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.7,
    ):
        """
        Create semantic links to similar memories.

        Links this unit to other units with similar meaning.

        Args:
            cursor: Database cursor
            agent_id: Agent ID
            unit_id: ID of the current unit
            embedding: Embedding of the current unit
            top_k: Number of similar units to link to
            threshold: Minimum similarity threshold
        """
        try:
            # Find similar units using vector similarity
            cursor.execute(
                """
                SELECT id, 1 - (embedding <=> %s::vector) AS similarity
                FROM memory_units
                WHERE agent_id = %s
                  AND id != %s
                  AND embedding IS NOT NULL
                  AND (1 - (embedding <=> %s::vector)) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, agent_id, unit_id, embedding, threshold, embedding, top_k)
            )

            similar_units = cursor.fetchall()

            # Create links to similar units
            links = []
            for similar_id, similarity in similar_units:
                links.append((unit_id, similar_id, 'semantic', float(similarity), None))

            if links:
                execute_values(
                    cursor,
                    """
                    INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, entity_id)
                    VALUES %s
                    ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                    """,
                    links
                )

        except Exception as e:
            print(f"ERROR: Failed to create semantic links: {str(e)}")
            import traceback
            traceback.print_exc()
            # Re-raise to trigger rollback at put_async level
            raise

    def search(
        self,
        agent_id: str,
        query: str,
        thinking_budget: int = 50,
        top_k: int = 10,
        live_tracer=None,
    ) -> List[Dict[str, Any]]:
        """
        Search memories using spreading activation (synchronous wrapper).

        This is a synchronous wrapper around search_async() for convenience.
        For best performance, use search_async() directly.

        Args:
            agent_id: Agent ID to search for
            query: Search query
            thinking_budget: How many units to explore (computational budget)
            top_k: Number of results to return
            live_tracer: Optional LiveSearchTracer for visualization

        Returns:
            List of memory units with their weights, sorted by relevance
        """
        # Run async version synchronously
        return asyncio.run(self.search_async(agent_id, query, thinking_budget, top_k, live_tracer))

    async def search_async(
        self,
        agent_id: str,
        query: str,
        thinking_budget: int = 50,
        top_k: int = 10,
        live_tracer=None,
    ) -> List[Dict[str, Any]]:
        """
        Search memories using spreading activation (ASYNC version).

        This implements the core SEARCH operation:
        1. Find entry points (most relevant units via vector search)
        2. Spread activation through the graph
        3. Weight results by activation + recency + frequency
        4. Return top results

        Args:
            agent_id: Agent ID to search for
            query: Search query
            thinking_budget: How many units to explore (computational budget)
            top_k: Number of results to return
            live_tracer: Optional LiveSearchTracer for visualization

        Returns:
            List of memory units with their weights, sorted by relevance
        """
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)

        search_start = time.time()
        print(f"\n[SEARCH] Starting search for query: '{query[:50]}...' (thinking_budget={thinking_budget}, top_k={top_k})")

        try:
            # Step 1: Generate query embedding
            step_start = time.time()
            query_embedding = self._generate_embedding(query)
            print(f"  [1] Generate query embedding: {time.time() - step_start:.3f}s")

            # Step 2: Find entry points
            step_start = time.time()
            cursor.execute(
                """
                SELECT id, text, context, event_date, access_count, embedding,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM memory_units
                WHERE agent_id = %s
                  AND embedding IS NOT NULL
                  AND (1 - (embedding <=> %s::vector)) >= 0.5
                ORDER BY embedding <=> %s::vector
                LIMIT 3
                """,
                (query_embedding, agent_id, query_embedding, query_embedding)
            )

            entry_points = cursor.fetchall()
            print(f"  [2] Find entry points: {len(entry_points)} found in {time.time() - step_start:.3f}s")

            if not entry_points:
                print(f"[SEARCH] Complete: 0 results in {time.time() - search_start:.3f}s")
                return []

            # Step 3: Spreading activation with budget
            step_start = time.time()
            visited = set()
            results = []
            budget_remaining = thinking_budget
            # Initialize entry points with their actual similarity scores instead of 1.0
            queue = [(dict(unit), unit["similarity"], True) for unit in entry_points]  # (unit, activation, is_entry)

            # Track substep timings
            update_access_time = 0
            calculate_weight_time = 0
            query_neighbors_time = 0
            process_neighbors_time = 0

            # Process nodes in batches for efficient neighbor querying
            BATCH_SIZE = 50
            nodes_to_process = []  # (unit, activation, is_entry_point)

            while queue and budget_remaining > 0:
                # Collect a batch of nodes to process
                while queue and len(nodes_to_process) < BATCH_SIZE and budget_remaining > 0:
                    current_unit, activation, is_entry_point = queue.pop(0)
                    unit_id = str(current_unit["id"])

                    if unit_id not in visited:
                        visited.add(unit_id)
                        budget_remaining -= 1
                        nodes_to_process.append((current_unit, activation, is_entry_point))

                if not nodes_to_process:
                    break

                # Update access counts for batch
                substep_start = time.time()
                node_ids = [str(node[0]["id"]) for node in nodes_to_process]
                cursor.execute(
                    "UPDATE memory_units SET access_count = access_count + 1 WHERE id::text = ANY(%s)",
                    (node_ids,)
                )
                update_access_time += time.time() - substep_start

                # Query neighbors for ALL nodes in batch at once
                substep_start = time.time()
                cursor.execute(
                    """
                    SELECT ml.from_unit_id, ml.to_unit_id, ml.weight,
                           mu.text, mu.context, mu.event_date, mu.access_count, mu.embedding
                    FROM memory_links ml
                    JOIN memory_units mu ON ml.to_unit_id = mu.id
                    WHERE ml.from_unit_id::text = ANY(%s)
                      AND ml.weight >= 0.1
                    ORDER BY ml.from_unit_id, ml.weight DESC
                    """,
                    (node_ids,)
                )
                all_neighbors = cursor.fetchall()
                query_neighbors_time += time.time() - substep_start

                # Group neighbors by from_unit_id
                substep_start = time.time()
                neighbors_by_node = {}
                for neighbor in all_neighbors:
                    from_id = str(neighbor["from_unit_id"])
                    if from_id not in neighbors_by_node:
                        neighbors_by_node[from_id] = []
                    neighbors_by_node[from_id].append(neighbor)

                # Process each node in the batch
                for current_unit, activation, is_entry_point in nodes_to_process:
                    unit_id = str(current_unit["id"])

                    # Calculate combined weight
                    event_date = current_unit["event_date"]
                    days_since = (utcnow() - event_date).total_seconds() / 86400

                    recency_weight = calculate_recency_weight(days_since)
                    frequency_weight = calculate_frequency_weight(current_unit.get("access_count", 0))

                    # Normalize frequency to [0, 1] range
                    frequency_normalized = (frequency_weight - 1.0) / 1.0

                    # Calculate semantic similarity between query and this memory
                    memory_embedding = current_unit.get("embedding")
                    if memory_embedding is not None:
                        # Cosine similarity = 1 - cosine distance
                        query_vec = np.array(query_embedding)
                        memory_vec = np.array(memory_embedding)
                        # Cosine similarity
                        dot_product = np.dot(query_vec, memory_vec)
                        norm_query = np.linalg.norm(query_vec)
                        norm_memory = np.linalg.norm(memory_vec)
                        semantic_similarity = dot_product / (norm_query * norm_memory) if norm_query > 0 and norm_memory > 0 else 0.0
                    else:
                        semantic_similarity = 0.0

                    # Combined weight: 30% activation, 30% semantic similarity, 25% recency, 15% frequency
                    final_weight = 0.3 * activation + 0.3 * semantic_similarity + 0.25 * recency_weight + 0.15 * frequency_normalized

                    # Notify tracer
                    if live_tracer:
                        live_tracer.visit_node(
                            node_id=unit_id,
                            text=current_unit["text"],
                            activation=activation,
                            recency=recency_weight,
                            frequency=frequency_weight,
                            weight=final_weight,
                            is_entry_point=is_entry_point,
                        )

                    results.append({
                        "id": unit_id,
                        "text": current_unit["text"],
                        "context": current_unit.get("context", ""),
                        "event_date": event_date.isoformat(),
                        "weight": final_weight,
                        "activation": activation,
                        "semantic_similarity": semantic_similarity,
                        "recency": recency_weight,
                        "frequency": frequency_weight,
                    })

                    # Spread to neighbors (from batch query results)
                    neighbors = neighbors_by_node.get(unit_id, [])
                    for neighbor in neighbors:
                        neighbor_id = str(neighbor["to_unit_id"])
                        if neighbor_id not in visited:
                            link_weight = neighbor["weight"]
                            new_activation = activation * link_weight * 0.8  # 0.8 = decay factor

                            if new_activation > 0.1:
                                queue.append(({
                                    "id": neighbor["to_unit_id"],
                                    "text": neighbor["text"],
                                    "context": neighbor.get("context", ""),
                                    "event_date": neighbor["event_date"],
                                    "access_count": neighbor["access_count"],
                                    "embedding": neighbor.get("embedding"),
                                }, new_activation, False))  # Not an entry point

                calculate_weight_time += time.time() - substep_start
                process_neighbors_time += time.time() - substep_start

                # Clear batch for next iteration
                nodes_to_process = []

            spreading_activation_time = time.time() - step_start
            num_batches = (len(visited) + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
            print(f"  [3] Spreading activation: {len(visited)} nodes visited in {spreading_activation_time:.3f}s")
            print(f"      [3.1] Update access counts: {update_access_time:.3f}s")
            print(f"      [3.2] Calculate weights: {calculate_weight_time:.3f}s")
            print(f"      [3.3] Query neighbors: {query_neighbors_time:.3f}s ({num_batches} batched queries)")
            print(f"      [3.4] Process neighbors: {process_neighbors_time:.3f}s")

            step_start = time.time()
            self.conn.commit()
            print(f"  [4] Commit: {time.time() - step_start:.3f}s")

            # Step 4: Sort by final weight and return top results
            step_start = time.time()
            results.sort(key=lambda x: x["weight"], reverse=True)
            top_results = results[:top_k]
            print(f"  [5] Sort and return top {top_k}: {time.time() - step_start:.3f}s")

            print(f"[SEARCH] Complete: {len(top_results)} results in {time.time() - search_start:.3f}s\n")
            return top_results

        except Exception as e:
            print(f"[SEARCH] ERROR after {time.time() - search_start:.3f}s: {str(e)}")
            self.conn.rollback()
            raise Exception(f"Failed to search memories: {str(e)}")
        finally:
            cursor.close()

    def delete_agent(self, agent_id: str) -> Dict[str, int]:
        """
        Delete all data for a specific agent (multi-tenant cleanup).

        This is much more efficient than dropping all tables and allows
        multiple agents to coexist in the same database.

        Deletes (with CASCADE):
        - All memory units for this agent
        - All entities for this agent
        - All associated links, unit-entity associations, and co-occurrences

        Args:
            agent_id: Agent ID to delete

        Returns:
            Dictionary with counts of deleted items
        """
        cursor = self.conn.cursor()

        try:
            # Count before deletion for reporting
            cursor.execute("SELECT COUNT(*) FROM memory_units WHERE agent_id = %s", (agent_id,))
            units_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM entities WHERE agent_id = %s", (agent_id,))
            entities_count = cursor.fetchone()[0]

            # Delete memory units (cascades to unit_entities, memory_links)
            cursor.execute("DELETE FROM memory_units WHERE agent_id = %s", (agent_id,))

            # Delete entities (cascades to unit_entities, entity_cooccurrences, memory_links with entity_id)
            cursor.execute("DELETE FROM entities WHERE agent_id = %s", (agent_id,))

            self.conn.commit()

            return {
                "memory_units_deleted": units_count,
                "entities_deleted": entities_count
            }

        except Exception as e:
            self.conn.rollback()
            raise Exception(f"Failed to delete agent data: {str(e)}")
        finally:
            cursor.close()

    def get_memory_graph_data(self, agent_id: str = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Get memory graph data for visualization.

        Args:
            agent_id: Optional agent ID (if None, returns all data)

        Returns:
            Tuple of (units, links) for visualization
        """
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)

        try:
            # Get all units (optionally filtered by agent)
            if agent_id:
                cursor.execute(
                    "SELECT id, text, context, event_date, access_count FROM memory_units WHERE agent_id = %s",
                    (agent_id,)
                )
            else:
                cursor.execute(
                    "SELECT id, text, context, event_date, access_count FROM memory_units"
                )
            units = [dict(row) for row in cursor.fetchall()]

            # Get all links (optionally filtered by agent)
            if agent_id:
                cursor.execute(
                    """
                    SELECT ml.from_unit_id, ml.to_unit_id, ml.link_type, ml.weight
                    FROM memory_links ml
                    JOIN memory_units mu1 ON ml.from_unit_id = mu1.id
                    JOIN memory_units mu2 ON ml.to_unit_id = mu2.id
                    WHERE mu1.agent_id = %s
                    """,
                    (agent_id,)
                )
            else:
                cursor.execute(
                    "SELECT from_unit_id, to_unit_id, link_type, weight FROM memory_links"
                )
            links = [dict(row) for row in cursor.fetchall()]

            return units, links

        except Exception as e:
            raise Exception(f"Failed to get memory graph data: {str(e)}")
        finally:
            cursor.close()

    def _extract_entities_batch_optimized(
        self,
        cursor,
        agent_id: str,
        unit_ids: List[str],
        sentences: List[str],
        context: str,
        fact_dates: List,
    ) -> List[tuple]:
        """
        Extract entities from ALL sentences in one batch (MUCH faster than sequential).

        Uses spaCy's batch processing to extract entities from all texts at once,
        then resolves and links them in bulk.

        Returns list of tuples for batch insertion: (from_unit_id, to_unit_id, link_type, weight, entity_id)
        """
        from .entity_resolver import extract_entities_batch

        try:
            # Step 1: Extract entities from ALL sentences in one batch (fast!)
            substep_start = time.time()
            all_entities = extract_entities_batch(sentences)
            total_entities = sum(len(ents) for ents in all_entities)
            print(f"  [6.1] spaCy NER (batch): {total_entities} entities from {len(sentences)} sentences in {time.time() - substep_start:.3f}s")

            # Step 2: Resolve entities in BATCH (much faster!)
            substep_start = time.time()
            step_6_2_start = time.time()

            # [6.2.1] Prepare all entities for batch resolution
            substep_6_2_1_start = time.time()
            all_entities_flat = []
            entity_to_unit = []  # Maps flat index to (unit_id, local_index)

            for unit_id, entities, fact_date in zip(unit_ids, all_entities, fact_dates):
                if not entities:
                    continue

                for local_idx, entity in enumerate(entities):
                    all_entities_flat.append({
                        'text': entity['text'],
                        'type': entity['type'],
                        'nearby_entities': entities,
                    })
                    entity_to_unit.append((unit_id, local_idx, fact_date))
            print(f"    [6.2.1] Prepare entities: {len(all_entities_flat)} entities in {time.time() - substep_6_2_1_start:.3f}s")

            # Resolve ALL entities in one batch call
            if all_entities_flat:
                # [6.2.2] Batch resolve entities
                substep_6_2_2_start = time.time()
                # Group by date for batch resolution (most will have same date)
                entities_by_date = {}
                for idx, (unit_id, local_idx, fact_date) in enumerate(entity_to_unit):
                    date_key = fact_date
                    if date_key not in entities_by_date:
                        entities_by_date[date_key] = []
                    entities_by_date[date_key].append((idx, all_entities_flat[idx]))

                # Resolve each date group in batch
                resolved_entity_ids = [None] * len(all_entities_flat)
                for fact_date, entities_group in entities_by_date.items():
                    indices = [idx for idx, _ in entities_group]
                    entities_data = [entity_data for _, entity_data in entities_group]

                    batch_resolved = self.entity_resolver.resolve_entities_batch(
                        agent_id=agent_id,
                        entities_data=entities_data,
                        context=context,
                        unit_event_date=fact_date
                    )

                    for idx, entity_id in zip(indices, batch_resolved):
                        resolved_entity_ids[idx] = entity_id
                print(f"    [6.2.2] Resolve entities: {len(all_entities_flat)} entities in {time.time() - substep_6_2_2_start:.3f}s")

                # [6.2.3] Create unit-entity links in BATCH
                substep_6_2_3_start = time.time()
                # Map resolved entities back to units and collect all (unit, entity) pairs
                unit_to_entity_ids = {}
                unit_entity_pairs = []
                for idx, (unit_id, local_idx, fact_date) in enumerate(entity_to_unit):
                    if unit_id not in unit_to_entity_ids:
                        unit_to_entity_ids[unit_id] = []

                    entity_id = resolved_entity_ids[idx]
                    unit_to_entity_ids[unit_id].append(entity_id)
                    unit_entity_pairs.append((unit_id, entity_id))

                # Batch insert all unit-entity links (MUCH faster!)
                self.entity_resolver.link_units_to_entities_batch(unit_entity_pairs)
                print(f"    [6.2.3] Create unit-entity links (batched): {len(unit_entity_pairs)} links in {time.time() - substep_6_2_3_start:.3f}s")

                print(f"  [6.2] Entity resolution (batched): {len(all_entities_flat)} entities resolved in {time.time() - step_6_2_start:.3f}s")
            else:
                unit_to_entity_ids = {}
                print(f"  [6.2] Entity resolution (batched): 0 entities in {time.time() - step_6_2_start:.3f}s")

            # Step 3: Create entity links between units that share entities
            substep_start = time.time()
            # Collect all unique entity IDs
            all_entity_ids = set()
            for entity_ids in unit_to_entity_ids.values():
                all_entity_ids.update(entity_ids)

            # For each entity, find all units that reference it (one query per entity)
            entity_to_units = {}
            for entity_id in all_entity_ids:
                cursor.execute(
                    """
                    SELECT unit_id
                    FROM unit_entities
                    WHERE entity_id = %s
                    """,
                    (entity_id,)
                )
                entity_to_units[entity_id] = [row[0] for row in cursor.fetchall()]

            # Create bidirectional links between units that share entities
            links = []
            for entity_id, units_with_entity in entity_to_units.items():
                # For each pair of units with this entity, create bidirectional links
                for i, unit_id_1 in enumerate(units_with_entity):
                    for unit_id_2 in units_with_entity[i+1:]:
                        # Bidirectional links
                        links.append((unit_id_1, unit_id_2, 'entity', 1.0, entity_id))
                        links.append((unit_id_2, unit_id_1, 'entity', 1.0, entity_id))

            print(f"  [6.3] Entity link creation: {len(links)} links for {len(all_entity_ids)} unique entities in {time.time() - substep_start:.3f}s")

            return links

        except Exception as e:
            print(f"ERROR: Failed to extract entities in batch: {str(e)}")
            import traceback
            traceback.print_exc()
            # Re-raise to trigger rollback at put_async level
            raise

    def _create_temporal_links_batch(
        self,
        cursor,
        agent_id: str,
        unit_ids: List[str],
        event_date: datetime,
        time_window_hours: int = 24,
    ):
        """
        Create temporal links for multiple units in one batch query.

        Uses a single query to find all relevant temporal connections.
        """
        if not unit_ids:
            return

        try:
            from psycopg2.extras import execute_values

            # Get ALL recent units within time window (single query)
            # Cast string IDs to UUIDs for comparison
            cursor.execute(
                """
                SELECT id, event_date
                FROM memory_units
                WHERE agent_id = %s
                  AND id::text != ALL(%s)
                  AND event_date >= %s
                ORDER BY event_date DESC
                """,
                (agent_id, unit_ids, event_date - timedelta(hours=time_window_hours))
            )

            recent_units = cursor.fetchall()

            # Create links from each new unit to all recent units
            links = []
            for unit_id in unit_ids:
                for recent_id, recent_event_date in recent_units:
                    # Calculate temporal proximity weight
                    time_diff_hours = abs((event_date - recent_event_date).total_seconds() / 3600)
                    weight = max(0.3, 1.0 - (time_diff_hours / time_window_hours))
                    links.append((unit_id, recent_id, 'temporal', weight, None))

            if links:
                execute_values(
                    cursor,
                    """
                    INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, entity_id)
                    VALUES %s
                    ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                    """,
                    links
                )

        except Exception as e:
            print(f"ERROR: Failed to create temporal links: {str(e)}")
            import traceback
            traceback.print_exc()
            # Re-raise to trigger rollback at put_async level
            raise

    def _create_temporal_links_batch_per_fact(
        self,
        cursor,
        agent_id: str,
        unit_ids: List[str],
        time_window_hours: int = 24,
    ):
        """
        Create temporal links for multiple units, each with their own event_date.

        Queries the event_date for each unit from the database and creates temporal
        links based on individual dates (supports per-fact dating).
        """
        if not unit_ids:
            return

        try:
            from psycopg2.extras import execute_values

            # Get the event_date for each new unit
            cursor.execute(
                """
                SELECT id, event_date
                FROM memory_units
                WHERE id::text = ANY(%s)
                """,
                (unit_ids,)
            )
            new_units = {str(row[0]): row[1] for row in cursor.fetchall()}

            # Create links based on each unit's individual event_date
            links = []
            for unit_id, unit_event_date in new_units.items():
                # Find units within the time window of THIS specific unit
                cursor.execute(
                    """
                    SELECT id, event_date
                    FROM memory_units
                    WHERE agent_id = %s
                      AND id != %s
                      AND event_date BETWEEN %s AND %s
                    ORDER BY event_date DESC
                    LIMIT 10
                    """,
                    (
                        agent_id,
                        unit_id,
                        unit_event_date - timedelta(hours=time_window_hours),
                        unit_event_date + timedelta(hours=time_window_hours)
                    )
                )

                recent_units = cursor.fetchall()

                for recent_id, recent_event_date in recent_units:
                    # Calculate temporal proximity weight
                    time_diff_hours = abs((unit_event_date - recent_event_date).total_seconds() / 3600)
                    weight = max(0.3, 1.0 - (time_diff_hours / time_window_hours))
                    links.append((unit_id, recent_id, 'temporal', weight, None))

            if links:
                execute_values(
                    cursor,
                    """
                    INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, entity_id)
                    VALUES %s
                    ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                    """,
                    links
                )

        except Exception as e:
            print(f"ERROR: Failed to create temporal links: {str(e)}")
            import traceback
            traceback.print_exc()
            # Re-raise to trigger rollback at put_async level
            raise

    def _create_semantic_links_batch(
        self,
        cursor,
        agent_id: str,
        unit_ids: List[str],
        embeddings: List[List[float]],
        top_k: int = 5,
        threshold: float = 0.7,
    ):
        """
        Create semantic links for multiple units efficiently.

        For each unit, finds similar units and creates links.
        """
        if not unit_ids or not embeddings:
            return

        try:
            from psycopg2.extras import execute_values

            all_links = []

            for unit_id, embedding in zip(unit_ids, embeddings):
                # Find similar units using vector similarity
                cursor.execute(
                    """
                    SELECT id, 1 - (embedding <=> %s::vector) AS similarity
                    FROM memory_units
                    WHERE agent_id = %s
                      AND id != %s
                      AND embedding IS NOT NULL
                      AND (1 - (embedding <=> %s::vector)) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding, agent_id, unit_id, embedding, threshold, embedding, top_k)
                )

                similar_units = cursor.fetchall()

                for similar_id, similarity in similar_units:
                    all_links.append((unit_id, similar_id, 'semantic', float(similarity), None))

            if all_links:
                execute_values(
                    cursor,
                    """
                    INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, entity_id)
                    VALUES %s
                    ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                    """,
                    all_links
                )

        except Exception as e:
            print(f"ERROR: Failed to create semantic links: {str(e)}")
            import traceback
            traceback.print_exc()
            # Re-raise to trigger rollback at put_async level
            raise

    def _insert_entity_links_batch(self, cursor, links: List[tuple]):
        """Insert all entity links in a single batch."""
        if not links:
            return

        try:
            from psycopg2.extras import execute_values
            execute_values(
                cursor,
                """
                INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, entity_id)
                VALUES %s
                ON CONFLICT (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING
                """,
                links
            )
        except Exception as e:
            print(f"Warning: Failed to insert entity links: {str(e)}")
