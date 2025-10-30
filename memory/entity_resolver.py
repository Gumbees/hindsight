"""
Entity extraction and resolution for memory system.

Uses spaCy for entity extraction and implements resolution logic
to disambiguate entities across memory units.
"""
import spacy
from typing import List, Dict, Optional, Set
from difflib import SequenceMatcher


# Load spaCy model (singleton)
_nlp = None


def get_nlp():
    """Get or load spaCy model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_entities(text: str) -> List[Dict[str, any]]:
    """
    Extract entities from text using spaCy.

    Args:
        text: Input text

    Returns:
        List of entities with text, type, and span info
    """
    nlp = get_nlp()
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        # Filter to important entity types
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT']:
            entities.append({
                'text': ent.text,
                'type': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
            })

    return entities


def extract_entities_batch(texts: List[str]) -> List[List[Dict[str, any]]]:
    """
    Extract entities from multiple texts in batch (MUCH faster than sequential).

    Uses spaCy's nlp.pipe() for efficient batch processing.

    Args:
        texts: List of input texts

    Returns:
        List of entity lists, one per input text
    """
    if not texts:
        return []

    nlp = get_nlp()

    # Process all texts in batch using nlp.pipe (significantly faster!)
    docs = list(nlp.pipe(texts, batch_size=50))

    all_entities = []
    for doc in docs:
        entities = []
        for ent in doc.ents:
            # Filter to important entity types
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT']:
                entities.append({
                    'text': ent.text,
                    'type': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                })
        all_entities.append(entities)

    return all_entities


class EntityResolver:
    """
    Resolves entities to canonical IDs with disambiguation.
    """

    def __init__(self, db_conn):
        """
        Initialize entity resolver.

        Args:
            db_conn: psycopg2 database connection
        """
        self.conn = db_conn

    def resolve_entities_batch(
        self,
        agent_id: str,
        entities_data: List[Dict],
        context: str,
        unit_event_date,
    ) -> List[str]:
        """
        Resolve multiple entities in batch (MUCH faster than sequential).

        Groups entities by type, queries candidates in bulk, and resolves
        all entities with minimal DB queries.

        Args:
            agent_id: Agent ID
            entities_data: List of dicts with 'text', 'type', 'nearby_entities'
            context: Context where entities appear
            unit_event_date: When this unit was created

        Returns:
            List of entity IDs in same order as input
        """
        if not entities_data:
            return []

        cursor = self.conn.cursor()

        try:
            import time
            start = time.time()

            # Group entities by type for efficient querying
            entities_by_type = {}
            for idx, entity_data in enumerate(entities_data):
                entity_type = entity_data['type']
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append((idx, entity_data))

            # Query ALL candidates for each type in batch
            all_candidates = {}  # Maps (entity_type, entity_text) -> list of candidates
            for entity_type, entities_list in entities_by_type.items():
                # Extract unique entity texts for this type
                entity_texts = list(set(e[1]['text'] for e in entities_list))

                # Query candidates for all texts at once
                from psycopg2.extras import execute_values
                cursor.execute(
                    """
                    SELECT canonical_name, id, metadata, last_seen, mention_count
                    FROM entities
                    WHERE agent_id = %s AND entity_type = %s
                    """,
                    (agent_id, entity_type)
                )
                type_candidates = cursor.fetchall()

                # Filter candidates in memory (faster than complex SQL for small datasets)
                for entity_text in entity_texts:
                    matching = []
                    entity_text_lower = entity_text.lower()
                    for canonical_name, ent_id, metadata, last_seen, mention_count in type_candidates:
                        canonical_lower = canonical_name.lower()
                        # Same matching logic as before
                        if (entity_text_lower == canonical_lower or
                            entity_text_lower in canonical_lower or
                            canonical_lower in entity_text_lower):
                            matching.append((ent_id, canonical_name, metadata, last_seen, mention_count))
                    all_candidates[(entity_type, entity_text)] = matching

            # Resolve each entity using pre-fetched candidates
            entity_ids = [None] * len(entities_data)
            entities_to_update = []  # (entity_id, unit_event_date)
            entities_to_create = []  # (idx, entity_data)

            for idx, entity_data in enumerate(entities_data):
                entity_text = entity_data['text']
                entity_type = entity_data['type']
                nearby_entities = entity_data.get('nearby_entities', [])

                candidates = all_candidates.get((entity_type, entity_text), [])

                if not candidates:
                    # Will create new entity
                    entities_to_create.append((idx, entity_data))
                    continue

                # Score candidates (same logic as before but with pre-fetched data)
                best_candidate = None
                best_score = 0.0
                best_name_similarity = 0.0

                nearby_entity_set = {e['text'].lower() for e in nearby_entities if e['text'] != entity_text}

                for candidate_id, canonical_name, metadata, last_seen, mention_count in candidates:
                    score = 0.0

                    # Name similarity
                    name_similarity = SequenceMatcher(
                        None,
                        entity_text.lower(),
                        canonical_name.lower()
                    ).ratio()
                    score += name_similarity * 0.5

                    # Temporal proximity
                    if last_seen:
                        days_diff = abs((unit_event_date - last_seen).total_seconds() / 86400)
                        if days_diff < 7:
                            temporal_score = max(0, 1.0 - (days_diff / 7))
                            score += temporal_score * 0.2

                    if score > best_score:
                        best_score = score
                        best_candidate = candidate_id
                        best_name_similarity = name_similarity

                # Apply threshold
                threshold = 0.4 if entity_type == 'PERSON' and best_name_similarity >= 0.95 else 0.6

                if best_score > threshold:
                    entity_ids[idx] = best_candidate
                    entities_to_update.append((best_candidate, unit_event_date))
                else:
                    entities_to_create.append((idx, entity_data))

            # Batch update existing entities
            if entities_to_update:
                from psycopg2.extras import execute_values
                execute_values(
                    cursor,
                    """
                    UPDATE entities SET
                        mention_count = mention_count + 1,
                        last_seen = data.last_seen
                    FROM (VALUES %s) AS data(id, last_seen)
                    WHERE entities.id = data.id::uuid
                    """,
                    entities_to_update
                )

            # Batch create new entities
            if entities_to_create:
                for idx, entity_data in entities_to_create:
                    entity_id = self._create_entity(
                        cursor, agent_id, entity_data['text'],
                        entity_data['type'], unit_event_date
                    )
                    entity_ids[idx] = entity_id

            return entity_ids

        finally:
            cursor.close()

    def resolve_entity(
        self,
        agent_id: str,
        entity_text: str,
        entity_type: str,
        context: str,
        nearby_entities: List[Dict],
        unit_event_date,
    ) -> str:
        """
        Resolve an entity to a canonical entity ID.

        Args:
            agent_id: Agent ID (entities are scoped to agents)
            entity_text: Entity text ("Alice", "Google", etc.)
            entity_type: Entity type (PERSON, ORG, etc.)
            context: Context where entity appears
            nearby_entities: Other entities in the same unit
            unit_event_date: When this unit was created

        Returns:
            Entity ID (creates new entity if needed)
        """
        cursor = self.conn.cursor()

        try:
            # Find candidate entities with same type and similar name
            cursor.execute(
                """
                SELECT id, canonical_name, metadata, last_seen
                FROM entities
                WHERE agent_id = %s
                  AND entity_type = %s
                  AND (
                    canonical_name ILIKE %s
                    OR canonical_name ILIKE %s
                    OR %s ILIKE canonical_name || '%%'
                  )
                ORDER BY mention_count DESC
                """,
                (agent_id, entity_type, entity_text, f"%{entity_text}%", entity_text)
            )

            candidates = cursor.fetchall()

            if not candidates:
                # New entity - create it
                return self._create_entity(
                    cursor, agent_id, entity_text, entity_type, unit_event_date
                )

            # Score candidates based on:
            # 1. Name similarity
            # 2. Context overlap (TODO: could use embeddings)
            # 3. Co-occurring entities
            # 4. Temporal proximity

            best_candidate = None
            best_score = 0.0
            best_name_similarity = 0.0

            nearby_entity_set = {e['text'].lower() for e in nearby_entities if e['text'] != entity_text}

            for candidate_id, canonical_name, metadata, last_seen in candidates:
                score = 0.0

                # 1. Name similarity (0-1)
                name_similarity = SequenceMatcher(
                    None,
                    entity_text.lower(),
                    canonical_name.lower()
                ).ratio()
                score += name_similarity * 0.5

                # 2. Co-occurring entities (0-0.5)
                # Get entities that co-occurred with this candidate before
                # Use the materialized co-occurrence cache for fast lookup
                cursor.execute(
                    """
                    SELECT e.canonical_name, ec.cooccurrence_count
                    FROM entity_cooccurrences ec
                    JOIN entities e ON (
                        CASE
                            WHEN ec.entity_id_1 = %s THEN ec.entity_id_2
                            WHEN ec.entity_id_2 = %s THEN ec.entity_id_1
                        END = e.id
                    )
                    WHERE ec.entity_id_1 = %s OR ec.entity_id_2 = %s
                    """,
                    (candidate_id, candidate_id, candidate_id, candidate_id)
                )
                co_entities = {row[0].lower() for row in cursor.fetchall()}

                # Check overlap with nearby entities
                overlap = len(nearby_entity_set & co_entities)
                if nearby_entity_set:
                    co_entity_score = overlap / len(nearby_entity_set)
                    score += co_entity_score * 0.3

                # 3. Temporal proximity (0-0.2)
                if last_seen:
                    days_diff = abs((unit_event_date - last_seen).total_seconds() / 86400)
                    if days_diff < 7:  # Within a week
                        temporal_score = max(0, 1.0 - (days_diff / 7))
                        score += temporal_score * 0.2

                if score > best_score:
                    best_score = score
                    best_candidate = candidate_id
                    best_name_similarity = name_similarity

            # Threshold for considering it the same entity
            # For PERSON entities with exact name match, use lower threshold
            threshold = 0.4 if entity_type == 'PERSON' and best_name_similarity >= 0.95 else 0.6

            if best_score > threshold:
                # Update entity
                cursor.execute(
                    """
                    UPDATE entities
                    SET mention_count = mention_count + 1,
                        last_seen = %s
                    WHERE id = %s
                    """,
                    (unit_event_date, best_candidate)
                )
                return best_candidate
            else:
                # Not confident - create new entity
                return self._create_entity(
                    cursor, agent_id, entity_text, entity_type, unit_event_date
                )

        finally:
            cursor.close()

    def _create_entity(
        self,
        cursor,
        agent_id: str,
        entity_text: str,
        entity_type: str,
        event_date,
    ) -> str:
        """
        Create a new entity.

        Args:
            cursor: Database cursor
            agent_id: Agent ID
            entity_text: Entity text
            entity_type: Entity type
            event_date: When first seen

        Returns:
            Entity ID
        """
        cursor.execute(
            """
            INSERT INTO entities (agent_id, canonical_name, entity_type, first_seen, last_seen, mention_count)
            VALUES (%s, %s, %s, %s, %s, 1)
            RETURNING id
            """,
            (agent_id, entity_text, entity_type, event_date, event_date)
        )
        entity_id = cursor.fetchone()[0]
        return entity_id

    def link_unit_to_entity(self, unit_id: str, entity_id: str):
        """
        Link a memory unit to an entity.
        Also updates co-occurrence cache with other entities in the same unit.

        Args:
            unit_id: Memory unit ID
            entity_id: Entity ID
        """
        cursor = self.conn.cursor()
        try:
            # Insert unit-entity link
            cursor.execute(
                """
                INSERT INTO unit_entities (unit_id, entity_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
                """,
                (unit_id, entity_id)
            )

            # Update co-occurrence cache: find other entities in this unit
            cursor.execute(
                """
                SELECT entity_id
                FROM unit_entities
                WHERE unit_id = %s AND entity_id != %s
                """,
                (unit_id, entity_id)
            )

            other_entities = [row[0] for row in cursor.fetchall()]

            # Update co-occurrences for each pair
            for other_entity_id in other_entities:
                self._update_cooccurrence(cursor, entity_id, other_entity_id)

        finally:
            cursor.close()

    def _update_cooccurrence(self, cursor, entity_id_1: str, entity_id_2: str):
        """
        Update the co-occurrence cache for two entities.

        Uses CHECK constraint ordering (entity_id_1 < entity_id_2) to avoid duplicates.

        Args:
            cursor: Database cursor
            entity_id_1: First entity ID
            entity_id_2: Second entity ID
        """
        # Ensure consistent ordering (smaller UUID first)
        if entity_id_1 > entity_id_2:
            entity_id_1, entity_id_2 = entity_id_2, entity_id_1

        cursor.execute(
            """
            INSERT INTO entity_cooccurrences (entity_id_1, entity_id_2, cooccurrence_count, last_cooccurred)
            VALUES (%s, %s, 1, NOW())
            ON CONFLICT (entity_id_1, entity_id_2)
            DO UPDATE SET
                cooccurrence_count = entity_cooccurrences.cooccurrence_count + 1,
                last_cooccurred = NOW()
            """,
            (entity_id_1, entity_id_2)
        )

    def link_units_to_entities_batch(self, unit_entity_pairs: List[tuple[str, str]]):
        """
        Link multiple memory units to entities in batch (MUCH faster than sequential).

        Also updates co-occurrence cache for entities that appear in the same unit.

        Args:
            unit_entity_pairs: List of (unit_id, entity_id) tuples
        """
        if not unit_entity_pairs:
            return

        cursor = self.conn.cursor()
        try:
            # Batch insert all unit-entity links
            from psycopg2.extras import execute_values
            execute_values(
                cursor,
                """
                INSERT INTO unit_entities (unit_id, entity_id)
                VALUES %s
                ON CONFLICT DO NOTHING
                """,
                unit_entity_pairs
            )

            # Build map of unit -> entities for co-occurrence calculation
            # Use sets to avoid duplicate entities in the same unit
            unit_to_entities = {}
            for unit_id, entity_id in unit_entity_pairs:
                if unit_id not in unit_to_entities:
                    unit_to_entities[unit_id] = set()
                unit_to_entities[unit_id].add(entity_id)

            # Update co-occurrences for all pairs in each unit
            cooccurrence_pairs = set()  # Use set to avoid duplicates
            for unit_id, entity_ids in unit_to_entities.items():
                entity_list = list(entity_ids)  # Convert set to list for iteration
                # For each pair of entities in this unit, create co-occurrence
                for i, entity_id_1 in enumerate(entity_list):
                    for entity_id_2 in entity_list[i+1:]:
                        # Skip if same entity (shouldn't happen with set, but be safe)
                        if entity_id_1 == entity_id_2:
                            continue
                        # Ensure consistent ordering (entity_id_1 < entity_id_2)
                        if entity_id_1 > entity_id_2:
                            entity_id_1, entity_id_2 = entity_id_2, entity_id_1
                        cooccurrence_pairs.add((entity_id_1, entity_id_2))

            # Batch update co-occurrences
            if cooccurrence_pairs:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                execute_values(
                    cursor,
                    """
                    INSERT INTO entity_cooccurrences (entity_id_1, entity_id_2, cooccurrence_count, last_cooccurred)
                    VALUES %s
                    ON CONFLICT (entity_id_1, entity_id_2)
                    DO UPDATE SET
                        cooccurrence_count = entity_cooccurrences.cooccurrence_count + 1,
                        last_cooccurred = EXCLUDED.last_cooccurred
                    """,
                    [(e1, e2, 1, now) for e1, e2 in cooccurrence_pairs]
                )

        finally:
            cursor.close()

    def get_units_by_entity(self, entity_id: str, limit: int = 100) -> List[str]:
        """
        Get all units that mention an entity.

        Args:
            entity_id: Entity ID
            limit: Max results

        Returns:
            List of unit IDs
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                """
                SELECT unit_id
                FROM unit_entities
                WHERE entity_id = %s
                ORDER BY unit_id
                LIMIT %s
                """,
                (entity_id, limit)
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()

    def get_entity_by_text(
        self,
        agent_id: str,
        entity_text: str,
        entity_type: Optional[str] = None
    ) -> Optional[str]:
        """
        Find an entity by text (for query resolution).

        Args:
            agent_id: Agent ID
            entity_text: Entity text to search for
            entity_type: Optional entity type filter

        Returns:
            Entity ID if found, None otherwise
        """
        cursor = self.conn.cursor()
        try:
            if entity_type:
                cursor.execute(
                    """
                    SELECT id FROM entities
                    WHERE agent_id = %s
                      AND entity_type = %s
                      AND canonical_name ILIKE %s
                    ORDER BY mention_count DESC
                    LIMIT 1
                    """,
                    (agent_id, entity_type, entity_text)
                )
            else:
                cursor.execute(
                    """
                    SELECT id FROM entities
                    WHERE agent_id = %s
                      AND canonical_name ILIKE %s
                    ORDER BY mention_count DESC
                    LIMIT 1
                    """,
                    (agent_id, entity_text)
                )

            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            cursor.close()
