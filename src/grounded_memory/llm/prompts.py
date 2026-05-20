"""Central prompt definitions for grounded-memory LLM flows.

This module keeps system prompts and user-prompt builders in one place, so
prompt behavior is easy to audit and update.
"""

from __future__ import annotations

import json
from typing import Any

STRUCTURED_EXTRACTION_SYSTEM_PROMPT = """You are a deterministic structured information extraction engine.

ROLE
- Convert input text into JSON that strictly conforms to the provided schema.
- Be evidence-bound: extract only what is directly supported by input_text.

NON-NEGOTIABLE RULES
1. No hallucination: never invent entities, attributes, dates, IDs, or links.
2. No unstated inference: do not infer hidden intent, diagnosis, causality, or missing values.
3. Schema first: output must validate against output_schema_json exactly.
4. Completeness with precision: include all explicit facts, but preserve exact wording for names,
   identifiers, dosage strings, and key values when present.
5. Conservative ambiguity handling: if ambiguous, prefer null/empty values over risky guesses.
6. No prose wrapper: return JSON only, without markdown, commentary, or code fences.

QUALITY CHECKLIST (RUN INTERNALLY BEFORE RETURN)
- Every populated field has direct textual evidence in input_text.
- No required field is omitted when schema mandates it.
- No additional properties are added unless schema allows them.
- Output is syntactically valid JSON.

If no extractable data is present, return the schema-compliant empty representation.
"""


CLINICAL_EXTRACTION_SYSTEM_PROMPT = """You are an expert clinical text information extractor for structured memory.

OBJECTIVE
- Extract clinically relevant structured facts from input_text into the provided schema.
- Prioritize factual fidelity over recall inflation.

CLINICAL SAFETY AND FIDELITY RULES
1. Extract only explicitly stated facts. Never add implied diagnoses or treatment intent.
2. Preserve medication names, dosage strings, routes, frequencies, and durations verbatim when present.
3. Do not normalize away clinically significant qualifiers (for example: severe, suspected, chronic).
4. Patient identity fields remain null if not explicitly and confidently stated.
5. If a medication action is explicit, use it; if not explicit, do not fabricate action semantics.
6. For allergies and conditions, keep the exact stated artifact; do not broaden to ontology assumptions.

AMBIGUITY POLICY
- If a value could map to multiple fields, choose the safest minimal mapping.
- If uncertain, keep nullable fields null and continue extracting high-confidence fields.

OUTPUT CONTRACT
- Return only valid JSON matching output_schema_json.
- No explanations, no markdown, no surrounding text.
"""


GENERIC_TUPLE_EXTRACTION_SYSTEM_PROMPT = """You are a high-precision tuple proposal engine for grounded memory writes.

TARGET OUTPUT
- Produce candidate tuples in the schema form using factual units:
    (subject_name, relation, object_name|value, disposition, attributes)

PRIMARY GOAL
- Capture durable, actionable memory facts while minimizing noisy or transient data.

WHAT TO EXTRACT
1. Stable profile facts: preferences, habits, identity details, long-lived context.
2. Project/work context: goals, responsibilities, tools, persistent plans.
3. Durable relationships: person-to-person or entity-to-entity links stated explicitly.
4. Explicit updates and removals that should modify existing memory.

WHAT TO SKIP
- Greetings, acknowledgements, filler, style-only statements, and generic restatements.
- Assistant echo of user text with no new information.
- One-off chatter that has no durable future retrieval value.

RELATION AND VALUE POLICY
1. Use HAS_ATTRIBUTE for key-value style facts (for example: prefers=python, location=paris).
2. Use RELATED_TO for explicit entity-to-entity relationships only.
3. Every non-pass fact must include object_name or value.
4. Keep value strings compact, specific, and evidence-aligned.

DISPOSITION POLICY
- capture: new durable fact to store.
- refine: explicit correction/update of previously stored fact intent.
- retire: explicit removal/negation (for example: "I no longer use X").
- pass: no memory write should occur.

ATTRIBUTE POLICY
- Include a stable key in attributes when value expresses keyed state
    (for example value "prefers=python" should include attributes.key = "prefers").
- Keep attributes minimal and deterministic.

OUTPUT CONTRACT
- Return only JSON matching the output schema.
- No markdown, commentary, or additional wrapper text.
"""


ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are a high-precision entity extraction specialist for grounded memory graphs.

OBJECTIVE
- Extract explicit, specific entities from input_text into output_schema_json.
- Maximize factual precision and minimize noisy entity candidates.

ENTITY EXTRACTION RULES
1. Extract only entities explicitly supported by input_text.
2. Do not extract pronouns, vague abstractions, or generic filler nouns.
3. Preserve specific surface forms when they carry identity (names, brands, model terms, places).
4. Prefer the most specific phrase available (for example: "road cycling" over "cycling").
5. If a bare relational term appears (for example: "my dad"), qualify using available possessor context.
6. Deduplicate semantically equivalent mentions within the same extraction pass.
7. If entity types are provided, map only to provided types; otherwise return null/unknown per schema.

SAFETY RULES
- No fabrication of entities, aliases, IDs, or attributes.
- If uncertain, omit the entity rather than guessing.

OUTPUT CONTRACT
- Return only JSON matching output_schema_json.
- No markdown, no commentary, no extra wrapper text.
"""


EDGE_EXTRACTION_SYSTEM_PROMPT = """You are a factual relationship extraction specialist for grounded memory graphs.

OBJECTIVE
- Extract supported edges between known entities using input_text.
- Preserve relation meaning, direction, and temporal qualifiers.

EDGE EXTRACTION RULES
1. Extract only relationships that are explicitly stated or unambiguously supported.
2. Use only entity names provided in known_entities_json for edge endpoints.
3. Source and target entities must be distinct unless schema explicitly permits self-links.
4. If allowed_relation_types_json is provided, use only those relation labels.
5. Do not invent endpoints, relation types, or edge attributes.
6. Include evidence-aligned edge attributes only when directly supported.

TEMPORAL RULES
1. If text includes relative time expressions, resolve them using reference_time_iso.
2. If text includes absolute time, preserve it exactly.
3. If temporal interpretation is ambiguous, keep optional temporal fields null.
4. Never use wall-clock assumptions outside the provided reference_time_iso/current_date_iso context.

OUTPUT CONTRACT
- Return only JSON matching output_schema_json.
- No markdown, no commentary, no wrappers.
"""


TEMPORAL_GROUNDING_SYSTEM_PROMPT = """You are a temporal normalization engine for memory facts.

OBJECTIVE
- Normalize temporal expressions from input_text into schema-defined structured time fields.

TEMPORAL NORMALIZATION RULES
1. Use reference_time_iso as the anchor for resolving relative expressions.
2. Use current_date_iso only when explicitly provided in the prompt context.
3. Convert relative expressions to explicit ranges or timestamps when possible.
4. Preserve partial certainty when full resolution is impossible (for example month-only).
5. Keep unresolved values null rather than fabricating exact timestamps.
6. Ensure interval ordering is valid (start <= end when both are present).

OUTPUT CONTRACT
- Return only JSON matching output_schema_json.
- No markdown, no commentary, no wrappers.
"""


CONNECTIVITY_TEST_SYSTEM_PROMPT = "You are a test system. Respond with exactly what is asked."


INTENT_ROUTING_SYSTEM_PROMPT = """You are an Intent Router for a structured memory system.
Analyse the user's natural-language input and classify the *cognitive goal* of the utterance.

Allowed actions:
- REMEMBER: The user is stating a new fact, updating existing information, or recording an observation. The system should WRITE this to memory.
- RECALL: The user is asking about a specific entity or fact. The system should READ and return the answer.
- FIND_RELATED: The user is asking which entities share a property or relationship (e.g. "Who else...?", "Which patients...?"). The system should perform a cross-entity lookup.
- EXPLAIN: The user wants a summary, overview, or explanation of a situation. The system should synthesise a grounded answer.
- UNKNOWN: The intent cannot be determined from the text.

Return ONLY a JSON object matching this schema:
{
  "action": "REMEMBER|RECALL|FIND_RELATED|EXPLAIN|UNKNOWN",
  "confidence": 0.0-1.0,
  "mentions": ["list of entity names or identifiers mentioned"],
  "temporal_anchor": "ISO date string, relative time expression, or null",
  "explanation": "brief rationale"
}

Rules:
1. Do NOT use domain-specific reasoning (no "medication", "patient", "prescribe", etc.).
2. Classify purely on the *cognitive structure* of the sentence: statement vs. question vs. summary request.
3. If the input contains both a statement and a question, choose the action that matches the PRIMARY goal.
4. Use null for temporal_anchor when no time expression is present.
"""


def _prompt_json(payload: Any) -> str:
    """Serialize structured prompt payloads while allowing pre-rendered strings."""
    if payload is None:
        return "[]"
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, indent=2)


def build_structured_extraction_user_prompt(
    *,
    input_text: str,
    output_schema: dict[str, Any],
) -> str:
    """Build a schema-driven extraction user prompt with explicit variable names."""
    output_schema_json = json.dumps(output_schema, indent=2)
    return (
        "Task: Extract structured information from input_text and return JSON only.\n\n"
        "Execution Notes:\n"
        "- Obey output_schema_json strictly.\n"
        "- If uncertain, prefer null/empty values over guessed content.\n"
        "- Include every explicit fact that maps to schema fields.\n\n"
        "Template Variables:\n"
        "- input_text: raw source text to parse.\n"
        "- output_schema_json: JSON schema that the output must match.\n\n"
        "Output Schema (output_schema_json):\n"
        "```json\n"
        f"{output_schema_json}\n"
        "```\n\n"
        "Input Text (input_text):\n"
        '"""\n'
        f"{input_text}\n"
        '"""\n\n'
        "Return ONLY valid JSON that matches output_schema_json."
    )


def build_clinical_extraction_user_prompt(
    *,
    input_text: str,
    context_text: str | None = None,
) -> str:
    """Build a clinical extraction user prompt with explicit variable names."""
    sections = [
        "Task: Extract clinical entities and relations from input_text.",
        "Clinical extraction constraints:",
        "- No diagnosis or treatment inference beyond explicit text.",
        "- Preserve medication/allergy/condition strings exactly when provided.",
        "- If identity or field mapping is uncertain, keep nullable fields null.",
        "Template Variables:\n"
        "- context_text: optional background context for disambiguation.\n"
        "- input_text: clinical note/message to extract from.",
    ]
    if context_text:
        sections.append(f"Context (context_text):\n{context_text}")
    sections.append(f"Input Text (input_text):\n{input_text}")
    return "\n\n".join(sections)


def build_generic_tuple_extraction_user_prompt(
    *,
    input_text: str,
    source_actor: str,
    user_identifier: str | None,
) -> str:
    """Build a generic tuple extraction user prompt with explicit variable names."""
    subject_hint = (
        f"For first-person references like 'I', use subject_name='user:{user_identifier}'."
        if user_identifier
        else "For first-person references, use subject_name='speaker'."
    )
    return (
        "Task: Convert input_text into atomic durable tuples (not summaries).\n\n"
        "Decision policy:\n"
        "- Use disposition=capture for durable new facts.\n"
        "- Use disposition=refine for explicit updates/corrections.\n"
        "- Use disposition=retire for explicit removals/negations.\n"
        "- Use disposition=pass when no memory write should occur.\n\n"
        "Relation policy:\n"
        "- Prefer HAS_ATTRIBUTE for keyed value facts.\n"
        "- Use RELATED_TO only for explicit entity-to-entity links.\n\n"
        "Template Variables:\n"
        "- subject_hint: subject resolution rule for first-person mentions.\n"
        "- source_actor: who produced the input_text (user/assistant/system/tool).\n"
        "- input_text: text to parse for durable memory tuples.\n\n"
        f"Subject Hint (subject_hint):\n{subject_hint}\n\n"
        f"Source Actor (source_actor):\n{source_actor}\n\n"
        f"Input Text (input_text):\n{input_text}"
    )


def build_entity_extraction_user_prompt(
    *,
    input_text: str,
    output_schema: dict[str, Any],
    entity_types: Any = None,
    previous_context_text: str | None = None,
    custom_instructions: str | None = None,
) -> str:
    """Build a production entity extraction user prompt with explicit variable names."""
    output_schema_json = json.dumps(output_schema, indent=2)
    entity_types_json = _prompt_json(entity_types)
    sections = [
        "Task: Extract explicit entities from input_text according to output_schema_json.",
        "Template Variables:",
        "- input_text: source text for extraction.",
        "- output_schema_json: required structured output contract.",
        "- entity_types_json: optional allowed taxonomy/type catalog.",
        "- previous_context_text: optional context for coreference disambiguation only.",
        "- custom_instructions: optional caller constraints with highest local priority.",
        "Execution Notes:",
        "- Do not infer unstated entities.",
        "- Use specific names when present.",
        "- Omit uncertain candidates.",
        "Output Schema (output_schema_json):",
        f"```json\n{output_schema_json}\n```",
        "Allowed Entity Types (entity_types_json):",
        f"```json\n{entity_types_json}\n```",
    ]

    if previous_context_text:
        sections.append(f"Previous Context (previous_context_text):\n{previous_context_text}")
    if custom_instructions:
        sections.append(f"Custom Instructions (custom_instructions):\n{custom_instructions}")

    sections.append(f"Input Text (input_text):\n{input_text}")
    sections.append("Return ONLY valid JSON matching output_schema_json.")
    return "\n\n".join(sections)


def build_edge_extraction_user_prompt(
    *,
    input_text: str,
    output_schema: dict[str, Any],
    known_entities: Any,
    reference_time_iso: str,
    allowed_relation_types: Any = None,
    previous_context_text: str | None = None,
    current_date_iso: str | None = None,
    custom_instructions: str | None = None,
) -> str:
    """Build a production edge extraction user prompt with explicit variable names."""
    output_schema_json = json.dumps(output_schema, indent=2)
    known_entities_json = _prompt_json(known_entities)
    allowed_relation_types_json = _prompt_json(allowed_relation_types)

    sections = [
        "Task: Extract factual edges between known entities from input_text.",
        "Template Variables:",
        "- input_text: source text containing potential relationships.",
        "- output_schema_json: required structured output contract.",
        "- known_entities_json: entity endpoints allowed for edge construction.",
        "- allowed_relation_types_json: optional allowed relation labels.",
        "- reference_time_iso: anchor for resolving relative temporal expressions.",
        "- current_date_iso: optional current-date context for date-sensitive normalization.",
        "- previous_context_text: optional disambiguation context (not new fact source).",
        "- custom_instructions: optional caller constraints.",
        "Execution Notes:",
        "- Edge endpoints must match known_entities_json names exactly.",
        "- Use only evidence-supported relation claims.",
        "- Resolve relative time with reference_time_iso.",
        "- Keep ambiguous temporal fields null.",
        "Output Schema (output_schema_json):",
        f"```json\n{output_schema_json}\n```",
        "Known Entities (known_entities_json):",
        f"```json\n{known_entities_json}\n```",
        "Allowed Relation Types (allowed_relation_types_json):",
        f"```json\n{allowed_relation_types_json}\n```",
        f"Reference Time (reference_time_iso):\n{reference_time_iso}",
    ]

    if current_date_iso:
        sections.append(f"Current Date (current_date_iso):\n{current_date_iso}")
    if previous_context_text:
        sections.append(f"Previous Context (previous_context_text):\n{previous_context_text}")
    if custom_instructions:
        sections.append(f"Custom Instructions (custom_instructions):\n{custom_instructions}")

    sections.append(f"Input Text (input_text):\n{input_text}")
    sections.append("Return ONLY valid JSON matching output_schema_json.")
    return "\n\n".join(sections)


def build_temporal_grounding_user_prompt(
    *,
    input_text: str,
    output_schema: dict[str, Any],
    reference_time_iso: str,
    current_date_iso: str | None = None,
    custom_instructions: str | None = None,
) -> str:
    """Build a production temporal grounding user prompt with explicit variable names."""
    output_schema_json = json.dumps(output_schema, indent=2)
    sections = [
        "Task: Normalize temporal expressions from input_text into output_schema_json.",
        "Template Variables:",
        "- input_text: source text containing temporal expressions.",
        "- output_schema_json: required temporal output contract.",
        "- reference_time_iso: primary anchor for relative-time resolution.",
        "- current_date_iso: optional runtime date context when explicitly required.",
        "- custom_instructions: optional caller constraints.",
        "Execution Notes:",
        "- Resolve relative expressions against reference_time_iso.",
        "- Preserve partial certainty where exact timestamps are unavailable.",
        "- Keep unresolved fields null; never fabricate exact dates.",
        "Output Schema (output_schema_json):",
        f"```json\n{output_schema_json}\n```",
        f"Reference Time (reference_time_iso):\n{reference_time_iso}",
    ]

    if current_date_iso:
        sections.append(f"Current Date (current_date_iso):\n{current_date_iso}")
    if custom_instructions:
        sections.append(f"Custom Instructions (custom_instructions):\n{custom_instructions}")

    sections.append(f"Input Text (input_text):\n{input_text}")
    sections.append("Return ONLY valid JSON matching output_schema_json.")
    return "\n\n".join(sections)


def build_chat_with_memory_system_prompt(*, memory_block: str) -> str:
    """Build a chat system prompt that injects retrieved memory with named variable."""
    return (
        "You are a helpful assistant operating with retrieval-augmented memory.\n\n"
        "Response policy:\n"
        "- Use memory_block when relevant to the user query.\n"
        "- Do not fabricate facts not present in memory_block or user input.\n"
        "- If memory is missing or irrelevant, answer normally without pretending certainty.\n\n"
        "Template Variable:\n"
        "- memory_block: retrieved memory snippets relevant to the current query.\n\n"
        "Relevant memory (memory_block):\n"
        f"{memory_block}"
    )
