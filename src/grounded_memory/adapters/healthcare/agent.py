"""
Healthcare Memory Agent

An agent specifically designed to handle clinical data using the
HealthcareDatabaseExtractor.
"""

import asyncio
from typing import Any

from grounded_memory.adapters.healthcare.extractor import HealthcareDatabaseExtractor
from grounded_memory.adapters.healthcare.lifecycle import (
    apply_medication_lifecycle_after_grounding,
)


class HealthcareMemoryAgent:
    """
    Healthcare domain-specific memory agent.
    """

    def __init__(
        self,
        memory_store: Any,
        grounding_operator: Any,
        llm_config: Any,
        domain_profile: str = "healthcare",
    ):
        self.memory_store = memory_store
        self.grounding_operator = grounding_operator
        self.llm_config = llm_config
        self.domain_profile = domain_profile
        self.extractor = HealthcareDatabaseExtractor(store=self.memory_store)

    async def process_interaction(
        self,
        raw_text: str,
        user_id: str | None = None,
        session_id: str | None = None,
        actor: str = "user",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:

        # 1. Extract Clinical Data
        pipeline_result = await self.extractor.process_interaction(
            raw_text=raw_text,
            user_id=user_id,
            session_id=session_id,
            actor=actor,
            metadata=metadata,
        )

        # 2. Ground candidates
        grounding_results = []
        for candidate in pipeline_result.candidate_facts:
            # We enforce sync grounding for now as GroundingOperator is synchronous
            result = self.grounding_operator.ground(candidate)
            apply_medication_lifecycle_after_grounding(
                store=self.memory_store,
                result=result,
            )
            grounding_results.append(result)

        # 3. Create a result object to match generic interface expected by the demo
        class HealthcareAgentResult:
            def __init__(self, inter, extracts, groundings):
                self.interaction = inter
                self.extracted_items = extracts
                self.grounding_results = groundings
                self.approved_facts = [g.validated_fact for g in groundings if g.is_success]
                self.rejected_facts = [g.rejection_record for g in groundings if not g.is_success]
                self.warnings = [
                    warning
                    for grounding in groundings
                    for warning in getattr(
                        getattr(grounding, "validation_result", None), "warnings", []
                    )
                ]
                self.dispositions = []

        return HealthcareAgentResult(
            pipeline_result.interaction, pipeline_result.extraction_result, grounding_results
        )

    def process(self, input_text: str, source: str = "user", **kwargs: Any) -> Any:
        """Backward-compatible sync wrapper for demo scripts.

        The healthcare adapter's native API is async, but several demos still
        call the legacy synchronous `process(...)` entrypoint used by the
        generic agent. This wrapper preserves that contract.
        """
        metadata = kwargs.pop("metadata", None)
        user_id = kwargs.pop("user_id", None)
        session_id = kwargs.pop("session_id", None)

        extra_metadata = {k: v for k, v in kwargs.items() if v is not None}
        if metadata is None:
            metadata = extra_metadata
        elif isinstance(metadata, dict):
            metadata = {**metadata, **extra_metadata}

        actor = source.strip().lower() if isinstance(source, str) else "user"

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.process_interaction(
                    raw_text=input_text,
                    user_id=user_id,
                    session_id=session_id,
                    actor=actor,
                    metadata=metadata,
                )
            )

        raise RuntimeError(
            "HealthcareMemoryAgent.process() cannot be used while an event loop is running. "
            "Await process_interaction(...) instead."
        )
