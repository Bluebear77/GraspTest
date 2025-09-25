from typing import Any

from pydantic import BaseModel, conlist


class KgConfig(BaseModel):
    kg: str
    endpoint: str | None = None
    entities_type: str | None = None
    properties_type: str | None = None
    notes_file: str | None = None
    example_index: str | None = None


class Config(BaseModel):
    model: str = "openai/gpt-5-mini"
    model_endpoint: str | None = None

    seed: int | None = None
    fn_set: str = "search_extended"
    notes_file: str | None = None

    knowledge_graphs: list[KgConfig] = [KgConfig(kg="wikidata")]

    # optional task specific parameters
    task_kwargs: dict[str, Any] = {}

    # kg function parameters
    search_top_k: int = 10
    # 10 total rows, 5 top and 5 bottom
    result_max_rows: int = 10
    # same for columns
    result_max_columns: int = 10
    # 10 total results, 10 top
    list_k: int = 10
    # force that all IRIs used in a SPARQL query
    # were previously seen
    know_before_use: bool = False

    # model inference parameters
    model_kwargs: dict[str, Any] = {}
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    # one of completions, responses, or None (for auto)
    api: str | None = None
    parallel_tool_calls: bool = False

    # completion parameters
    max_completion_tokens: int = 8192  # 8k, leaves enough space for reasoning models
    completion_timeout: float = 120.0
    max_steps: int = 100

    # example parameters
    num_examples: int = 3
    force_examples: str | None = None
    random_examples: bool = False

    # enable feedback loop
    feedback: bool = False
    max_feedbacks: int = 2


class NotesConfig(Config):
    # additional parameters specific to taking notes with GRASP
    max_notes: int = 16
    max_note_length: int = 512
    num_rounds: int = 5

    # adapt model can be different from the main model
    note_taking_model: str | None = None
    note_taking_model_endpoint: str | None = None
    note_taking_max_steps: int = 50

    # and have different decoding parameters
    note_taking_temperature: float | None = None
    note_taking_top_p: float | None = None
    note_taking_reasoning_effort: str | None = None
    note_taking_reasoning_summary: str | None = None
    note_taking_api: str | None = None


class NotesInput(BaseModel):
    kg: str
    file: str


class NotesFromInputsConfig(NotesConfig):
    # input files with input-output pairs
    inputs: conlist(NotesInput, min_length=1)  # type: ignore
    samples_per_round: int = 3
    samples_per_file: int | None = None
