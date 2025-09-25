from pydantic import BaseModel

from grasp.configs import Config, NotesConfig
from grasp.functions import TaskFunctions
from grasp.manager import KgManager
from grasp.tasks.exploration.functions import call_function as call_note_function
from grasp.tasks.exploration.functions import note_functions
from grasp.utils import format_enumerate, format_list


class ExplorationState(BaseModel):
    notes: list[str] = []
    kg_notes: dict[str, list[str]] = {}


def rules() -> list[str]:
    return [
        "Avoid notes about entity or property identifiers just for the sake of not \
having to look them up again.",
        "As you hit the limits on the number of notes and their length, \
gradually generalize your notes, discard unnecessary details, and move \
notes that can be useful across knowledge graphs to the general section.",
    ]


def system_information(config: Config) -> str:
    assert isinstance(config, NotesConfig)
    return f"""\
You are a note-taking assistant. Your task is to \
explore knowledge graphs and to take notes about them using the \
provided functions. Stop the exploration and note-taking process \
by calling the stop function once you are done.

Your notes should help a knowledge graph agent to better understand and \
navigate the knowledge graphs. You can take notes specific to \
a certain knowledge graph, as well as general notes that might be \
useful across knowledge graphs. For exploration, think about what domains \
the knowledge graphs might cover and what types of questions a user might want to answer \
with them.

You are only allowed {config.max_notes} notes at max per knowledge graph and for the \
general notes, such that you are forced to prioritize and to keep them as widely \
applicable as possible. Notes are limited to {config.max_note_length} characters to \
ensure they are concise and to the point.

Examples of potentially useful types of notes include:
- overall structure and schema of the knowledge graphs
- peculiarities of the knowledge graphs
- strategies when encountering certain types of errors
- tips for when and how to use certain functions"""


def input(state: ExplorationState) -> str:
    kg_specific_notes = format_list(
        f"{kg}:\n{format_enumerate(kg_specific_notes, indent=2)}"
        for kg, kg_specific_notes in sorted(state.kg_notes.items())
    )
    return f"""\
Explore the available knowledge graphs. Add to, delete from, or update the following \
notes along the way.

Knowledge graph specific notes:
{kg_specific_notes}

General notes across knowledge graphs:
{format_enumerate(state.notes)}"""


def output(state: ExplorationState) -> dict:
    kg_specific_notes = format_list(
        f"{kg}:\n{format_enumerate(kg_specific_notes, indent=2)}"
        for kg, kg_specific_notes in sorted(state.kg_notes.items())
    )
    formatted = f"""\
Exploration completed.

Knowledge graph specific notes:
{kg_specific_notes}

General notes across knowledge graphs:
{format_enumerate(state.notes)}"""

    return {
        "type": "output",
        "notes": state.notes,
        "kg_notes": state.kg_notes,
        "formatted": formatted,
    }


def call_function(
    config: Config,
    managers: list[KgManager],
    fn_name: str,
    fn_args: dict,
    known: set[str],
    state: ExplorationState | None = None,
    example_indices: dict | None = None,
) -> str:
    assert isinstance(config, NotesConfig)
    assert state is not None, "State must be provided for exploration task"
    return call_note_function(
        state.kg_notes,
        state.notes,
        fn_name,
        fn_args,
        config.max_notes,
        config.max_note_length,
    )


def functions(managers: list[KgManager]) -> TaskFunctions:
    return note_functions(managers), call_function
