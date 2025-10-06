from typing import Any

from pydantic import BaseModel

from grasp.functions import (
    ExecutionResult,
    execute_sparql,
    find_manager,
    update_known_from_selections,
)
from grasp.manager import KgManager
from grasp.sparql.item import get_sparql_items, selections_from_items
from grasp.sparql.types import Selection


class Sample(BaseModel):
    id: str | None = None

    def input(self) -> Any:
        raise NotImplementedError

    def inputs(self) -> list[str]:
        raise NotImplementedError


def prepare_sparql_result(
    sparql: str,
    kg: str,
    managers: list[KgManager],
    max_rows: int,
    max_columns: int,
    known: set[str] | None = None,
) -> tuple[ExecutionResult, list[Selection]]:
    manager, _ = find_manager(managers, kg)

    result = execute_sparql(
        managers,
        kg,
        sparql,
        max_rows,
        max_columns,
        known,
    )

    try:
        _, items = get_sparql_items(sparql, manager)
        selections = selections_from_items(items)
        if known is not None:
            update_known_from_selections(known, selections, manager)
    except Exception:
        selections = []

    return result, selections


def format_sparql_result(
    manager: KgManager,
    result: ExecutionResult,
    selections: list[Selection],
) -> str:
    fmt = f"SPARQL query over {manager.kg}:\n{result.sparql}"

    if selections:
        fmt += f"\n\n{manager.format_selections(selections)}"

    fmt += f"\n\nExecution result:\n{result.formatted}"
    return fmt
