from typing import Any

from pydantic import BaseModel

from grasp.functions import execute_sparql, find_manager, update_known_from_selections
from grasp.manager import KgManager
from grasp.sparql.item import get_sparql_items, selections_from_items


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
) -> tuple[str, str, str]:
    manager, _ = find_manager(managers, kg)

    try:
        result, sparql = execute_sparql(
            managers,
            kg,
            sparql,
            max_rows,
            max_columns,
            known,
            return_sparql=True,
        )
        sparql = manager.prettify(sparql)
    except Exception as e:
        result = f"Failed to execute SPARQL query:\n{e}"

    try:
        _, items = get_sparql_items(sparql, manager)
        selections = selections_from_items(items)
        if known is not None:
            update_known_from_selections(known, selections, manager)
        selections = manager.format_selections(selections)
    except Exception as e:
        selections = f"Failed to determine used entities and properties:\n{e}"

    return sparql, selections, result


def format_sparql_result(
    sparql: str,
    selections: str,
    result: str,
    kg: str,
) -> str:
    fmt = f"SPARQL query over {kg}:\n{sparql.strip()}"

    if selections:
        fmt += f"\n\n{selections}"

    fmt += f"\n\nExecution result:\n{result.strip()}"
    return fmt
