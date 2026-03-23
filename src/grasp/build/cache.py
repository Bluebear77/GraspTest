import dbm
import json
import os
from itertools import batched

from tqdm import trange
from universal_ml_utils.logging import setup_logging

from grasp.configs import KgConfig
from grasp.manager import load_kg_manager
from grasp.manager.utils import load_data
from grasp.utils import get_index_dir


def build_caches(
    kg: str,
    endpoint: str | None = None,
    limit: int | None = None,
    batch_size: int = 100,
    overwrite: bool = False,
    log_level: str | int | None = None,
) -> None:
    setup_logging(log_level)

    manager = load_kg_manager(
        KgConfig(kg=kg, endpoint=endpoint),
        skip_indices=True,
    )

    kg_index_dir = get_index_dir(kg)

    info_sparqls = {
        "entities": manager.entity_info_sparql,
        "properties": manager.property_info_sparql,
    }

    for typ, info_sparql in info_sparqls.items():
        sub_dir = os.path.join(kg_index_dir, typ)

        cache_path = os.path.join(sub_dir, "info.cache", "db")
        os.makedirs(cache_path, exist_ok=True)

        data = load_data(sub_dir)
        cache_limit = limit if limit is not None else len(data)

        dbm_mode = "n" if overwrite else "c"
        with dbm.open(cache_path, dbm_mode) as db:
            for ids in batched(
                trange(min(len(data), cache_limit), desc=f"Caching infos for {typ}"),
                batch_size,
            ):
                identifiers = []
                for id in ids:
                    identifier = data.identifier(id)
                    if identifier is None or identifier in db:
                        continue
                    identifiers.append(identifier)

                if not identifiers:
                    continue

                infos = manager.get_infos_for_identifiers(identifiers, info_sparql)
                for identifier, info in infos.items():
                    db[identifier] = json.dumps(info)
