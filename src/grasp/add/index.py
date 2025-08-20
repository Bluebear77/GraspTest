import os
import time
from logging import Logger

from search_index import PrefixIndex, SimilarityIndex
from universal_ml_utils.logging import get_logger

from grasp.configs import KgConfig
from grasp.manager.utils import load_data_and_mapping
from grasp.utils import get_index_dir


def build_index(
    index_dir: str,
    index_type: str,
    logger: Logger,
    overwrite: bool = False,
    sim_precision: str | None = None,
    sim_batch_size: int = 256,
    sim_embedding_dim: int | None = None,
) -> None:
    data, _ = load_data_and_mapping(index_dir)

    out_dir = os.path.join(index_dir, index_type)
    if os.path.exists(out_dir) and not overwrite:
        logger.info(
            f"Index of type {index_type} already exists at {out_dir}. Skipping build."
        )
        return

    os.makedirs(out_dir, exist_ok=True)
    start = time.perf_counter()
    logger.info(f"Building {index_type} index at {out_dir}")

    if index_type == "prefix":
        PrefixIndex.build(data, out_dir)
    elif index_type == "similarity":
        SimilarityIndex.build(
            data,
            out_dir,
            batch_size=sim_batch_size,
            embedding_dim=sim_embedding_dim,
            precision=sim_precision,
            show_progress=True,
        )
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    end = time.perf_counter()
    logger.info(f"Index build took {end - start:.2f} seconds")


def build_indices(
    knowledge_graph: KgConfig,
    overwrite: bool = False,
    log_level: str | int | None = None,
    sim_precision: str | None = None,
    sim_batch_size: int = 256,
    sim_embedding_dim: int | None = None,
) -> None:
    logger = get_logger("GRASP INDEX", log_level)

    index_dir = get_index_dir(knowledge_graph.kg)

    # entities
    entities_dir = os.path.join(index_dir, "entities")
    entities_type = knowledge_graph.entities_type or "prefix"
    build_index(
        entities_dir,
        entities_type,
        logger,
        overwrite,
        sim_precision,
        sim_batch_size,
        sim_embedding_dim,
    )

    # properties
    properties_dir = os.path.join(index_dir, "properties")
    properties_type = knowledge_graph.properties_type or "similarity"
    build_index(
        properties_dir,
        properties_type,
        logger,
        overwrite,
        sim_precision,
        sim_batch_size,
        sim_embedding_dim,
    )
