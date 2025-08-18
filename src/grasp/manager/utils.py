import os
import time
from typing import Any, Type

from search_index import IndexData, PrefixIndex, SearchIndex, SimilarityIndex
from universal_ml_utils.io import load_json
from universal_ml_utils.logging import get_logger

from grasp.manager.mapping import Mapping
from grasp.utils import get_index_dir


def load_data_and_mapping(
    index_dir: str,
    mapping_cls: Type[Mapping] | None = None,
) -> tuple[IndexData, Mapping]:
    try:
        data = IndexData.load(
            os.path.join(index_dir, "data.tsv"),
            os.path.join(index_dir, "offsets.bin"),
        )
    except Exception as e:
        raise ValueError(f"Failed to load index data from {index_dir}") from e

    if mapping_cls is None:
        mapping_cls = Mapping

    try:
        mapping = mapping_cls.load(
            data,
            os.path.join(index_dir, "mapping.bin"),
        )
    except Exception as e:
        raise ValueError(f"Failed to load mapping from {index_dir}") from e

    return data, mapping


def load_index_and_mapping(
    index_dir: str,
    index_type: str,
    mapping_cls: Type[Mapping] | None = None,
    **kwargs: Any,
) -> tuple[SearchIndex, Mapping]:
    logger = get_logger("KG INDEX LOADING")
    start = time.perf_counter()

    if index_type == "prefix":
        index_cls = PrefixIndex
    elif index_type == "similarity":
        index_cls = SimilarityIndex
    else:
        raise ValueError(f"Unknown index type {index_type}")

    data, mapping = load_data_and_mapping(index_dir, mapping_cls)

    try:
        index = index_cls.load(
            data,
            os.path.join(index_dir, index_type),
            **kwargs,
        )
    except Exception as e:
        raise ValueError(f"Failed to load {index_type} index from {index_dir}") from e

    end = time.perf_counter()

    logger.debug(f"Loading {index_type} index from {index_dir} took {end - start:.2f}s")

    return index, mapping


def load_entity_index_and_mapping(
    name: str,
    index_dir: str | None = None,
    index_type: str | None = None,
    **kwargs: Any,
) -> tuple[SearchIndex, Mapping]:
    if index_dir is None:
        default_dir = get_index_dir()
        index_dir = os.path.join(default_dir, name, "entities")

    return load_index_and_mapping(
        index_dir,
        # for entities use prefix index by default
        index_type or "prefix",
        **kwargs,
    )


def load_property_index_and_mapping(
    kg: str,
    index_dir: str | None = None,
    index_type: str | None = None,
    **kwargs: Any,
) -> tuple[SearchIndex, Mapping]:
    if index_dir is None:
        default_dir = get_index_dir()
        index_dir = os.path.join(default_dir, kg, "properties")

    mapping_cls = WikidataPropertyMapping if kg == "wikidata" else None

    return load_index_and_mapping(
        index_dir,
        # for properties use similarity index by default
        index_type or "similarity",
        mapping_cls,
        **kwargs,
    )


def load_example_index(dir: str, **kwargs: Any) -> SimilarityIndex:
    data = IndexData.load(
        os.path.join(dir, "data.tsv"),
        os.path.join(dir, "offsets.bin"),
    )

    return SimilarityIndex.load(data, os.path.join(dir, "index"), **kwargs)


def load_kg_prefixes(kg: str) -> dict[str, str]:
    index_dir = get_index_dir()
    prefix_file = os.path.join(index_dir, kg, "prefixes.json")
    if not os.path.exists(prefix_file):
        return {}

    return load_json(prefix_file)


def load_kg_notes(kg: str, notes_file: str | None = None) -> list[str]:
    if notes_file is None:
        index_dir = get_index_dir()
        notes_file = os.path.join(index_dir, kg, "notes.json")

    if not os.path.exists(notes_file):
        return []

    return load_json(notes_file)  # type: ignore


def load_general_notes(notes_file: str | None = None) -> list[str]:
    if notes_file is None:
        index_dir = get_index_dir()
        notes_file = os.path.join(index_dir, "notes.json")

    if not os.path.exists(notes_file):
        return []

    return load_json(notes_file)  # type: ignore


def load_kg_indices(
    kg: str,
    entities_dir: str | None = None,
    entities_type: str | None = None,
    entities_kwargs: dict[str, Any] | None = None,
    properties_dir: str | None = None,
    properties_type: str | None = None,
    properties_kwargs: dict[str, Any] | None = None,
) -> tuple[SearchIndex, SearchIndex, Mapping, Mapping]:
    if entities_type != "similarity" and entities_kwargs:
        # entities kwargs only used for similarity index
        entities_kwargs.clear()

    ent_index, ent_mapping = load_entity_index_and_mapping(
        kg,
        entities_dir,
        entities_type,
        **(entities_kwargs or {}),
    )

    if properties_type == "prefix" and properties_kwargs:
        # properties kwargs only used for prefix index
        properties_kwargs.clear()

    # try to share embedding model between entities and properties
    # if entities also use a similarity index
    if entities_type == "similarity" and (
        not properties_kwargs or properties_kwargs.get("model") is None
    ):
        properties_kwargs = properties_kwargs or {}
        properties_kwargs["model"] = ent_index.model

    prop_index, prop_mapping = load_property_index_and_mapping(
        kg,
        properties_dir,
        properties_type,
        **(properties_kwargs or {}),
    )

    return ent_index, prop_index, ent_mapping, prop_mapping


def get_common_sparql_prefixes() -> dict[str, str]:
    return {
        "rdf": "<http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "<http://www.w3.org/2000/01/rdf-schema#",
        "owl": "<http://www.w3.org/2002/07/owl#",
        "xsd": "<http://www.w3.org/2001/XMLSchema#",
        "foaf": "<http://xmlns.com/foaf/0.1/",
        "skos": "<http://www.w3.org/2004/02/skos/core#",
        "dct": "<http://purl.org/dc/terms/",
        "dc": "<http://purl.org/dc/elements/1.1/",
        "prov": "<http://www.w3.org/ns/prov#",
        "schema": "<http://schema.org/",
        "geo": "<http://www.opengis.net/ont/geosparql#",
        "geosparql": "<http://www.opengis.net/ont/geosparql#",
        "gn": "<http://www.geonames.org/ontology#",
        "bd": "<http://www.bigdata.com/rdf#",
        "hint": "<http://www.bigdata.com/queryHints#",
        "wikibase": "<http://wikiba.se/ontology#",
        "qb": "<http://purl.org/linked-data/cube#",
        "void": "<http://rdfs.org/ns/void#",
    }


def get_index_desc(index: SearchIndex) -> str:
    if not is_sim_index(index):
        index_type = "Prefix-keyword index"
        dist_info = "number of exact and prefix keyword matches"

    else:
        index_type = "Similarity index"
        dist_info = "vector embedding distance"

    return f"{index_type} ranking by {dist_info}"


def is_sim_index(index: SearchIndex) -> bool:
    return index.get_type() == "similarity"


def clip(s: str, max_len: int = 64) -> str:
    if len(s) <= max_len + 3:  # 3 for "..."
        return s

    # clip string to max_len  + 3 by stripping out middle part
    half = max_len // 2
    first = s[:half]
    last = s[-half:]
    return first + "..." + last
