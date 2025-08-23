import csv
import os
from logging import Logger
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote_plus

import requests
from search_index import IndexData, Mapping
from universal_ml_utils.io import dump_lines, dump_text
from universal_ml_utils.logging import get_logger

from grasp.configs import KgConfig
from grasp.manager.utils import get_common_sparql_prefixes, load_kg_prefixes
from grasp.sparql.utils import (
    find_longest_prefix,
    get_endpoint,
    is_iri,
    load_entity_index_sparql,
    load_property_index_sparql,
)
from grasp.utils import get_index_dir


def download_data(
    data_file: str,
    endpoint: str,
    sparql: str,
    logger: Logger,
    prefixes: dict[str, str],
    params: dict[str, str] | None = None,
    add_id_as_label: bool = False,
    overwrite: bool = False,
) -> None:
    if os.path.exists(data_file) and not overwrite:
        logger.info(f"Data already exists at {data_file}, skipping download")
        return

    logger.info(
        f"Downloading data to {data_file} from {endpoint} "
        f"with parameters {params or {}} and SPARQL:\n{sparql}"
    )

    stream = stream_csv(endpoint, sparql, params)
    dump_lines(prepare_csv(stream, prefixes, logger, add_id_as_label), data_file)


def build_data_and_mapping(
    data_file: str,
    logger: Logger,
    overwrite: bool = False,
) -> None:
    path = Path(data_file)
    offsets_file = Path.with_name(path, "offsets.bin")
    mapping_file = Path.with_name(path, "mapping.bin")
    if not offsets_file.exists() or overwrite:
        # build index data
        logger.info(f"Building offsets file at {offsets_file}")
        IndexData.build(data_file, offsets_file.as_posix())
    else:
        logger.info(f"Offsets file already exists at {offsets_file}, skipping build")

    data = IndexData.load(data_file, offsets_file.as_posix())

    if not mapping_file.exists() or overwrite:
        # build mapping
        logger.info(f"Building mapping file at {mapping_file}")
        Mapping.build(data, mapping_file.as_posix())  # type: ignore
    else:
        logger.info(f"Mapping file already exists at {mapping_file}, skipping build")


def get_data(
    knowledge_graph: KgConfig,
    entity_query: str | None = None,
    property_query: str | None = None,
    params: dict[str, str] | None = None,
    overwrite: bool = False,
    log_level: str | int | None = None,
) -> None:
    logger = get_logger("GRASP DATA", log_level)

    if knowledge_graph.endpoint is None:
        endpoint = get_endpoint(knowledge_graph.kg)
        logger.info(
            f"Using endpoint {endpoint} for {knowledge_graph.kg} because "
            "no endpoint is set in the config"
        )
    else:
        endpoint = knowledge_graph.endpoint

    prefixes = get_common_sparql_prefixes()
    prefixes.update(load_kg_prefixes(knowledge_graph.kg))

    kg_dir = get_index_dir(knowledge_graph.kg)

    # entities
    ent_dir = os.path.join(kg_dir, "entities")
    os.makedirs(ent_dir, exist_ok=True)
    out_file = os.path.join(ent_dir, "data.tsv")
    ent_sparql = entity_query or load_entity_index_sparql()
    download_data(
        out_file,
        endpoint,
        ent_sparql,
        logger,
        prefixes,
        params,
        overwrite=overwrite,
    )
    dump_text(ent_sparql, os.path.join(ent_dir, "index.sparql"))
    build_data_and_mapping(out_file, logger, overwrite)

    # properties
    prop_dir = os.path.join(kg_dir, "properties")
    os.makedirs(prop_dir, exist_ok=True)
    out_file = os.path.join(prop_dir, "data.tsv")
    prop_sparql = property_query or load_property_index_sparql()
    download_data(
        out_file,
        endpoint,
        prop_sparql,
        logger,
        prefixes,
        params,
        add_id_as_label=True,  # for properties we also want to search via id
        overwrite=overwrite,
    )
    dump_text(prop_sparql, os.path.join(prop_dir, "index.sparql"))
    build_data_and_mapping(out_file, logger, overwrite)


def stream_csv(
    endpoint: str,
    sparql: str,
    params: dict[str, str] | None = None,
) -> Iterator[list[str]]:
    try:
        headers = {
            "Accept": "text/csv",
            "Content-Type": "application/sparql-query",
            "User-Agent": "grasp-data-bot",
        }

        response = requests.post(
            endpoint,
            data=sparql,
            params=params,
            headers=headers,
            stream=True,
        )
        response.raise_for_status()

        lines = (line.decode("utf-8") for line in response.iter_lines())
        for row in csv.reader(lines):
            # pad to 3 columns
            while len(row) < 3:
                row.append("")
            yield row

    except Exception as e:
        raise ValueError(f"Failed to stream csv: {e}") from e


def split_iri(iri: str) -> tuple[str, str]:
    if not is_iri(iri):
        return "", iri

    # split iri into prefix and last part after final / or #
    last_hashtag = iri.rfind("#")
    last_slash = iri.rfind("/")
    last = max(last_hashtag, last_slash)
    if last == -1:
        return "", iri[1:-1]
    else:
        return iri[1:last], iri[last + 1 : -1]


def camel_case_split(s: str) -> str:
    # split camelCase into words
    # find uppercase letters
    words = []
    last = 0
    for i, c in enumerate(s):
        if c.isupper() and i > 0 and s[i - 1].islower():
            words.append(s[last:i])
            last = i

    if last < len(s):
        words.append(s[last:])

    return " ".join(words)


def get_object_name_from_id(obj_id: str, prefixes: dict[str, str]) -> str:
    pfx = find_longest_prefix(obj_id, prefixes)
    if pfx is None:
        # no known prefix, split after final / or # to get objet name
        _, obj_name = split_iri(obj_id)
    else:
        _, long = pfx
        obj_name = obj_id[len(long) : -1]

    # url decode the object name
    return unquote_plus(obj_name)


def get_label_from_id(obj_id: str, prefixes: dict[str, str]) -> str:
    obj_name = get_object_name_from_id(obj_id, prefixes)
    label = " ".join(camel_case_split(part) for part in split_at_punctuation(obj_name))
    return label.strip()


# we consider _, -, and . as url punctuation
PUNCTUATION = {"_", "-", "."}


def split_at_punctuation(s: str) -> Iterator[str]:
    start = 0
    for i, c in enumerate(s):
        if c not in PUNCTUATION:
            continue

        yield s[start:i]
        start = i + 1

    if start < len(s):
        yield s[start:]


def ordered_unique(lst: list[str]) -> list[str]:
    seen = set()
    unique = []
    for item in lst:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def prepare_csv(
    lines: Iterator[list[str]],
    prefixes: dict[str, str],
    logger: Logger,
    add_id_as_label: bool = False,
) -> Iterator[str]:
    num = 0

    # parser = load_iri_and_literal_parser()

    # skip original header
    next(lines)

    # yield own header
    yield "\t".join(["id", "labels"])

    for line in lines:
        assert len(line) == 3, f"Expected 3 columns, got {len(line)}: {line}"

        id, label, synonyms = line

        # wrap id with brackets
        id = f"<{id}>"

        # filter out empty label and synonyms
        labels = []
        if label:
            labels.append(label)
        for syn in synonyms.split(";;;"):
            if syn:
                labels.append(syn)

        if not labels:
            # label is empty, try to get it from the object id
            labels.append(get_label_from_id(id, prefixes))

        if add_id_as_label:
            # add id of item to labels
            object_name = get_object_name_from_id(id, prefixes)
            labels.append(object_name)

        # make sure no duplicates are in the labels
        labels = ordered_unique(labels)
        yield "\t".join([id] + labels)

        num += 1
        if num % 1_000_000 == 0:
            logger.info(f"Processed {num:,} items so far")
