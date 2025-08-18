import csv
import os
from copy import deepcopy
from logging import Logger
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import unquote_plus

import requests
from search_index import IndexData, Mapping
from universal_ml_utils.io import dump_lines
from universal_ml_utils.logging import get_logger

from grasp.add.utils import get_qlever_prefixes
from grasp.configs import KgConfig
from grasp.manager.utils import get_common_sparql_prefixes, load_kg_prefixes
from grasp.sparql.utils import (
    find_longest_prefix,
    get_endpoint,
    is_iri,
    load_entity_query,
    load_property_query,
)
from grasp.utils import get_index_dir


def download_data(
    data_file: str,
    endpoint: str,
    query: str,
    logger: Logger,
    prefixes: dict[str, str],
    headers: dict[str, str] | None = None,
    add_id_as_synonyms: bool = False,
    overwrite: bool = False,
) -> None:
    if os.path.exists(data_file) and not overwrite:
        logger.info(f"Data already exists at {data_file}, skipping download")
        return

    logger.info(
        f"Downloading data to {data_file} from {endpoint} "
        f"with headers {headers or {}} and query:\n{query}"
    )
    stream = stream_lines(endpoint, query, headers)
    dump_lines(prepare_lines(stream, prefixes, logger, add_id_as_synonyms), data_file)


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
        Mapping.build(data, mapping_file.as_posix(), 3)  # type: ignore
    else:
        logger.info(f"Mapping file already exists at {mapping_file}, skipping build")


def get_data(
    knowledge_graph: KgConfig,
    entity_query: str | None = None,
    property_query: str | None = None,
    headers: dict[str, str] | None = None,
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

    if not headers:
        headers = {}
    else:
        headers = deepcopy(headers)

    headers.update({"Accept": "text/csv", "Content-Type": "application/sparql-query"})

    prefixes = load_kg_prefixes(knowledge_graph.kg)
    if not prefixes:
        try:
            prefixes = get_qlever_prefixes(endpoint)
        except Exception as e:
            logger.warning(f"Failed to get QLever prefixes from {endpoint}: {e}")
            prefixes = get_common_sparql_prefixes()

    kg_dir = get_index_dir(knowledge_graph.kg)

    # entities
    ent_dir = os.path.join(kg_dir, "entities")
    os.makedirs(ent_dir, exist_ok=True)
    out_file = os.path.join(ent_dir, "data.tsv")
    download_data(
        out_file,
        endpoint,
        entity_query or load_entity_query(),
        logger,
        prefixes,
        headers,
        overwrite=overwrite,
    )
    build_data_and_mapping(out_file, logger, overwrite)

    # properties
    prop_dir = os.path.join(kg_dir, "properties")
    os.makedirs(prop_dir, exist_ok=True)
    out_file = os.path.join(prop_dir, "data.tsv")
    download_data(
        out_file,
        endpoint,
        property_query or load_property_query(),
        logger,
        prefixes,
        headers,
        add_id_as_synonyms=True,  # for properties we also want to search via id
        overwrite=overwrite,
    )
    build_data_and_mapping(out_file, logger, overwrite)


def stream_lines(
    url: str,
    data: Any,
    headers: dict[str, str] | None = None,
) -> Iterator[str]:
    try:
        response = requests.post(url, data=data, headers=headers, stream=True)
        response.raise_for_status()
        yield from response.iter_lines(decode_unicode=True)
    except Exception as e:
        raise ValueError(f"Failed to stream rows: {e}") from e


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


def clean(s: str) -> str:
    return " ".join(s.split())


def prepare_lines(
    data: Iterable[str],
    prefixes: dict[str, str],
    logger: Logger,
    add_id_as_synonym: bool = False,
) -> Iterator[str]:
    reader = csv.reader(data)
    num = 0

    # skip original header
    next(reader)

    # yield own header
    yield "\t".join(["label", "score", "synonyms", "id", "infos"])

    for row in reader:
        # remove \n and \t from each column
        row = [clean(col) for col in row]

        try:
            label, score, syns, id, infos = row
        except Exception:
            logger.warning(f"Got malformed row: {row}")
            continue

        # add brackets to id
        id = f"<{id}>"

        if not label:
            # label is empty, try to get it from synonyms or the object id
            if syns:
                # use the first synonym as label
                # keep rest of synonyms
                label, *rest = syns.split(";;;")
                syns = ";;;".join(rest)
            else:
                label = get_label_from_id(id, prefixes)

        elif add_id_as_synonym:
            # add id of item to synonyms
            object_name = get_object_name_from_id(id, prefixes)
            if object_name != label and syns:
                syns = f"{syns};;;{object_name}"
            elif object_name != label:
                syns = object_name

        # if args.osm_planet_entities:
        # for osm planet entities, score is a wikidata id
        # wid = f"<{score}>"
        # score = get_osm_planet_score_from_wikidata_id(wid)

        score = "0" if not score else score

        yield "\t".join([label, score, syns, id, infos])

        num += 1
        if num % 1_000_000 == 0:
            logger.info(f"Processed {num:,} items so far")
