from urllib.parse import urlparse, urlunparse

import requests


def get_qlever_prefixes(endpoint: str) -> dict[str, str]:
    parse = urlparse(endpoint)
    parse.encode()
    split = parse.path.split("/")
    assert len(split) >= 1, "Endpoint path must contain at least one segment"
    split.insert(len(split) - 1, "prefixes")
    path = "/".join(split)
    parse._replace(path=path)
    prefix_url = urlunparse(parse)

    response = requests.get(prefix_url)
    response.raise_for_status()
    prefixes = {}
    for line in response.text.splitlines():
        line = line.strip()
        if not line:
            continue
        assert line.startswith("PREFIX "), "Each line must start with 'PREFIX '"
        _, rest = line.split(" ", 1)
        prefix, uri = rest.split(":", 1)
        prefixes[prefix.strip()] = uri.strip()[:-1]
    return prefixes
