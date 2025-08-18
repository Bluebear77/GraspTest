import json
import sys

from grasp.add.utils import qlever_prefixes_to_json

if __name__ == "__main__":
    prefixes = qlever_prefixes_to_json(sys.stdin)
    print(json.dumps(prefixes, indent=2))
