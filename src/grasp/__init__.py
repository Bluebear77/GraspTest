from importlib import metadata

try:
    __version__ = metadata.version("grasp")
except metadata.PackageNotFoundError:
    __version__ = "unknown"


# patch stuff here that should affect the behavior of the
# whole package

# Patch for WQSP and CWQ evaluation, because QLever
# supports xsd:date but not xsd:dateTime
# from functools import wraps
# from grasp.sparql import utils
#
# original_execute = utils.execute
#
# original_fix_prefixes = utils.fix_prefixes
#
# print("Patching SPARQL execute to replace xsd:dateTime with xsd:date")
#
#
# @wraps(original_execute)
# def patched_execute(sparql: str, *args, **kwargs):
#     sparql = sparql.replace("xsd:dateTime", "xsd:date")
#     sparql = sparql.replace("xsd:datetime", "xsd:date")
#     return original_execute(sparql, *args, **kwargs)
#
#
# utils.execute = patched_execute
#
# print("Patching SPARQL fix_prefixes to replace OR with ||")
#
#
# @wraps(original_fix_prefixes)
# def patched_fix_prefixes(sparql: str, *args, **kwargs) -> str:
#     sparql = sparql.replace(" OR ", " || ")
#     return original_fix_prefixes(sparql, *args, **kwargs)
#
#
# utils.fix_prefixes = patched_fix_prefixes


# Patch for endpoints requiring Basic Auth
# import os
# from functools import wraps
#
# from grasp.sparql import utils
#
# original_execute = utils.execute
#
# if os.path.exists("auth.txt"):
#     print("Found auth.txt, will use it for Basic Auth")
#     with open("auth.txt", "r") as f:
#         lines = f.read().splitlines()
#         USER = lines[0]
#         PASSWD = lines[1]
#
#     print(f"Patching SPARQL execute to add Basic Auth headers with {USER}/{PASSWD}")
#
#     @wraps(original_execute)
#     def patched_execute(*args, **kwargs):
#         kwargs["auth"] = (USER, PASSWD)
#         return original_execute(*args, **kwargs)
#
#     utils.execute = patched_execute
