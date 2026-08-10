# Marks tests/ as a package.
#
# test_preprocess.py does `from tests.test_end_to_end import SCHEMA_SRC`. Under
# pytest's default prepend import mode, that resolves only if this file exists:
# without it pytest puts tests/ itself on sys.path and `tests` is not importable
# as a package. It passed locally on a stale __pycache__ and failed on the first
# clean checkout, which is exactly the failure mode CI is for.
