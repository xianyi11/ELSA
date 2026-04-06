"""Small helpers (paths, legacy unpickling). Importing this package registers the ``partition`` shim."""
import sys

from . import partition_compat

sys.modules.setdefault("partition", partition_compat)
