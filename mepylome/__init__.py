# If C++ parser should be added, before installation do: export MEPYLOME_CPP=1
try:
    from _mepylome import IdatParser as _IdatParser
except ModuleNotFoundError:
    pass
from mepylome.dtypes import *

# Suppress pyranges warnings
# TODO
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
