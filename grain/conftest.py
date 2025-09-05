import sys
import pytest

RUN_IN_PYTEST = "pytest" in sys.argv[0]


#@pytest.fixture(scope="module", autouse=True)
#def set_flags(request):
#    breakpoint()
