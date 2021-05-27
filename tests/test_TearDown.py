import shutil
import pytest


@pytest.fixture(scope="session")
def Testteardown():
    yield
    shutil.rmtree('tmp')
