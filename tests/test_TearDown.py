import os
import shutil
import pytest


@pytest.fixture(scope="session", autouse=True)
def teardown():
    yield
    if os.path.exists('tmp'):
        shutil.rmtree('tmp')
    if os.path.exists('AutoClass'):
        shutil.rmtree('AutoClass')
    if os.path.exists('AutoReg'):
        shutil.rmtree('AutoReg')
