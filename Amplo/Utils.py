import re
import json
import pandas as pd
from typing import Union


def boolean_input(question: str) -> bool:
    x = input(question + ' [y / n]')
    if x.lower() == 'n' or x.lower() == 'no':
        return False
    elif x.lower() == 'y' or x.lower() == 'yes':
        return True
    else:
        print('Sorry, I did not understand. Please answer with "n" or "y"')
        return boolean_input(question)


def clean_keys(data: pd.DataFrame) -> pd.DataFrame:
    # Clean Keys
    new_keys = {}
    for key in data.keys():
        if isinstance(key, int):
            new_keys[key] = 'feature_{}'.format(key)
        else:
            new_keys[key] = re.sub('[^a-z0-9\n]', '_', str(key).lower()).replace('__', '_')
    data = data.rename(columns=new_keys)
    return data


def parse_json(json_string: Union[str, dict]) -> Union[str, dict]:
    if isinstance(json_string, dict):
        return json_string
    else:
        try:
            return json.loads(json_string
                              .replace("'", '"')
                              .replace("True", "true")
                              .replace("False", "false")
                              .replace("nan", "NaN")
                              .replace("None", "null"))
        except json.decoder.JSONDecodeError:
            print('[AutoML] Cannot validate, impassable JSON.')
            print(json_string)
            return json_string


def check_dataframe_quality(data: pd.DataFrame) -> bool:
    assert not data.isna().any().any()
    assert not data.isnull().any().any()
    assert not (data.dtypes == object).any().any()
    assert not (data.dtypes == str).any().any()
    assert data.max().max() < 1e38 and data.min().min() > -1e38
    return True
