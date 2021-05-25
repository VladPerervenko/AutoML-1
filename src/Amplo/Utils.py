import re
import json


def boolean_input(question):
    x = input(question + ' [y / n]')
    if x.lower() == 'n' or x.lower() == 'no':
        return False
    elif x.lower() == 'y' or x.lower() == 'yes':
        return True
    else:
        print('Sorry, I did not understand. Please answer with "n" or "y"')
        return boolean_input(question)


def notification():
    # import requests
    # oAuthToken = 'xoxb-1822915844353-1822989373697-zsFM6CuC6VGTxBjHUcdZHSdJ'
    # url = 'https://slack.com/api/chat.postMessage'
    # data = {
    #     "token": oAuthToken,
    #     "channel": "automl",
    #     "text": message,
    #     "username": "AutoML",
    # }
    # requests.post(url, data=data)
    pass


def clean_keys(data):
    # Clean Keys
    new_keys = {}
    for key in data.keys():
        new_keys[key] = re.sub('[^a-zA-Z0-9 \n]', '_', key.lower()).replace('__', '_')
    data = data.rename(columns=new_keys)
    return data


def parse_json(json_string):
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
