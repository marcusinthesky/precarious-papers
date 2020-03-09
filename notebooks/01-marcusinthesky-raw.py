# %%
from typing import Dict
import requests


# %%
def map_jobs(jobs: Dict):
    """
    See: https://github.com/OpenFIGI/api-examples/blob/master/python/example-with-requests.py

        curl 'https://api.openfigi.com/v2/search' \
        --request POST \
        --header 'Content-Type: application/json' \
        --header f'X-OPENFIGI-APIKEY: {context.config_loader.get("secrets.yml")["openfigi"]}' \
        --data '{"query":"IMB"}'

    :jobs : list(dict)
        A list of dicts that conform to the OpenFIGI API request structure. See
        https://www.openfigi.com/api#request-format for more information. Note
        rate-limiting requirements when considering length of `jobs`.

    :list: (dict)
        One dict per item in `jobs` list that conform to the OpenFIGI API
        response structure.  See https://www.openfigi.com/api#response-fomats
        for more information.
    """
    openfigi_url = "https://api.openfigi.com/v1/mapping"
    openfigi_headers = {"Content-Type": "text/json"}
    if openfigi_apikey:
        openfigi_headers["X-OPENFIGI-APIKEY"] = openfigi_apikey
    response = requests.post(url=openfigi_url, headers=openfigi_headers, json=jobs)
    if response.status_code != 200:
        raise Exception("Bad response code {}".format(str(response.status_code)))
    return response.json()
