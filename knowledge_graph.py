import json
import urllib.parse
import urllib.request

# Google Knowledge Graph API key
api_key = "AIzaSyDPZceqCgLVGytRa14EOvYfcYarjfqMLm0"

query = 'university'
service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
params = {
    'query': query,
    # 'ids': 'kg:/m/0gg594v',
    'limit': 3,
    'indent': True,
    'key': api_key,
}

url = service_url + '?' + urllib.parse.urlencode(params)

with urllib.request.urlopen(url) as response:
    bb = response.read()
    obj = bb.decode('utf-8')

obj = json.loads(obj)
items = obj['itemListElement']
for element in items:
    print(element['result']['name'] + ' (' + str(element['resultScore']) + ')')

