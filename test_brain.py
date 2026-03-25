import requests
from io import BytesIO
from PIL import Image

url = "http://localhost:8000"
res = requests.post(f"{url}/api/auth/login", json={"username":"testuser2", "password":"Test@1234"})
if res.status_code != 200:
    print(f"Login failed: {res.text}")
    exit(1)
token = res.json()['token']

img = Image.new('RGB', (224, 224), color = 'white')
buf = BytesIO()
img.save(buf, format='JPEG')
buf.seek(0)

headers = {"Authorization": f"Bearer {token}"}
files = {"file": ("test.jpg", buf, "image/jpeg")}
try:
    res = requests.post(f"{url}/api/brain", headers=headers, files=files)
    print(f"Status: {res.status_code}")
    import json
    parsed = res.json()
    # Mask base64 to avoid huge output
    if 'images' in parsed and 'original' in parsed['images']:
        parsed['images']['original'] = "<base64_hidden>"
    print(f"Response: {json.dumps(parsed)}")
except Exception as e:
    print(f"Request failed: {e}")
