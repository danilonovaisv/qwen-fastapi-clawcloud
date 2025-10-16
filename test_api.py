# test_api.py (atualizado)
import requests

BASE_URL = "https://ulkabjpxlzur.sg-members-1.clawcloudrun.com"

print("🧪 Testando health...")
print(requests.get(f"{BASE_URL}/health").json())

print("\n🧪 Testando generate...")
res = requests.post(f"{BASE_URL}/generate", json={
    "model": "mock",
    "positive": "a red apple",
    "negative": "",
    "width": 512,
    "height": 512,
    "steps": 20,
    "cfg": 1.0,
    "sampler": "euler",
    "scheduler": "simple"
})
print("✅ Imagem gerada (base64 length):", len(res.json()["latent"]))
