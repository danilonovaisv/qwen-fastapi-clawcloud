# test_api.py
import requests
import base64

BASE_URL = "http://localhost:8000"  # Altere para sua URL no Claw Cloud em produção

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    print("✅ Health:", r.json())

def test_load_model():
    r = requests.post(f"{BASE_URL}/load-model", json={"model_type": "UNET", "filename": "test.safetensors"})
    print("✅ Load Model:", r.json())

def test_generate():
    r = requests.post(f"{BASE_URL}/generate", json={
        "model": "mock_model",
        "positive": "a red apple",
        "negative": "blurry",
        "width": 512,
        "height": 512,
        "steps": 20,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "normal"
    })
    print("✅ Generate (latent length):", len(r.json()["latent"]))

def test_gemini():
    # Simula uma imagem pequena em base64
    from PIL import Image
    import io
    img = Image.new("RGB", (64, 64), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    try:
        r = requests.post(f"{BASE_URL}/gemini/generate-prompt", json={
            "prompt": "Describe this image",
            "image_b64": img_b64
        })
        print("✅ Gemini response:", r.json()["text"][:100] + "...")
    except Exception as e:
        print("⚠️ Gemini failed (maybe no API key):", e)

if __name__ == "__main__":
    print("🧪 Testing API...")
    test_health()
    test_load_model()
    test_generate()
    test_gemini()
    print("🏁 All tests completed.")
