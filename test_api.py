import pytest
import httpx
from fastapi import status

# Example valid test payload
VALID_BODY = {
    "prompt": "Write a poem about the ocean.",
    "sys_prompt": "You are a helpful AI assistant.",
    "credentials": {"api_key": "dummy"},
    "model": "llama3.2",
    "retry": 2,
    "prompt_values": {"mood": "calm"},
    "ollama_host": "http://host.docker.internal:11434",
    "max_tokens": 512,
    "image": "https://marketplace.canva.com/EAGZEaj1Dl0/1/0/1280w/canva-beige-aesthetic-motivational-quote-instagram-post-WAe2YFurmmg.jpg"
}

BASE_URL = "http://127.0.0.1:8006/api"  # change if different

# ---------------------------
# 1️⃣ Happy path test
# ---------------------------
@pytest.mark.asyncio
async def test_vlad_test_success():
    """Check if the endpoint responds successfully."""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60) as client:
        response = await client.post("/VLAD/TEST", json=VALID_BODY)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "llm_response" in data
    assert "prompt_used" in data
    assert "ollama_host" in data

# ---------------------------
# 2️⃣ Missing required field
# ---------------------------
@pytest.mark.asyncio
async def test_vlad_test_missing_prompt():
    """Should return 422 if a required field is missing."""
    bad_body = VALID_BODY.copy()
    del bad_body["prompt"]

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
        response = await client.post("/VLAD/TEST", json=bad_body)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

# ---------------------------
# 3️⃣ Invalid data type
# ---------------------------
@pytest.mark.asyncio
async def test_vlad_test_invalid_type():
    """Send an invalid data type to trigger validation error."""
    bad_body = VALID_BODY.copy()
    bad_body["max_tokens"] = "five hundred"  # should be int

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
        response = await client.post("/VLAD/TEST", json=bad_body)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

# ---------------------------
# 4️⃣ Test retry parameter
# ---------------------------
@pytest.mark.asyncio
async def test_vlad_test_retry_param():
    """Ensure retry parameter is handled."""
    custom_body = VALID_BODY.copy()
    custom_body["retry"] = 5

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60) as client:
        response = await client.post("/VLAD/TEST", json=custom_body)

    assert response.status_code == 200
    data = response.json()
    assert data["ollama_host"] == custom_body["ollama_host"]

# ---------------------------
# 5️⃣ Test large max_tokens value
# ---------------------------
@pytest.mark.asyncio
async def test_vlad_test_large_token_limit():
    """Send a large max_tokens value to check system behavior."""
    custom_body = VALID_BODY.copy()
    custom_body["max_tokens"] = 20000

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120) as client:
        response = await client.post("/VLAD/TEST", json=custom_body)

    assert response.status_code in (200, 400, 422)

# ---------------------------
# 6️⃣ Test invalid Ollama host
# ---------------------------
@pytest.mark.asyncio
async def test_vlad_test_invalid_host():
    """Simulate invalid host URL."""
    bad_body = VALID_BODY.copy()
    bad_body["ollama_host"] = "http://invalid-host:1234"

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
        response = await client.post("/VLAD/TEST", json=bad_body)

    # Depending on implementation, could be 400 or 500
    assert response.status_code in (400, 500, 502)
