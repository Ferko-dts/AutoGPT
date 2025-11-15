import pytest
import httpx
import base64
from fastapi import status

IMAGE_URL = "https://marketplace.canva.com/EAGZEaj1Dl0/1/0/1280w/canva-beige-aesthetic-motivational-quote-instagram-post-WAe2YFurmmg.jpg"

BASE_URL = "http://127.0.0.1:8006/api"  # change if your server runs elsewhere


async def get_base64_image_from_url(url: str) -> str:
    """Download image from URL and return it as a base64-encoded string."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")


@pytest.mark.asyncio
async def create_valid_body() -> dict:
    """Helper to create a valid payload with a base64 image."""
    image_b64 = await get_base64_image_from_url(IMAGE_URL)
    return {
        "prompt": "Write a poem about the ocean.",
        "sys_prompt": "You are a helpful AI assistant.",
        "credentials": {
            "id": "test-user-123",
            "provider": "openai",
            "type": "api_key",
            "api_key": "dummy"
        },
        "model": "llama3.2",
        "retry": 2,
        "prompt_values": {"mood": "calm"},
        "ollama_host": "http://host.docker.internal:11434",
        "max_tokens": 512,
        "image": image_b64,  # 👈 now base64
    }


# ---------------------------
# 1️⃣ Happy path test
# ---------------------------
@pytest.mark.asyncio
async def test_vlad_test_success_base64():
    """Check if the endpoint responds successfully with base64 image."""
    body = await create_valid_body()

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60) as client:
        response = await client.post("/VLAD/TEST", json=body)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "llm_response" in data
    assert "prompt_used" in data
    assert "ollama_host" in data


# ---------------------------
# 2️⃣ Missing required field
# ---------------------------
@pytest.mark.asyncio
async def test_vlad_test_missing_prompt_base64():
    """Should return 422 if 'prompt' field is missing."""
    body = await create_valid_body()
    del body["prompt"]

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
        response = await client.post("/VLAD/TEST", json=body)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# ---------------------------
# 3️⃣ Invalid data type
# ---------------------------
@pytest.mark.asyncio
async def test_vlad_test_invalid_type_base64():
    """Send an invalid data type to trigger validation error."""
    body = await create_valid_body()
    body["max_tokens"] = "five hundred"  # should be int

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
        response = await client.post("/VLAD/TEST", json=body)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# ---------------------------
# 4️⃣ Test retry parameter
# ---------------------------
@pytest.mark.asyncio
async def test_vlad_test_retry_param_base64():
    """Ensure retry parameter is handled correctly."""
    body = await create_valid_body()
    body["retry"] = 5

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60) as client:
        response = await client.post("/VLAD/TEST", json=body)

    assert response.status_code == 200
    data = response.json()
    assert data["ollama_host"] == body["ollama_host"]


# ---------------------------
# 5️⃣ Test large max_tokens value
# ---------------------------
@pytest.mark.asyncio
async def test_vlad_test_large_token_limit_base64():
    """Send a large max_tokens value to check system behavior."""
    body = await create_valid_body()
    body["max_tokens"] = 20000

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120) as client:
        response = await client.post("/VLAD/TEST", json=body)

    assert response.status_code in (200, 400, 422)


# ---------------------------
# 6️⃣ Test invalid Ollama host
# ---------------------------
@pytest.mark.asyncio
async def test_vlad_test_invalid_host_base64():
    """Simulate invalid host URL."""
    body = await create_valid_body()
    body["ollama_host"] = "http://invalid-host:1234"

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
        response = await client.post("/VLAD/TEST", json=body)

    assert response.status_code in (400, 500, 502)
