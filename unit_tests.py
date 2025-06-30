# tests/test_app.py
import io
import pytest
from app import app  # make sure this points at your app module

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_classify_valid_image(client):
    # sends a real JPG, expects 200 + a "label" string back
    with open("references/test-image.jpg", "rb") as img:
        data = {"image": (img, "test-image.jpg")}
        resp = client.post(
            "/classify",
            data=data,
            content_type="multipart/form-data"
        )
    assert resp.status_code == 200
    assert "label" in resp.json
    assert isinstance(resp.json["label"], str)


def test_classify_missing_image(client):
    # no file field at all → 400 + error message
    resp = client.post(
        "/classify",
        data={}, 
        content_type="multipart/form-data"
    )
    assert resp.status_code == 400
    assert "error" in resp.json


def test_classify_invalid_filetype(client):
    # non‐image extension → 400 + error message
    fake = io.BytesIO(b"notanimage")
    data = {"image": (fake, "file.txt")}
    resp = client.post(
        "/classify",
        data=data,
        content_type="multipart/form-data"
    )
    assert resp.status_code == 400
    assert "error" in resp.json


def test_enable_dark_mode(client):
    resp = client.post("/toggle-theme", json={"dark_mode": True})
    assert resp.status_code == 200
    assert resp.json["theme"] == "dark"


def test_disable_dark_mode(client):
    resp = client.post("/toggle-theme", json={"dark_mode": False})
    assert resp.status_code == 200
    assert resp.json["theme"] == "light"


def test_toggle_theme_missing_flag(client):
    resp = client.post("/toggle-theme", json={})
    assert resp.status_code == 400
    assert "error" in resp.json
