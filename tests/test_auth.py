from fastapi.testclient import TestClient
from src.api.routes import app

client = TestClient(app)


def test_default_admin_login():
    """Default admin (seeded on startup) can login and receives a token"""
    resp = client.post(
        "/api/v1/login",
        params={
            "username": "ece30861defaultadminuser",
            "password": "correcthorsebatterystaple123(!__+@**(A;DROP TABLE packages)"
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"
    assert "token" in data


def test_register_login_logout_happy_path():
    # register
    resp = client.post("/api/v1/register", params={"username": "testuser1", "password": "pass1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "registered"

    # login
    resp = client.post("/api/v1/login", params={"username": "testuser1", "password": "pass1"})
    assert resp.status_code == 200
    data = resp.json()
    token = data.get("token")
    assert token

    # logout
    resp = client.post("/api/v1/logout", params={"token": token})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "logged_out"


def test_register_duplicate():
    # register user
    resp = client.post("/api/v1/register", params={"username": "testuser2", "password": "pass2"})
    assert resp.status_code == 200

    # duplicate register should return 409
    resp = client.post("/api/v1/register", params={"username": "testuser2", "password": "pass2"})
    assert resp.status_code == 409


def test_login_wrong_password():
    # register user
    resp = client.post("/api/v1/register", params={"username": "testuser3", "password": "pass3"})
    assert resp.status_code == 200

    # wrong password
    resp = client.post("/api/v1/login", params={"username": "testuser3", "password": "wrong"})
    assert resp.status_code == 401


def test_logout_unknown_token():
    resp = client.post("/api/v1/logout", params={"token": "nonexistenttoken123"})
    assert resp.status_code == 404
