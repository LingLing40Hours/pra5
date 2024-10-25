import pytest
from pathlib import Path
import json
from application import application


@pytest.fixture
def client():
    with application.app_context():
        yield application.test_client()

def test_fake1(client):
    result = client.post('/load_model', json={'text': 'Noise and light pollution from car traffic kills roadside trees.'})
    assert result.json['prediction'] == 'FAKE'

def test_fake2(client):
    result = client.post('/load_model', json={'text': 'Toronto metro relies on cigarette smoke of passengers to keep passageways free of insects.'})
    assert result.json['prediction'] == 'FAKE'

def test_real1(client):
    result = client.post('/load_model', json={'text': 'ECE444 TAs go on strike as their work is being done by students attending the course.'})
    assert result.json['prediction'] == 'REAL'

def test_real2(client):
    result = client.post('/load_model', json={'text': 'Judge Rules White Girl Will Be Tried As Black Adult.'})
    assert result.json['prediction'] == 'REAL'