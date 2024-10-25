import pytest
from pathlib import Path
import json
from application import application
import requests
import time
import csv


BASE_URL = 'placeholder'

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

def test_latency_fake1(client):
    timestamps = []
    for _ in range(100):
        start_time = time.time()
        #response = requests.post(BASE_URL, json={'text': text})
        client.post('/load_model', json={'text': 'Noise and light pollution from car traffic kills roadside trees.'})
        end_time = time.time()
        timestamps.append(end_time - start_time)

    write_to_csv('pollution killed trees', timestamps)

def test_latency_fake2(client):
    timestamps = []
    for _ in range(100):
        start_time = time.time()
        #response = requests.post(BASE_URL, json={'text': text})
        client.post('/load_model', json={'text': 'Toronto metro relies on cigarette smoke of passengers to keep passageways free of insects.'})
        end_time = time.time()
        timestamps.append(end_time - start_time)

    write_to_csv('cigarettes as insect repellent', timestamps)

def test_latency_real1(client):
    timestamps = []
    for _ in range(100):
        start_time = time.time()
        #response = requests.post(BASE_URL, json={'text': text})
        client.post('/load_model', json={'text': 'ECE444 TAs go on strike as their work is being done by students attending the course.'})
        end_time = time.time()
        timestamps.append(end_time - start_time)

    write_to_csv('ECE444 TA strike', timestamps)

def test_latency_real2(client):
    timestamps = []
    for _ in range(100):
        start_time = time.time()
        #response = requests.post(BASE_URL, json={'text': text})
        client.post('/load_model', json={'text': 'Judge Rules White Girl Will Be Tried As Black Adult.'})
        end_time = time.time()
        timestamps.append(end_time - start_time)

    write_to_csv('ONN heading', timestamps)

def write_to_csv(filename, timestamps):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Latency (seconds)'])
        for timestamp in timestamps:
            writer.writerow([timestamp])