from fastapi.testclient import TestClient
from app.delays_api import app, AirportInput

client = TestClient(app)

def test_welcome():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the flight delay estimation service. API is working."}

def test_predict_incomplete():
    response = client.get("/predict/search")
    assert response.status_code == 405

def test_predict_incomplete2():
    response = client.get("predict/search", headers={"arrival_airport": "PDY", "departure_time": "0400", "arrival_time": "0800"
})
    assert response.status_code == 405

def test_predict():
    airport = AirportInput(arrival_airport="SLC", departure_time="900", arrival_time="1230")
    assert airport.arrival_airport=="SLC"
    assert airport.departure_time=="900"
    assert airport.arrival_time=="1230"