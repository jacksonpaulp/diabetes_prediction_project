import requests

url = 'http://localhost:8000/predict'

patient = {
  "highbp": 1,
  "highchol": 1,
  "cholcheck": 1,
  "smoker": 0,
  "stroke": 0,
  "heartdiseaseorattack": 0,
  "physactivity": 1,
  "fruits": 1,
  "veggies": 1,
  "hvyalcoholconsump": 0,
  "anyhealthcare": 1,
  "nodocbccost": 0,
  "diffwalk": 0,
  "sex": 1,
  "education": 3,
  "income": 4,
  "age": 5,
  "bmi": 25.5,
  "genhlth": 3,
  "menthlth": 0,
  "physhlth": 0
}

response = requests.post(url, json=patient)

predictions = response.json()

if predictions['diabetes']:
    print('patient is likely to have diabetes, take necessary actions')
else:
    print('patient is not likely to have diabetes')