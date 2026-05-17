# src/mock_search.py
def mock_search(query, api_key=None, count=5):
    # Returns canned results helpful for testing
    examples = [
        {"title":"Common cold - Symptoms and causes - Mayo Clinic", "url":"https://www.mayoclinic.org/cold", "snippet":"Symptoms of the common cold include runny nose, sore throat, cough..."},
        {"title":"Flu (influenza) - Symptoms & causes - CDC", "url":"https://www.cdc.gov/flu", "snippet":"Influenza symptoms: fever, cough, sore throat, body aches, fatigue..."},
        {"title":"Migraine - Overview - NHS", "url":"https://www.nhs.uk/migraine", "snippet":"Migraine symptoms: severe headache, nausea, sensitivity to light and sound..."},
        {"title":"Gastroenteritis (stomach flu) - WebMD", "url":"https://www.webmd.com/gastroenteritis", "snippet":"Symptoms include diarrhea, vomiting, abdominal cramps..."},
        {"title":"Urinary tract infection - Symptoms - Cleveland Clinic", "url":"https://www.clevelandclinic.org/uti", "snippet":"UTI symptoms: burning with urination, frequent urination, lower abdominal pain..."}
    ]
    return examples[:count]
