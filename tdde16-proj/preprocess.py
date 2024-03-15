import pandas as pd
import json
from zipfile import ZipFile
import spacy
import re
from tqdm import tqdm

# anforande-202122 är trasig


DATASETS = ["anforande-200910", "anforande-201314", 
                  "anforande-201718", "anforande-202223"]



nlp = spacy.load("sv_core_news_sm")

# preprocess all speeches from an entire year
def preprocess_zip_lda(filename):
    out = []
    with ZipFile("raw_datasets/" + filename + ".json.zip", 'r') as dataset:
        for file in tqdm(dataset.namelist()):
            if file.endswith(".json"):
                with dataset.open(file) as f:
                    data = json.load(f)
                data = data["anforande"]

                if not data["anforandetext"]:
                    continue
                speech = data["anforandetext"]
                doc = nlp(speech)
                speech_tokens = []
                for token in doc:
                    if not token.is_stop and token.is_alpha and len(token) > 2:
                        speech_tokens.append(token.text)
                out.append(speech_tokens)
           
    return out
                

def preprocess_zips():
    """ Takes the relevant information from each speech
    Returns a DataFrame containing year, speech and party
    """

    speech_data = []
    for name in DATASETS: 
        with ZipFile("raw_datasets/" + name + ".json.zip", 'r') as dataset:
            print(name)
            for file in tqdm(dataset.namelist()):
                if file.endswith(".json"):
                    with dataset.open(file) as f:
                        data = json.load(f)
                    data = data["anforande"]

                    if not data["anforandetext"]:
                        continue

                    speech = data["anforandetext"]
                    re.sub(re.compile("<.*?>"), "", speech)  # remove html tags

                    relevant_data = {
                        "year": data["dok_rm"],
                        "date": data["dok_datum"].split()[0],
                        "party": data["parti"],
                        "speech": speech
                    }
                    temp_df = pd.DataFrame([relevant_data])
                    speech_data += [temp_df]

    df = pd.concat(speech_data, ignore_index=True)


    
    with ZipFile("sp.json.zip", 'w') as z:
        z.writestr("sp.json", data=df.to_json(orient="records"))
    
if __name__ == "__main__":
    preprocess_zips()
