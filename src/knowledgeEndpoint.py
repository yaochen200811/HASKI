from fastapi import FastAPI
from pydantic import BaseModel
from threading import Thread
from uuid import uuid4
import time
import requests
from openai import OpenAI
import json
import os
import hashlib
import pickle
import traceback
import yaml

# --------------------------------------------------------------------------
# Load system configuration
# --------------------------------------------------------------------------
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

CACHE_DIR = CONFIG["knowledge"]["cache_dir"]
EMAIL = CONFIG["knowledge"]["email"]

client_url = CONFIG["knowledge"]["client_url"]
client_key = CONFIG["knowledge"]["client_key"]
MODEL = CONFIG["knowledge"]["model"]


os.makedirs(CACHE_DIR, exist_ok=True)

HEADERS = {
    f"User-Agent": "LLMKnowledgeExtraction/1.0 ({EMAIL})"
}

# Initialize app
app = FastAPI()

# Initialize pipeline
client = OpenAI(base_url=client_url, api_key=client_key)

# In-memory store for question → status and result
knowledge_store = {}

# --------------------------------------------------------------------------
# Util Functions
# --------------------------------------------------------------------------

# Request model
class QuestionRequest(BaseModel):
    question: str
    
def get_cache_filename(url: str, params: dict) -> str:
    """
    Generate a deterministic cache filename for an HTTP request.

    The request URL and query parameters are hashed into a unique filename,
    allowing identical Wikidata requests to reuse previously downloaded
    responses and reduce network traffic.
    """
    key = f"{url}?{json.dumps(params, sort_keys=True)}"
    hashed = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hashed}.pkl")

def get_from_cache(url: str, params: dict):
    """
    Retrieve a cached HTTP response.

    Returns
    -------
    dict or None

        Cached response if available.

        None otherwise.
    """
    cache_file = get_cache_filename(url, params)
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None

def save_to_cache(url: str, params: dict, data):
    """
    Persist an HTTP response to the local cache.

    Cached responses avoid repeated requests to Wikidata and help prevent
    API rate limiting.
    """
    cache_file = get_cache_filename(url, params)
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)

def get_llm_cache_filename(prompt: str, model: str) -> str:
    """
    Generate a deterministic cache filename for LLM topic extraction.
    """
    key = f"{model}:{prompt}"
    hashed = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"llm_{hashed}.pkl")

def get_from_llm_cache(prompt: str, model: str):
    """
    Retrieve cached topic extraction results produced by the language model.

    Returns
    -------
    list or None
        Previously extracted search topics if available.
    """
    cache_file = get_llm_cache_filename(prompt, model)
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "data" in obj:
            return obj["data"]
        return obj
    return None

def save_to_llm_cache(prompt: str, model: str, data):
    """
    Persist LLM topic extraction results to the local cache.
    """
    cache_file = get_llm_cache_filename(prompt, model)
    payload = {"data": data}
    with open(cache_file, "wb") as f:
        pickle.dump(payload, f)

def safe_request(url: str, params: dict = {}, max_retries: int = 5):
    """
    Execute a cached HTTP GET request with automatic retry.

    If the request is rate limited (HTTP 429), it waits for the
    server-specified retry interval before attempting again.

    Parameters
    ----------
    url : str
        Target API endpoint.

    params : dict
        URL query parameters.

    max_retries : int
        Maximum retry attempts after rate limiting.

    Returns
    -------
    dict or None
        JSON response from the server, or None if all attempts fail.
    """
    cached = get_from_cache(url, params)
    if cached is not None:
        return cached

    for _ in range(max_retries):
        response = requests.get(url, params=params, headers=HEADERS)
        if response.status_code == 200:
            json_data = response.json()
            save_to_cache(url, params, json_data)
            return json_data
        elif response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "5"))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
        else:
            print(f"Request failed: {response.status_code}")
            break
    return None

def extract_search_topics(question: str, max_topics: int = 3) -> list:
    """
    Extract Wikidata search targets from a user question using an LLM.

    For example,

        Question:
            "Who won the Nobel Prize in Physics in 2024?"

    may produce

        [
            {
                "entity": "2024 Nobel Prize in Physics",
                "props": [
                    "winner",
                    "motivation"
                ]
            }
        ]

    The structured output is validated using a JSON schema before being
    returned.

    Results are cached to avoid repeated LLM inference.
    """
    system_prompt = (
        "You are a tool that identifies entities and their factual properties.\n"
        f"Given the question below, if knowledge is required, extract up to {max_topics} key entities and a list of relevant properties for each entity to perform a Wikidata search.\n"
    )

    user_prompt = f"Question: {question}"

    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    # --- Check cache first ---
    cached = get_from_llm_cache(full_prompt, MODEL)
    if cached is not None:
        return cached

    # --- Define output schema ---
    response_schema = {
        "title": "Entity",
        "description": "Key entities and a list of required properties for every entity for Wikidata search",
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "props": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
        },
    }

    # --- LLM Call ---
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_schema", "json_schema": {"strict": True, "schema": response_schema}},
    )

    json_output = response.choices[0].message.content
    topics = json.loads(json_output)

    # --- Cache the result ---
    save_to_llm_cache(full_prompt, MODEL, topics)

    return topics

def get_entity_id(label):
    """
    Search Wikidata for the entity identifier (QID) corresponding to an entity
    label.
    """
    url = "https://www.wikidata.org/w/rest.php/wikibase/v1/search/items"
    params = {
        "q": label,
        "language": "en",
    }
    response = safe_request(url, params=params)
    return response['results'][0]['id'] if response['results'] else None

def get_property_id(label):
    """
    Search Wikidata for the property identifier (PID).
    """
    url = f"https://www.wikidata.org/w/rest.php/wikibase/v1/search/properties"
    params = {
        "q": label,
        "language": "en",
    }
    response = safe_request(url, params=params)
    return response['results'][0]['id'] if response['results'] else None

def get_property_value(entity_id, property_id):
    """
    Retrieve the value of a Wikidata property.

    If the property value is itself another Wikidata entity (QID), the
    entity label is automatically resolved so that human-readable text is
    returned instead of an identifier.
    """
    url = f"https://www.wikidata.org/w/rest.php/wikibase/v1/entities/items/{entity_id}/statements"
    params = {
        "property": property_id,
    }
    response = safe_request(url, params=params)
    content = response[f'{property_id}'][0]['value']['content'] if response[f'{property_id}'] else None
    if isinstance(content, str) and content.startswith('Q'):
        url = f"https://www.wikidata.org/w/rest.php/wikibase/v1/entities/items/{content}/labels/en"
        content = safe_request(url)
    return content

def get_description(entity_id):
    """
    Retrieve the description of a Wikidata entity.
    """
    url = f"https://www.wikidata.org/w/rest.php/wikibase/v1/entities/items/{entity_id}/descriptions/en"
    response = safe_request(url)
    return response


def fetch_from_wikidata(topics: list) -> dict:
    """
    Retrieve factual descriptions and requested properties for every
    extracted topic.

    The retrieval procedure consists of:

        1. locate the Wikidata entity,
        2. obtain its description,
        3. retrieve every requested property,
        4. resolve referenced entities into readable labels.

    Parameters
    ----------
    topics : list

        Output of extract_search_topics().

    Returns
    -------
    dict

        Mapping from

            "<entity>"

        or

            "<entity>, <property>"

        to the retrieved factual value.
    """
    descriptions = {}

    
    for topic in topics:
        try:
            entity = topic['entity']
            props = topic['props']
            entity_id = get_entity_id(entity)
            if entity_id == 429:
                return 429
            value = get_description(entity_id)
            if value == 429:
                return 429
            descriptions[f"{entity}"] = value
        except Exception as e:
                continue

        for prop in props:
            try:
                property_id = get_property_id(prop)
                if property_id == 429:
                    return 429
                if not property_id:
                    continue
                value = get_property_value(entity_id, property_id)
                if value == 429:
                    return 429
                if value:
                    descriptions[f"{entity}, {prop}"] = value
            except Exception as e:
                continue
    
    
    return descriptions


def background_knowledge_fetch(question_id: str, question: str):
    """
    Execute the complete knowledge retrieval pipeline in a background thread.

    This function performs the following steps:

        1. Extract search topics using the language model.
        2. Retrieve factual information from Wikidata.
        3. Store the retrieved knowledge in the shared knowledge store.
        4. Update the retrieval status so that the Generation Service can
           poll for completion.

    Running the pipeline in a separate thread allows the Primary Reasoning
    Model to continue generating text while knowledge retrieval proceeds
    concurrently.
    """
    try:
        print(f"Start Getting topics for: {question}")
        topics = extract_search_topics(question)
        print(f"Start feching for: {topics}")
        descriptions = fetch_from_wikidata(topics)
        if descriptions == 429:
            knowledge_store[question_id] = {
                "status": "error",
                "error": "too many request"
            }
        else:
            if len(descriptions) == 0:
                descriptions = ""
            knowledge_store[question_id] = {
                "status": "ready",
                "question": question,
                "topics": topics,
                "knowledge": descriptions
            }
        print(f"Done: {question_id}")
    except Exception as e:
        traceback.print_exc()
        knowledge_store[question_id] = {
            "status": "error",
            "error": str(e)
        }

# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.post("/fetch-knowledge")
def fetch_knowledge(req: QuestionRequest):
    """
    Start asynchronous knowledge retrieval.

    A unique request identifier is returned immediately while the retrieval
    pipeline executes in a background thread. The caller should later poll
    /check-knowledge using the returned identifier.
    """
    question_id = str(uuid4())
    knowledge_store[question_id] = {
        "status": "fetching",
        "question": req.question
    }
    print(MODEL)
    Thread(
        target=background_knowledge_fetch,
        args=(question_id, req.question),
        daemon=True
    ).start()
    return {"id": question_id, "message": "Knowledge fetching started."}

@app.get("/check-knowledge")
def check_knowledge(id: str):
    """
    Query the current status of an asynchronous retrieval request.

    Possible statuses include:

        fetching
            Retrieval is still in progress.

        ready
            Knowledge has been successfully retrieved.

        error
            Retrieval failed.

        not_found
            Unknown request identifier.
    """
    entry = knowledge_store.get(id)
    if not entry:
        return {"status": "not_found", "message": "Invalid ID or expired."}
    return entry

