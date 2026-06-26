from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import whoami, login
from typing import List
import yaml

# --------------------------------------------------------------------------
# Load system configuration
# --------------------------------------------------------------------------
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

mode = CONFIG["timing"]["mode"]
device = CONFIG["timing"]["device"]

app = FastAPI()

login(token=CONFIG["timing"]["huggingface_token"])

class TimingRequest(BaseModel):
    question: str
    output: str
    attention: List[float]

tokenizer = None
timing_model = None

timing_model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(timing_model_name)

# --------------------------------------------------------------------------
# Util Functions
# --------------------------------------------------------------------------

def switch_mode(mode):
    """
    Load the IPM corresponding to the selected mode.

    Different checkpoints implement different timing prediction strategies.
    Reloading allows the service to switch between models without restarting
    the server.

    Parameters
    ----------
    mode : str
        Identifier of the timing prediction model to load.

        Supported values include:

        • KI-RL-AS
        • KI-RL-AS-B
    """
    global tokenizer, timing_model    
    # Select the checkpoint corresponding to the requested timing policy.
    checkpoint_dir = "model/base"

    if mode == 'KI-RL-AS':
        checkpoint_dir = "model/ppo/KI-RL-AS"
        timing_model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            device_map=device,
        )
    elif mode == 'KI-RL-AS-B':
        checkpoint_dir = "model/ppo/KI-RL-AS-B"
        timing_model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            device_map=device,
        )

def check_good_timing(question, output, attention):
    """
    Predict whether external knowledge should be injected.

    The IPM receives the current reasoning state as
    a natural language prompt consisting of:

        • the original question,
        • the partially generated reasoning,
        • the attention distribution over the question and previously
          generated paragraphs.

    The model performs binary classification by generating either
    "True" or "False", indicating whether the current generation step
    is an appropriate injection point.

    Parameters
    ----------
    question : str
        Original user question.

    output : str
        Partial response generated so far.

    attention : List[float]
        Attention assigned to the original question and each completed
        reasoning paragraph.

    Returns
    -------
    str

        "True"
            Inject external knowledge.

        "False"
            Continue generation before injecting knowledge.
    """
    att_total = sum(attention)
    att_str = ""
    for i in range(1, len(attention)):
        att_str += f"Paragraph {i}: {attention[i] / att_total}"
    prompt = f'''Given the following question and partial model output, determine whether it is appropriate to add additional knowledge at this point in the response.

Question:

{question}

Partial Output:

{output}

Attention Distribution:

Question: {attention[0] / att_total}
{att_str}

Final Answer:
Respond with only True or False: '''
    inputs = tokenizer(prompt, return_tensors="pt").to(timing_model.device)
    outputs = timing_model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][-1], skip_special_tokens=True)


# Load the default Injection Prediction Model during service startup.
switch_mode(mode)

# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.post("/timing")
def evaluate(req: TimingRequest):
    """
    Evaluate whether the current reasoning state is suitable for
    knowledge injection.

    The endpoint forwards the request to the IPM
    and returns the binary prediction used by the Generation Service.
    """
    timing = check_good_timing(req.question,
        req.output,
        req.attention)
    return {"timing": timing}

@app.post("/reload-model")
def reload_model(mode: str):
    """
    Reload the IPM.

    Parameters
    ----------
    mode : str
        Timing prediction model to load.

    Returns
    -------
    dict
        Reload status and active model identifier.
    """
    global timing_model

    if timing_model is not None:
        del timing_model
        torch.mps.empty_cache() 
        print("Old model unloaded")

    switch_mode(mode)

    return {"status": "reloaded", "mode": mode}