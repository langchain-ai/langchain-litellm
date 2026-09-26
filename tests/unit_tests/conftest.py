import os

# Read the model map shipped in the locked litellm wheel, not the one fetched on import.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
