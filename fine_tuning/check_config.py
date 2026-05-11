"""Quick sanity check: verifies GOOGLE_LLM_API_KEY is set and both Gemini calls work."""
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv(Path(__file__).parent / ".env")

import os

print("Checking GOOGLE_LLM_API_KEY...", end=" ")
key = os.getenv("GOOGLE_LLM_API_KEY")
if not key:
    print("MISSING — set GOOGLE_LLM_API_KEY in .env and retry")
    raise SystemExit(1)
print(f"found (ends in ...{key[-4:]})")

print("Testing get_ipa_string...", end=" ", flush=True)
from fine_tuning.data_preprocessing import get_ipa_string
ipa = get_ipa_string("Haitian Creole", "Bondye renmen nou")
print(f"OK → {ipa}")

print("Testing identify_phonetic_component_within_text...", end=" ", flush=True)
from fine_tuning.data_preprocessing import identify_phonetic_component_within_text
sample_text = "Bondye renmen nou tout. Li voye Pitit li sèl pou nou ka gen lavi."
result = identify_phonetic_component_within_text("Haitian Creole", ipa, sample_text)
print(f"OK → {result}")

print("\nAll checks passed.")
