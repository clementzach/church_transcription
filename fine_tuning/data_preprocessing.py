from transformers import pipeline
import torch
import json

# HuggingFace model ID — change to whichever Gemma 4 variant you want to use.
# Requires ~62 GB unified memory for bfloat16; use a quantized GGUF via llama-cpp
# or set load_in_4bit=True (needs bitsandbytes) if memory is tight.
MODEL_ID = "google/gemma-4-12b-it"

_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = pipeline(
            "text-generation",
            model=MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # uses MPS on Apple Silicon, CUDA if available, else CPU
        )
    return _pipe


def _generate(prompt: str) -> str:
    pipe = _get_pipe()
    messages = [{"role": "user", "content": prompt}]
    result = pipe(messages, max_new_tokens=512, do_sample=False)
    # chat-template pipelines return messages list; last entry is the assistant turn
    return result[0]["generated_text"][-1]["content"].strip()


def get_ipa_string(lang_name, utterance):
    """
    Get a phonetic representation of an input string using IPA.
    Used to convert possibly-garbled speech-to-text outputs into phonetic intermediaries.
    """
    prompt = (
f"""Write this possibly garbled {lang_name} transcription out using the international phonetic alphabet.

Break up words into individual syllables and separate each syllable with a " | " delimiter.

If words appear in other languages, include the phonetic transcriptions of those words in your response.

Each chunk should contain only one syllable. If there are multi-syllable words, break each one up and put another " | " delimiter between them.

Do not include any explanation or surrounding text.

Example input: ```I acknowledged that I made a mistake and asked for forgiveness```
Example output: ```aɪ | æk | nɒl | ɪdʒd | ðæt | aɪ | meɪd | ə | mɪs | teɪk | ænd | æskt | fər | fər | ɡɪv | nəs```
Your input: ```{utterance}```"""
    )
    return _generate(prompt)


def identify_phonetic_component_within_text(lang_name, phonetic_string, possible_matching_string):
    example_input = {
"phonetic_string": "ki sa pa pa nu ki na syɛl la de zi re də nu mɛm",
"possible_matching_string": "Jezikri toujou ap aple nou, epi Li itilize nou, sèvitè òdinè Li yo, pou mennen pitit Li yo vini jwenn Li. Kisa Papa nou ki nan Syèl la dezire de nou menm? Èske w konprann ke lè w te nan egzistans premòtèl ou a, Papa nou ki nan Syèl la t ap prepare w pou lavi ou sou tè a?"
}

    example_input_nomatch = {
"phonetic_string": "ɛ pi ti ɡa sɔ̃ ɑ̃ tɔ̃ bə su ʃə val la ɛ pi li ble sə pu la vi",
"possible_matching_string": """Mond enkonstan sa a sanble l souvan sekwe pa tanpèt, ensèten, pafwa plen chans, epi—twò souvan—pa gen chans. Poutan, nan mond tribilasyon sa a,1 nou konnen ke tout bagay ap travay ansanm pou byen moun ki renmen Bondye yo.2 Anfèt, si nou mache dwat epi nou sonje alyans nou yo, tout bagay ap travay ansanm pou byen nou."""
}

    example_input_partial_only = {
"phonetic_string": "sak di mã ti ka ta te rã pli pu jo ʒã de te ",
"possible_matching_string": """Non Jezikri te anba nèt e sètènman li pat an karaktè gra. Chak dimanch kat la te ranpli. Pou yon jèn detantè Prètriz Aawon ke m te ye, Sentsèn ak reyinyon Sentsèn te pran yon sans nouvo, elaji, ak espirityèl pou mwen. Mwen tap tann jou Dimanch yo ak opòtinite pou m pran Sentsèn nan avèk enpasyans paske konpreyansyon mwen sou Sakrifis Ekspyatwa Sovè a t ap transfòme mwen. Chak Dimanch, jis jounen jodi a, lè m pran Sentsèn, mwen kapab wè kat mwen an epi revize lis mwen an. Kounyeya se Sovè limanite ki toujou sou lis mwen an, anvan tout lòt bagay."""
}

    current_input = {
"phonetic_string": phonetic_string.replace(" | ", " "),
"possible_matching_string": possible_matching_string.replace("\n", "").replace('"', '')
}

    prompt = (
f"""Your role is to take a phonetic {lang_name} input and match it to the portion of the provided text that matches the phonetic syllables exactly, with a one-to-one mapping between each chunk in the phonetic input and each syllable of words you include in your output.

Only match the portion of the provided text that matches the phonetic transcription. Return partial sentences or paragraphs if the phonetic input only covers part of a text input.

Do not include any additional words in the provided text.

If the phonetic string does not match anything in the provided text, write out ```---UNMATCHED---``` and no additional text.

Example input 1: ```{json.dumps(example_input)}```
Example output 1: ```Kisa Papa nou ki nan Syèl la dezire de nou menm?```

Example input 2: ```{json.dumps(example_input_nomatch)}```
Example output 2: ```---UNMATCHED---```

Example input 3: ```{json.dumps(example_input_partial_only)}```
Example output 3: ```Chak dimanch kat la te ranpli. Pou yon jèn detantè```

Do not write out any context or additional output. All syllables in words provided in your output should map directly to exactly one syllable provided in the input.

Your input: ```{json.dumps(current_input)}```
"""
    )
    return _generate(prompt)
