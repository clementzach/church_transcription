import anthropic
anthropic_client = anthropic.Anthropic()
import json

def get_ipa_string(lang_name, utterance):
    """
    Get a phonetic representation of an input string using IPA. 
    Used to convert possibly-garbled speech-to-text outputs into a phonetic intermediaries. 

    Args: 
        lang_name: The language to use. 
        utterance: An utterance to transcribe into IPA form.
        
    """
    ipa_resp = anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": (
f"""Write this possibly garbled {lang_name} transcription out using the international phonetic alphabet.

Break up words into individual syllables and separate each syllable with a " | " delimiter. 

If words appear in other langages, include the phonetic transcriptions of those words in your response. 

Each chunk should contain only one syllable. If there are multi-syllable words, break each one up and put another " | " delimiter between them.

Do not include any explanation or surrounding text.Do not include any explanation or surrounding text.

Example input: ```I acknowledged that I made a mistake and asked for forgiveness```
Example output: ```aɪ | æk | nɒl | ɪdʒd | ðæt | aɪ | meɪd | ə | mɪs | teɪk | ænd | æskt | fər | fər | ɡɪv | nəs```
Your input: ```{utterance}```"""
                    ),
                }],)
    ipa = ipa_resp.content[0].text.strip()
    return ipa


def identify_phonetic_component_within_text(lang_name, phonetic_string, possible_matching_string):
    example_input = {
"phonetic_string": "ki sa pa pa nu ki na syɛl la de zi re də nu mɛm",
"possible_matching_string": "Jezikri toujou ap aple nou, epi Li itilize nou, sèvitè òdinè Li yo, pou mennen pitit Li yo vini jwenn Li. Kisa Papa nou ki nan Syèl la dezire de nou menm? Èske w konprann ke lè w te nan egzistans premòtèl ou a, Papa nou ki nan Syèl la t ap prepare w pou lavi ou sou tè a?"
}

    example_input_nomatch = {
"phonetic_string": "ɛ pi ti ɡa sɔ̃ ɑ̃ tɔ̃ bə su ʃə val la ɛ pi li ble sə pu la vi",
"possible_matching_string": "Mond enkonstan sa a sanble l souvan sekwe pa tanpèt, ensèten, pafwa plen chans, epi—twò souvan—pa gen chans. Poutan, nan mond tribilasyon sa a,1 “nou konnen ke tout bagay ap travay ansanm pou byen moun ki renmen Bondye yo.”2 Anfèt, si nou mache dwat epi nou sonje alyans nou yo, “tout bagay ap travay ansanm pou byen nou.”"
}
    
    current_input = {
"phonetic_string": phonetic_string.replace(" | ", " "),
"possible_matching_string": possible_matching_string.replace("\n", "").replace('"', '')
}
    
    ipa_resp = anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": (
f"""Your role is to take a phonetic in {lang_name} input and match it to the corresponding portion of the provided text. 

If it does not match anything in the provided text, write out ```---UNMATCHED---``` and no additional text.

Example input 1: ```{json.dumps(example_input)}```
Example output 1: ```Kisa Papa nou ki nan Syèl la dezire de nou menm?```

Example input 2: ```{json.dumps(example_input_nomatch)}```
Example output 2: ```---UNMATCHED---```

Do not write out any context or additional output.

Your input: ```{json.dumps(current_input)}```
"""
                    ),
                }],)
    ipa = ipa_resp.content[0].text.strip()
    return ipa