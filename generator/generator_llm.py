from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

prompt = """You are a noun-phrase chunker for molecular descriptions.

Task:
Wrap each NOMINAL GROUP (noun phrase) that could be replaced in the dataset with the tags:
[[NP: ... ]]

Rules (STRICT):
- Do NOT change, add, remove, or reorder any character except inserting the tags.
- Only insert tags [[NP: and ]].
- Do NOT output anything else.
- A noun phrase here is a contiguous span denoting a chemical class, ion/state relation, molecule name, biological role, organism, or any entity-like phrase.
- DO NOT split inside chemical or biological names. Keep them intact (examples: "sn-glycerol 3-phosphate", "Saccharomyces cerevisiae", "NIR-2(2-)", "L-lysine").
- Do NOT create very small noun phrases (no splitting into parts like "sn-glycerol 3-" and "'s phosphate"). If unsure, keep the whole larger phrase as one NP.
- Do NOT wrap connectors such as: "The molecule is", "that is", "which is", "arising from", "It has a role as", "It is a conjugate acid of", "It is a conjugate base of". These connectors must remain outside NP tags.

Example:

Input:
The molecule is a cyanine dye and an organic potassium salt. It has a role as a fluorochrome. It contains a NIR-2(2-).

Output:
The molecule is [[NP: a cyanine dye and an organic potassium salt]]. It has a role as [[NP: a fluorochrome]]. It contains [[NP: a NIR-2(2-)]].

Now process this input:

Input:
The molecule is an organophosphate oxocation that is the dication of sn-glycerol 3-phosphate arising from deprotonation of both phosphate OH groups. It has a role as a human metabolite and a Saccharomyces cerevisiae metabolite. It is a conjugate acid of a sn-glycerol 3-phosphate.

Output:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.0,
    do_sample=False
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))