import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

PROMPT = """
You are a span selector for molecular descriptions.

Task:
Select the character spans corresponding to MAXIMAL noun phrases (NPs)
that could be replaced as whole units in the dataset.

STRICT RULES:
- Output ONLY character spans as start,end (one per line).
- Do NOT rewrite, correct, or output the text itself.
- NPs MUST be MAXIMAL: include the full noun phrase with all complements.
- NEVER select partial noun phrases.
- NEVER split chemical or biological names.
- NEVER split around numbers, hyphens, or apostrophes.
- NEVER select spans starting with "of", "from", "with", "and", "or".
- NEVER select spans ending with "-", "'", or "of".
- Do NOT include connectors such as:
  "that is", "arising from", "It has a role as",
  "It is a conjugate acid of", "It is a conjugate base of".

If unsure, DO NOT select the span.

Example:

Text:
The molecule is a cyanine dye and an organic potassium salt.

Output:
16,55

Now process this text:

Text:
The molecule is an organophosphate oxocation that is the dication of sn-glycerol 3-phosphate arising from deprotonation of both phosphate OH groups. It has a role as a human metabolite and a Saccharomyces cerevisiae metabolite. It is a conjugate acid of a sn-glycerol 3-phosphate.
"""

inputs = tokenizer(
        PROMPT,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.15,
        no_repeat_ngram_size=8,
        eos_token_id=tokenizer.eos_token_id,
    )

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)
