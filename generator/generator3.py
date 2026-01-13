import spacy

nlp = spacy.load("en_core_web_sm")

text = """
The molecule is a tetrahydroxyflavanone that is flavanone substituted by hydroxy groups at positions 5, 7, 3' and 4' respectively.
It is a tetrahydroxyflavanone and a member of 3'-hydroxyflavanones.
"""
doc = nlp(text)
def get_subjects(sent):
    return [tok for tok in sent if tok.dep_ in ("nsubj", "nsubjpass")]
def get_noun_chunks(sent):
    return list(sent.noun_chunks)
def get_relative_clauses(sent):
    return [tok for tok in sent if tok.dep_ == "relcl"]

doc = nlp(text)

for sent in doc.sents:
    print("PHRASE :", sent.text)

    subjects = get_subjects(sent)
    print("  Sujet :", [s.text for s in subjects])

    noun_chunks = get_noun_chunks(sent)
    print("  Groupes nominaux :")
    for nc in noun_chunks:
        print("   -", nc.text)

    relatives = get_relative_clauses(sent)
    print("  Propositions relatives :", [r.text for r in relatives])

    print()
