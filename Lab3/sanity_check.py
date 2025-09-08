from radon.raw import analyze # lines of code
from radon.complexity import cc_visit # cyclomatic complexity
from radon.metrics import mi_visit # maintainability index

import torch
from transformers import AutoTokenizer, AutoModel

import sacrebleu
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

code_before = "Source Code (before)"
code_after = "Source Code (current)"

mi_del = "MI_Change"
cc_del = "CC_Change"
loc_del = "LOC_Change"
sem_sim = "Semantic_Similarity"
token_sim = "Token_Similarity"
sem_class = "Semantic_Class"
token_class = "Token_Class"
agree = "Classes_Agree"

def mi_cc_loc(code): # mi,cc,loc
    try:
        if not isinstance(code, str) or code.strip() == "":
            return None, None, None
        mi = mi_visit(code, True)
        cc_scores = cc_visit(code) # list of blocks
        avg_cc = sum(c.complexity for c in cc_scores) / len(cc_scores) if cc_scores else 0
        # loc, lloc, sloc, comments, multi, blank = analyze(code)
        tup = analyze(code)
        return mi, avg_cc, tup.loc
    except Exception as e:
        print(e)
        return None, None, None

def get_embedding(text):
    if text.strip() == "":
        hidden_size = model.config.hidden_size
        return torch.zeros(1, hidden_size)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1) # uses cls by default, wasn't great, hence using mean pooling here
    return embeddings

def cosine_similarity(vec1, vec2):
    return torch.nn.functional.cosine_similarity(vec1, vec2).item()

def codebert(code_bef, code_aft):
    emb_before = get_embedding(code_bef)
    emb_after = get_embedding(code_aft)
    sim = cosine_similarity(emb_before, emb_after)
    return sim

def comp_bleu(code_bef, code_aft):
    bleu = (sacrebleu.corpus_bleu([code_aft], [[code_bef]]).score)/100.0
    return bleu

code1 = "def add(a, b):\n    return a + b"
code2 = "def sum_numbers(x, y):\n    return x + y"

print("radon metrics")
mi, cc, loc = mi_cc_loc(code1)
print(f"code 1: MI={mi:.2f}, CC={cc:.2f}, LOC={loc}")
mi2, cc2, loc2 = mi_cc_loc(code2)
print(f"code 2: MI={mi2:.2f}, CC={cc2:.2f}, LOC={loc2}")
delmi = mi2-mi
delcc = cc2-cc
delloc = loc2-loc
print(f"change in mi = {delmi:.2f}")
print(f"change in cc = {delcc:.2f}")
print(f"change in loc = {delloc:.2f}")

print("codebert")
print("code 1 vs code 2:", codebert(code1, code2))

print("bleu")
print("code 1 vs code 2:", comp_bleu(code1, code2))