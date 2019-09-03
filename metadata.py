categories = [
    "adenocarcinoma",
    "agc",
    "asch",
    "ascus",
    "dysbacteriosis",
    "hsil",
    "lsil",
    "monilia",
    "normal",
    "vaginalis"
]

cat2id = dict([(name, idx + 1) for idx, name in enumerate(categories)])
id2cat = dict([(idx + 1, name) for idx, name in enumerate(categories)])
