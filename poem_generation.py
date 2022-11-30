import tune_the_model as ttm
from nltk.tokenize import sent_tokenize
import numpy as np
from tqdm import tqdm
from variables import TOKEN, POEM_START_MODEL_ID, POEM_CONTINUE_MODEL_ID

ttm.set_api_key(TOKEN)

model = ttm.TuneTheModel.from_id(POEM_START_MODEL_ID)
cont_model = ttm.TuneTheModel.from_id(POEM_CONTINUE_MODEL_ID)


def generate_start():
    start_prompt = "The poem: "
    start = model.generate(start_prompt, num_hypos=3, temperature=0.9)
    lines_count = [s.count('\n') for s in start]
    start = start[np.argmax(lines_count)]
    return start


def generate_full_poem(stanza_nums=4, context_len=1):
    start = generate_start()
    prevs = [start]
    stanzas = [start]

    for i in range(stanza_nums):
        # print(f'Generating Part №{i + 1}...')
        if i == 0:
            cont = cont_model.generate(start,
                                       num_hypos=3,
                                       min_tokens=15,
                                       max_tokens=190,
                                       temperature=0.8)
        else:
            cont = cont_model.generate('\n'.join(prevs),
                                       num_hypos=3,
                                       min_tokens=15,
                                       max_tokens=190,
                                       temperature=0.8)
        lines_count = [s.count('\n') for s in cont]
        cont = cont[np.argmax(lines_count)]
        if len(prevs) > context_len - 1:
            prevs = prevs[1:]
        prevs.append(cont)
        stanzas.append(cont)

    text = '\n\n'.join(stanzas)
    if text[-1] not in '.!?':
        text = ' '.join(sent_tokenize(text)[:-1])

    return text

poems = []
N = 25
for i in tqdm(range(N)):
    poem = generate_full_poem()
    poems.append(poem)

with open('poems.txt', 'w', encoding='utf-8') as f:
    f.write('\n###\n'.join(poems))

