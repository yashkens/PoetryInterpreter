import tune_the_model as ttm
from tqdm import tqdm
from variables import TOKEN, ANALYSIS_START

ttm.set_api_key(TOKEN)

model = ttm.TuneTheModel.from_id(ANALYSIS_START)

with open('poems.txt', 'r', encoding='utf-8') as f:
    poem_text = f.read()

poems = poem_text.split('\n###\n')


def start_analysis(poem):
    analysis_start = model.generate(poem, num_hypos=1, temperature=0.9, min_tokens=50, max_tokens=190)
    return analysis_start[0]


texts = []
for poem in tqdm(poems):
    texts.append(start_analysis(poem))

with open('ana.txt', 'w', encoding='utf-8') as f:
    f.write('\n###\n'.join(texts))
