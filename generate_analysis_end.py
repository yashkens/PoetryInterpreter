from transformers import GPTJForCausalLM, AutoTokenizer
import time
import random
import RAKE
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

stop_words = list(set(stopwords.words('english')))
rake = RAKE.Rake(stop_words)

start = time.time()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
model.parallelize()
end = time.time()
print(f'Model loaded. Time elapsed: {end-start}')

sad_comments = [
    "It's depressing to me, that not many undertand the true meaning",
    "It saddens me, that not many undertand",
    "It's unfair, that the poem was not appreciated",
    "Poem seems to be underappreciated",
    "How unfortunate, that not many saw the real point",
    "It's almost irritating, that none seem to understand"
]

finish_prompts = [
    "All in all,",
    "To wrap up",
    "Concluding",
    "Ultimately,",
    "In brief,",
    "In short,"
]


def cut_leftovers(text):
    if text[-1] not in '.!?':
        text = '.'.join(text.split('.')[:-1]) + '.'
    return text


def continue_analysis(prev_text, prompt_suffix='', max_len=250, temp=0.9):
    prompt = prev_text + ' ' + prompt_suffix
    input_ids = tokenizer.encode(prompt.strip(), return_tensors="pt").cuda()
    output = model.generate(input_ids, do_sample=True, max_length=max_len, temperature=temp)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return cut_leftovers(text)


def finish_analysis(prompt):
    length = len(prompt.split())+100

    # analysis continuation
    analysis_contd = continue_analysis(prompt, max_len=length)

    # adding some sadness and irritation
    sad_prompt = random.choice(sad_comments)
    sad_contd = continue_analysis(analysis_contd, prompt_suffix=sad_prompt, max_len=length+100, temp=0.8)

    # bring to conclusion
    end_prompt = random.choice(finish_prompts)
    final = continue_analysis(sad_contd, prompt_suffix=end_prompt, max_len=length+150, temp=0.8)
    return final


def get_keywords(sent):
    words = rake.run(sent, minFrequency=1, maxWords=4)
    keywords = [word[0] for word in words]
    return keywords


def latex_format(text):
    poem, analysis = text.split('Poem Analysis:\n')

    keywords = get_keywords(analysis)
    keywords = [k for k in keywords if len(k.split()) > 1]
    for word in keywords:
        analysis = analysis.replace(word, '\\textbf{' + word + '}')

    parts = analysis.split('\n\n')
    for i, part in enumerate(parts):
        sents = sent_tokenize(part)
        if len(sents) > 3:
            parts[i] = ' '.join(sents[:3]) + '\n\n' + ' '.join(sents[3:])
    analysis = '\n\n'.join(parts)

    lines = poem.split('\n')
    for i, line in enumerate(lines):
        if line.strip() != '':
            lines[i] = '\\textit{' + line + '}'
    poem = '\n'.join(lines)
    poem = '\\begin{center}\n' + poem + '\n\end{center}'

    poem = poem.replace('\n', '\n\\\\')
    poem = poem.replace('\\\n', r'\break')
    poem = poem.replace(r'\break', r'break')
    poem = poem.replace(r'\break\break', r'\break')
    analysis = analysis.replace('\n', '\n\\\\')

    return poem + '\n\n' + analysis


with open('poems.txt', 'r', encoding='utf-8') as f:
    p_text = f.read()
with open('ana.txt', 'r', encoding='utf-8') as f:
    a_text = f.read()

poems = p_text.split('\n###\n')
anas = a_text.split('\n###\n')

prompts = []

for i in range(len(poems)):
    text = poems[i] + '\n\nPoem Analysis:\n' + anas[i]
    prompts.append(text)

answers = []

for prompt in tqdm(prompts):
    final = finish_analysis(prompt)
    res = latex_format(final)
    answers.append(res)

full = '\n\n\\newpage\n\n'.join(answers)
with open('to_overleaf.txt', 'w', encoding='utf-8') as f:
    f.write(full)
