import RAKE
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize


class PoemCollection:
    def __init__(self, n, poem_generator):
        stop_words = list(set(stopwords.words('english')))
        self.rake = RAKE.Rake(stop_words)
        self.n = n
        self.poems = []
        self.analyzes = []
        self.latex_texts = []
        for i in tqdm(range(self.n)):
            poem = poem_generator.generate_full(stanza_nums=4)
            self.poems.append(poem)
        print(f"Generated all {self.n} poems!")

    def __getitem__(self, idx):
        return self.poems[idx], self.analyzes[idx]

    def save_poems(self, save_path, sep) -> None:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(sep.join(self.poems))

    def save_analyzes(self, save_path, sep) -> None:
        if len(self.analyzes) != self.n:
            raise Exception("No analyzes yet!")
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(sep.join(self.analyzes))

    def save_poem_with_analyzes(self, save_path, sep) -> None:
        if len(self.analyzes) != self.n:
            raise Exception("No analyzes yet!")
        texts = [self.poems[i] + '\n\nPoem Analysis:\n\n' + self.analyzes[i] for i in range(self.n)]
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(sep.join(texts))

    def save_to_latex(self, save_path) -> None:
        if len(self.latex_texts) == 0:
            self.format_into_latex()
        full_latex = '\n\n\\newpage\n\n'.join(self.latex_texts)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(full_latex)

    def analyze(self, analysis_generators) -> None:
        start_model, cont_model = analysis_generators
        for i in tqdm(range(self.n)):
            start = start_model.analyze_poem(self.poems[i])
            prompt = self.poems[i] + '\n\nPoem Analysis:\n' + start
            analysis = cont_model.finish_analysis(prompt)
            analysis = analysis.split('Poem Analysis:')[1].strip()
            self.analyzes.append(analysis)
        print(f"Generated all {self.n} analyzes!")

    def get_keywords(self, sent):
        words = self.rake.run(sent, minFrequency=1, maxWords=4)
        keywords = [word[0] for word in words]
        return keywords

    def format_into_latex(self) -> None:
        if len(self.analyzes) != self.n:
            raise Exception("Can't format a latex file without analyzes!")

        for i in tqdm(range(self.n)):
            analysis = self.analyzes[i]
            poem = self.poems[i]

            # find keywords and make them bold
            keywords = self.get_keywords(analysis)
            keywords = [k for k in keywords if len(k.split()) > 1]
            for word in keywords:
                analysis = analysis.replace(word, '\\textbf{' + word + '}')

            # add more new lines fro formatting
            parts = analysis.split('\n\n')
            for j, part in enumerate(parts):
                sents = sent_tokenize(part)
                if len(sents) > 3:
                    parts[j] = ' '.join(sents[:len(sents)//2]) + '\n\n' + ' '.join(sents[len(sents)//2:])
            analysis = '\n\n'.join(parts)

            # italicize poem lines
            lines = poem.split('\n')
            for i, line in enumerate(lines):
                if line.strip() != '':
                    lines[i] = '\\textit{' + line + '}'
            poem = '\n'.join(lines)
            poem = '\\begin{center}\n' + poem + '\n\end{center}'

            # a bit more formatting
            poem = poem.replace('\n', '\n\\\\')
            poem = poem.replace('\\\n', r'\break')
            poem = poem.replace(r'\break', r'break')
            poem = poem.replace(r'\break\break', r'\break')
            analysis = analysis.replace('\n', '\n\\\\')

            self.latex_texts.append(poem + '\n\n' + analysis)
