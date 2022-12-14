import RAKE
import logging
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from typing import List
from copy import copy

from poem_generation import PoemGenerator
from analysis_generation import AnalysisGenerator

logger = logging.getLogger()


class PoemCollection:
    """
    PoemCollection is able to generate and collect poems,
    analyze them, format into latex style and save any of these parts into files.

    Attributes
    ----------
    n : int
        Current number of poems in the collection.
    poems : List[str]
        List of all of the poems in the collection.
    """
    def __init__(self) -> None:
        stop_words = list(set(stopwords.words('english')))
        self._rake = RAKE.Rake(stop_words)
        self.n = 0
        self.poems = []
        self._analyzes = []
        self._latex_texts = []

    def __len__(self):
        return self.n

    def add_poems(self, num: int, poem_generator: PoemGenerator) -> List[str]:
        """Generates num number of poems and adds them to the existing collection."""
        for i in tqdm(range(num)):
            poem = poem_generator.generate_poem(stanza_nums=4)
            self.poems.append(poem)
        self.n += num
        return copy(self.poems[-num:])

    def save_poems(self, save_path: str, sep: str) -> None:
        """Saves all poems in the collection into a file with a given name with a given separator."""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(sep.join(self.poems))

    def save_analyzes(self, save_path: str, sep: str) -> None:
        """Saves all analyzes texts into a file with a given name with a given separator."""
        if len(self._analyzes) != len(self.poems):
            raise Exception(f"Number of analyzes is not equal to number of poems!\n"
                            f"Poems: {len(self.poems) }; Analyzes: {len(self._analyzes)}")
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(sep.join(self._analyzes))

    def save_poem_with_analyzes(self, save_path: str, sep: str) -> None:
        """
        Saves poems with analysis into a file with a given name with a given separator.
        Uses the separator to divide pairs poem & analysis between each other.
        """
        if len(self._analyzes) != len(self.poems):
            raise Exception(f"Number of analyzes is not equal to number of poems!\n"
                            f"Poems: {len(self.poems) }; Analyzes: {len(self._analyzes)}")
        texts = [self.poems[i] + '\n\nPoem Analysis:\n\n' + self._analyzes[i] for i in range(self.n)]
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(sep.join(texts))

    def save_to_latex(self, save_path: str) -> None:
        """
        Saves latex format of poems with analysis into a file with a given name.
        Formats pairs which were not formatted yet, first.
        """
        if len(self._latex_texts) != len(self.poems):
            self.format_into_latex()
        full_latex = '\n\n\\newpage\n\n'.join(self._latex_texts)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(full_latex)

    def analyze(self, analysis_generator: AnalysisGenerator) -> List[str]:
        """Generates analyzes for poems in the collections, which were not analyzed yet."""
        diff = len(self.poems) - len(self._analyzes)
        for i in tqdm(range(len(self._analyzes), len(self.poems))):
            analysis = analysis_generator.generate_analysis(self.poems[i])
            self._analyzes.append(analysis)
        return copy(self._analyzes[-diff:])

    def _get_keywords(self, sent: str) -> List[str]:
        """Finds keywords in a sentence using Rake algorithm."""
        words = self._rake.run(sent, minFrequency=1, maxWords=4)
        keywords = [word[0] for word in words]
        return keywords

    def format_into_latex(self) -> List[str]:
        """
        Formats pairs of poems with analysis so they could be transferred to Latex files.
        Keywords and marked bold.
        Requires the same number of poems and analyzes.
        """
        if len(self._analyzes) != len(self.poems):
            raise Exception(f"Number of analyzes is not equal to number of poems!\n"
                            f"Poems: {len(self.poems) }; Analyzes: {len(self._analyzes)}")

        diff = len(self._analyzes) - len(self._latex_texts)
        for i in tqdm(range(len(self._latex_texts), len(self.poems))):
            analysis = self._analyzes[i]
            poem = self.poems[i]

            # find keywords and make them bold
            keywords = self._get_keywords(analysis)
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

            self._latex_texts.append(poem + '\n\n' + analysis)
        return copy(self._latex_texts[-diff:])
