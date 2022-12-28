from pipeline import PoemCollection
from poem_generation import PoemGenerator
from analysis_generation import AnalysisGenerator
from variables import TOKEN, POEM_START_MODEL_ID, POEM_CONTINUE_MODEL_ID, ANALYSIS_START_MODEL_ID
import unittest
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger()


class TestPipeline(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up variables for tests...")
        self.collection = PoemCollection()
        self.poem_generator = PoemGenerator(TOKEN, POEM_START_MODEL_ID, POEM_CONTINUE_MODEL_ID)

    @classmethod
    def setUpClass(cls):
        logger.info("Setting up the class...")

    def tearDown(self):
        logger.info("Tearing down...")

    def doCleanups(self):
        logger.info("Cleaning up...")

    def test_poem_generation(self):
        logger.info("Testing Poem Generation...")
        n1 = 2
        poems1 = self.collection.add_poems(num=n1, poem_generator=self.poem_generator)
        poem_len = self.collection.n
        self.assertIsInstance(poems1, list)
        self.assertEqual(poem_len, n1)

        n2 = 1
        poems2 = self.collection.add_poems(num=n2, poem_generator=self.poem_generator)
        poem_len = self.collection.n
        self.assertIsInstance(poems2, list)
        self.assertEqual(poem_len, n1 + n2)

    def test_latex_formatting(self):
        logger.info("Testing Latex formatting...")
        analysis_generator = AnalysisGenerator(TOKEN, ANALYSIS_START_MODEL_ID)

        poems1 = self.collection.add_poems(num=2, poem_generator=self.poem_generator)

        self.assertRaises(Exception, self.collection.format_into_latex)

        self.collection.analyze(analysis_generator)
        latex_texts = self.collection.format_into_latex()
        self.assertIsInstance(latex_texts, list)
        self.assertEqual(len(latex_texts), self.collection.n,
                         f"Latex tests: {len(latex_texts)}; Poems: {self.collection.n}")

