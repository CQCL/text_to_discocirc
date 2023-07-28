import spacy
from lambeq import BobcatParser
from parameterized import parameterized

from helpers.ccg_to_diag_test_pipeline import ccg_to_diag_test
from helpers.UnitTestBaseClass import UnitTestBaseClass

sentences = [
    "Alice thinks she likes Bob",
    "Charles walks and Alice likes her work , Bob and her cat",
    "Charles walks and Alice likes her work , Bob",
    "Alice likes her work , Bob and her cat",
    "Charles likes cats and Alice and Bob like their work",
    "Alice and Bob like their work and their job",
    "Dave like food and Alice , Bob and Eve like houses , their work , cats , their job and dogs",
    "Alice likes her work",
    'Mary extremely and quickly runs',
    "Alice likes Bob but she prefers her work",
    "Bob likes dogs and Alice likes cheese , her work , human and her life and Charlie and Dave like their computer",
    "Bob likes dogs and Alice likes cheese , her work , human , her life and her children",
    "Alice likes her work and her life",
    "Alice likes her boring work",
    "Tall Alice likes her boring work",
    "ALice think she will like Bob",
    'Alice , Bob and Claire who like beer and wine walked',
    'Alice likes her work and Bob also likes her work',
    'Alice who Bob gives flowers to in bedroom , walks',
    'Alice who Bob likes walks',
    'Alice and Bob like their work',
    'Alice likes her work and Bob likes his cat',
    'Alice thinks she is funny and she is happy and Bob and Claire think they are green',
    'Red Alice thinks she is funny',
    'Alice thinks she is funny and Bob thinks he is green',
    'Alice and Bob think they are funny',
    'While the hare was busy sleeping , his friend the tortoise won the race',
    'Alice prefers his work but she likes Bob',
    'Alice and Bob ran as they were afraid',
    'Boring Bob likes his boring self',
    'Despite her difficulty, Wilma came to understand the point',
    'Although he was busy with his boring work , Peter had enough of it',
    'He and his wife decided they needed a holiday',
    'They travelled to Spain because they loved the country very much',
    'Alice \'s mother likes her cat',
    'I thought the plane would be awful , but it was not',
    'Looking around he found the letter',
    'Alice quickly and slowly runs',
    'Alice runs quickly and slowly',
    'Alice and Bob',
    'Alice eats',
    'Alice quickly gives flowers',
    'Alice who Bob likes walks',
    'Alice who likes Bob walks',
    'Alice gives flowers with Bob',
    'Alice gives flowers to Bob',
    'Alice fully loves Bob',
    'Alice quickly eats',
    'Alice quickly eats fish',
    'Alice quickly eats red fish',
    'Alice quickly loves very red Bob',
    'Alice quickly rapidly loudly loves very very red Bob',
    'Alice quickly and rapidly loves Bob and very red Claire',
    'Alice quickly and rapidly loves Bob',
    'I know of Alice loving and hating Bob',
    'I surely know certainly of Alice quickly loving Bob',
    'I know certainly of Alice quickly loving Bob',
    "Alice knows of : Bob liking Claire and Dave hating Eve",
    'Alice knows that Bob loves Claire',
    'Alice runs to the kitchen',
    'Alice knows that Bob loves Claire , Claire hates Bob',
    'Alice loves Bob and Claire loves Dave',
    'Alice loves Bob and Bob loves Claire',
    'The lawyers went to work',
    'Before he went to the treasury',
    'I know of Alice loving Bob',
    'I know of Alice quickly loving Bob',
    'I know certainly of Alice quickly loving Bob',
    'I know that Alice hates Bob and Claire loves Dave',
    'Alice loves Bob and Claire hates Dave',
    'Alice quickly loves and hates Bob',
    'I dreamt that Alice went to the shop and bought ice cream',
    'Bob likes flowers that Claire picks',
    'Fred is no longer in the office',
    'How many objects is Mary carrying',
    'Alice quickly and hastily eats',
    'The son of a physicist , Mr. Hahn skipped first grade because his reading ability was so far above his classmates',
    'Alice , Bob and Claire',
    'Alice , Bob and Claire drank',
    'Alice likes what she knows',
    'Focus on what company knows best',
    "We did n't have much of a choice , Cray Computer 's chief financial officer , Gregory Barnum , said in an interview",
    'Alice who Bob gives flowers to , walks',
    'Alice who Bob gives flowers to in bedroom , walks',
    'Alice likes what Bob knows',
    'Alice likes what is known to Bob',
    'Before looking around , he noticed a letter',
    'He noticed a letter before looking around',
    'Alice and John drank water',
    'Alice , Bob and John drank water , beer and wine',
    'Alice , Bob and John who drank water , beer and wine walked',
    'Alice and Bob who drank water and wine walked',
    'If a farmer owns a donkey he beats it',
]

parser = BobcatParser()
config = {
    "draw_result": True,
    "draw_steps": False,
    "type_check_ccg": True,
    "semantic_rewrites": True,
    "coreference_resolution": True,
}
spacy_model = spacy.load('en_core_web_trf')
spacy_model.add_pipe('coreferee')

class CCGToDiagTests(UnitTestBaseClass):
    """
    Runs the ccg_to_diag_test for each sentence in the sentences list.
    """
    @parameterized.expand(sentences)
    def test_sequence(self, sentence):
        print(sentence)
        self.test_logger = sentence
        ccg_parse = parser.sentence2tree(sentence)
        print(ccg_parse.deriv())
        ccg_to_diag_test(self, config, ccg_parse, sentence, spacy_model)
