# https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/nlp/Token_Classification_Named_Entity_Recognition.ipynb#scrollTo=DQhsamclRtxJ
from nemo.collections.nlp.models import TokenClassificationModel

queries = [
    'Jerry had appendicitis last week so I will not be going to the concert.',
    'While George was at the Walmart he read an article describing the dangers of Lyme disease.',
    'We bought four shirts from the Nvidia gear store in Santa Clara.',
    'Tony was just at the hospital and was diagnosed with diabetes'
]

finetuned_model = TokenClassificationModel.restore_from('./demo_ner_model.nemo')

results = finetuned_model.add_predictions(queries)

for q, r in zip(queries, results):
    print("QUERY: " + q + "\nRESULT: " + r + "\n\n")

