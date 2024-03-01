# https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/nlp/Token_Classification_Named_Entity_Recognition.ipynb#scrollTo=DQhsamclRtxJ
from nemo.collections.nlp.models import QAModel
from nemo.collections.nlp.models import TokenClassificationModel

model = QAModel.from_pretrained(model_name="qa_squadv2.0_bertlarge") 
results, nbest = model.inference('./demo_questions.json', num_samples=4)

tc_input = []

print("\n")
for question_id in results:
    print(results[question_id])
    tc_input.append(results[question_id][1])
print("\n")

finetuned_model = TokenClassificationModel.restore_from('./demo_ner_model.nemo')

tc_results = finetuned_model.add_predictions(tc_input)

for q, r in zip(tc_input, tc_results):
    print("QUERY: " + q + "\nRESULT: " + r + "\n\n")