from django.http import JsonResponse
from django.shortcuts import render
from django.views.generic import TemplateView
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = 'cardiffnlp/twitter-roberta-base-sentiment'
model = AutoModelForSequenceClassification.from_pretrained(name)
tokenizer = AutoTokenizer.from_pretrained(name)
labels = ['negative', 'neutral', 'positive']


def analyze(request):
    input_ = request.GET.get('input')

    if not input_:
        probabilities = [0, 0, 0]
    else:
        encodings = tokenizer(
            input_,
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )
        logits = model(**encodings)
        probabilities = list(
            map(float, logits[0][0].detach().softmax(0).numpy()),
        )

    data = {
        label: probability for label, probability in zip(labels, probabilities)
    }

    return JsonResponse(data)


class MainView(TemplateView):
    template_name = 'main.html'
