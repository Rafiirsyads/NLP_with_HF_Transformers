<h1 align="center"> Natural Language Processing  with Hugging Face Transformers </h1>
<p align="center"> Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">

</div>

## Name : Rafi Irsyad Saharso

## My todo : 

### 1. Example 1 - Sentiment Analysis

```
# TODO :
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("This month, I developed important competencies in working on a project, both in completing my individual tasks and collaborating with the team.")
```

Result : 

```
[{'label': 'POSITIVE', 'score': 0.999505877494812}]
```

Analysis on example 1 : 

The sentiment analysis model accurately classifies the sentence as positive, with a very high confidence score of 0.9995. This indicates strong alignment between the model's understanding and the optimistic tone of the text. The sentence reflects personal growth, achievement, and teamwork—elements typically associated with positive sentiment.


### 2. Example 2 - Topic Classification

```
# TODO :
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "The patient was prescribed antibiotics to treat a bacterial infection and advised to rest for a few days.",
    candidate_labels=["medicine", "technology", "education"],
)
```

Result : 

```
{'sequence': 'The patient was prescribed antibiotics to treat a bacterial infection and advised to rest for a few days.',
 'labels': ['medicine', 'technology', 'education'],
 'scores': [0.9658596515655518, 0.02169395424425602, 0.012446344830095768]}
```

Analysis on example 2 : 

The zero-shot classification model correctly identifies "medicine" as the most relevant label, with a high confidence score of 0.9659. This indicates the model’s strong ability to associate the context prescribing antibiotics and treating infection with the medical domain.

### 3. Example 3 and 3.5 - Text Generator

```
# TODO
generator = pipeline('text-generation', model = 'gpt2')
generator("If you want to be successful, you need to", max_length = 30, num_return_sequences=1)
```

Result : 

```
[{'generated_text': "If you want to be successful, you need to know how to get it done.\n\nIt's a very simple exercise to find out what steps you can take to reach your goal. It's easy, but you probably won't want to do it.\n\n4. Give yourself permission.\n\nThere are a few things you may want to do when your goal is reached.\n\n1. Talk to someone.\n\nA lot of people think that they can see through your lies. But they can't.\n\nIt's important to talk to people who are looking for some direction to go.\n\nIn fact, if you want to see if your goal is reaching the right level of confidence, it might be best to talk to someone who knows something about you.\n\n2. Talk to others.\n\nTalk to other people. They're much more likely to want to reach your goal.\n\nAnd if you've talked to someone who doesn't know anything about you, you'll want to know where they are.\n\n3. Make friends.\n\nAsk others for help.\n\nSometimes they'll be interested in your goals, but they might not have the time or money to get to the next level.\n\nLet's say you're a writer.\n\nI"}]
```

Analysis on example 3 : 

The text-generation model produces a coherent and contextually relevant continuation of the prompt, framing it within a serious socio-political issue regarding nuclear weapons legislation. The output maintains a formal tone and logical flow, presenting concerns from an advocacy organization about the potential negative impacts of the legislation. While the generated text is detailed and focused, it exceeds the specified max_length parameter, indicating a possible issue with enforcing length constraints.

```
# TODO :
unmasker = pipeline("fill-mask", "distilroberta-base")
unmasker("He always <mask> his homework on time", top_k=4)
```

Result : 

```
[{'score': 0.40919846296310425,
  'token': 473,
  'token_str': ' does',
  'sequence': 'He always does his homework on time'},
 {'score': 0.2357645034790039,
  'token': 222,
  'token_str': ' did',
  'sequence': 'He always did his homework on time'},
 {'score': 0.1849287748336792,
  'token': 11630,
  'token_str': ' finishes',
  'sequence': 'He always finishes his homework on time'},
 {'score': 0.05892690643668175,
  'token': 25830,
  'token_str': ' completes',
  'sequence': 'He always completes his homework on time'}]
```

Analysis on example 3.5 : 

The fill-mask pipeline effectively predicts the missing word in the sentence by generating contextually appropriate verbs related to completing homework. The top prediction, "does," has the highest confidence score, fitting naturally and idiomatically within the sentence. The other suggestions—"did," "finishes," and "completes" also make sense.

### 4. Example 4 - Name Entity Recognition (NER)

```
# TODO
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("My name is Rafi. I am from Chicago, and I work at IBM.")
```

Result : 

```
[{'entity_group': 'PER',
  'score': np.float32(0.9988527),
  'word': 'Rafi',
  'start': 11,
  'end': 15},
 {'entity_group': 'LOC',
  'score': np.float32(0.9991744),
  'word': 'Chicago',
  'start': 27,
  'end': 34},
 {'entity_group': 'ORG',
  'score': np.float32(0.9988211),
  'word': 'IBM',
  'start': 50,
  'end': 53}]
```

Analysis on example 4 : 

The named entity recognizer accurately identifies key entities within the sentence, correctly labeling "Rafi" as a person (PER), "Chicago" as a location (LOC), and "IBM" as an organization (ORG). The grouped entities are clearly extracted with high confidence scores, indicating strong certainty in the model’s predictions.

### 5. Example 5 - Question Answering

```
# TODO
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "What is the capital city of France?"
context = "France is a country in Western Europe. Its capital city, Paris, is known for its art, fashion, and culture."
qa_model(question = question, context = context)
```

Result : 

```
{'score': 0.9941478371620178, 'start': 57, 'end': 62, 'answer': 'Paris'}
```

Analysis on example 5 : 

The question-answering model accurately identifies "Paris" as the answer to the question about the capital city of France. With a very high confidence score of 0.994, the model demonstrates strong comprehension of the context and the ability to pinpoint the precise answer span within the given text.

### 6. Example 6 - Text Summarization

```
# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer(
    """
A Data Analyst is a professional who collects, processes, and interprets data to help organizations make informed decisions.
They use statistical tools, programming languages like Python or R, and data visualization techniques to identify trends, patterns, and insights from structured and unstructured data.
Data Analysts play a crucial role in turning raw data into meaningful information that can improve business performance, optimize operations, and support strategic planning.
Their work often involves cleaning data, conducting exploratory data analysis (EDA), and presenting findings through dashboards or reports for stakeholders.
"""
)
```

Result : 

```
[{'summary_text': ' A Data Analyst is a professional who collects, processes, and interprets data to help organizations make informed decisions . They use statistical tools, programming languages like Python or R to identify trends, patterns, and insights from structured and unstructured data . Their work often involves cleaning data, conducting exploratory data analysis (EDA), and presenting findings through dashboards or reports for stakeholders .'}]

```

Analysis on example 6 :

The summarization pipeline successfully condenses the original paragraph while preserving the key concepts and structure. It highlights the main responsibilities of a Data Analyst, such as collecting, processing, and interpreting data, as well as using tools like Python or R to extract insights.

### 7. Example 7 - Translation

```
# TODO :
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Hari ini masak apa, chef?")
```

Result : 

```
[{'translation_text': "Qu'est-ce qu'on fait aujourd'hui, chef ?"}]

```

Analysis on example 7 :

The translation model provides a contextually accurate French translation of the Indonesian sentence. It effectively captures the intent and tone of the original. The output preserves both meaning and emotional nuance, demonstrating the model’s strength in handling informal and aspirational expressions.

---

## Analysis on this project

This project provides a hands-on introduction to a range of NLP tasks using Hugging Face pipelines. Each example is straightforward and highlights practical applications. The diverse selection of models illustrates the versatility of transformer-based approaches in addressing various language-related challenges.