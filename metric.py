from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from nltk.translate import meteor_score

rouge = Rouge()

def metric_count(title, references):
    #calculate metrics on one picture
    bleu = sentence_bleu(references, title) #BLEU
    meteor = meteor_score.meteor_score(references, title) #METEOR

    rouge_uni = 0
    rouge_bi = 0
    for ref in references:
      rouge_uni += rouge.get_scores(title, ref)[0]["rouge-1"]["f"]
      rouge_bi += rouge.get_scores(title, ref)[0]["rouge-2"]["f"]
    rouge_uni /= len(references) #F1 ROUGE UNIGRAMS
    rouge_bi /= len(references) #F1 ROUGE BIGRAMS

    return bleu, meteor, rouge_uni, rouge_bi