---
title: "Never Trust Your Dataset"
date: 2021-05-10T12:23:40+02:00
draft: false
categories:
- article-analysis
tags:
- dataset
- annotation
- data
---
Datasets are ubiquitous in machine learning. There is literally nothing to learn without -labeled or unlabeled- datasets. Lack of datasets has impeded the progress in NLP for low-resource languages: most of the academic work in NLP focus on English and to a lesser extent to a couple of high-resource languages (Spanish, German, Japanese, French,...).

Recently, a diverse team of NLP researchers studied the quality of web-crawled corpora that are behind most of the progress in NLP in the last few years (Caswell et al. 2021). More specifically, they studied 3 parallel corpora used for machine translation (CCAligned, ParaCrawl, WikiMatrix) and two monolingual corpora (OSCAR and mC4) used to train language-specific language models.

To assess the quality of these corpora, they sampled a small fraction (~100 samples) of each dataset for a diverse set of languages and annotated each sample given the following taxonomy: correct, single word or phrases, boilerplate content, incorrect translation, wrong language, non-linguistic content, offensive and pornographic. They focused on low- or mid-resource languages but included high-resource languages as well for comparison. Their study reveals that all studied datasets had quality issues, especially for low-resource languages.

Most severe problems were found for CCAligned and WikiMatrix, with 44 of the 65 audited languages for CCAligned containing under 50% correct sentences, and 19 of the 20 in WikiMatrix. Worse, across all datasets, for 15/205 languages, there was not a single correct sentence among the samples. To a lesser extent, the authors also discovered quality problems for monolingual corpora: there were 11/51 and 9/48 languages with under 50% correct sentences for OSCAR and mC4 respectively.

The origin of some of these issues is discussed in the paper, and includes incorrect language identification or can be attributed to the automatic alignment method used when building parallel corpora. However, low-resource languages are disproportionately affected by these quality problems. For example, samples from mC4 of the 8 largest languages showed near-perfect quality. In these corpora, likely, the data of low-resource languages were not manually inspected at all, leading to quality issues that could have been spotted even by non-native speakers, as showed by Caswell et al.

Of course, data quality issues are not exclusive to web-crawled corpora. Recently, one of my coworkers ([who also happens to blog about NLP and machine learning!](http://keramitas.io/)) was working on a new locality detection pipeline that leverages a NER model. To detect localities, he used the WikiNER dataset (Nothman et al. 2013), as it is one of the only (free) resources available in French to train a NER model.

When he dived into the dataset, he noticed some weird entity spans (only relevant spans are showed here):

> <span style="background-color: #FFFF00">Les<sub>LOC</sub></span> frontières terrestres sud-africaines atteignent 4750 km de long (Botswana 1840 km ; Lesotho 909 km ; Namibie 855 km ; Mozambique 491 km ; Swaziland 430 km ; Zimbabwe 225 km).

Here, "Les" is not a locality but a determiner.

> <span style="background-color: #FFFF00">Si<sub>MISC</sub></span> bien que l'année suivante, elle mit sa priorité dans les initiatives régionales telles que le Mercosur ou la Banque du Sud après une décennie de partenariat avec les États-Unis.

"Si" is not a MISC entity, it means "if" in French.

> L' Argentine est membre permanent du Mercosur (communauté économique des pays de <span style="background-color: #FFFF00">l'Amérique<sub>LOC</sub></span> <span style="background-color: #FFFF00">du Sud<sub>LOC</sub></span>) avec le Brésil, le Paraguay, l' Uruguay et le Venezuela ; cinq autres pays y sont associés : la Bolivie, le Chili, le Pérou, la Colombie et l'Équateur.

"l'Amérique" and "du Sud" should not be splitted in two entities: "l'Amérique du Sud"

> Il prend l'initiative de plusieurs opérations diplomatiques en faveur de <span style="background-color: #FFFF00">la France<sub>LOC</sub></span>, qui auraient, d'après lui, changé le cours de la guerre.

The determiner "la" shouldn't be part of the entity span in "la France"

The presence of errors in the WikiNER dataset shouldn't come as a surprise, as it is introduced by the authors as a silver-standard dataset, i.e. of lower quality than a dataset that was fully annotated manually. However, a check over the French dataset revealed that more than 7% of all samples were affected by the identified issues, which suggests that fixing these errors could boost the entity detection performance of the downstream NER model.
As it would have been too time-consuming to manually check and correct the ~160k sentences of the dataset, all splits of the dataset (train, dev, test) were cleaned using a fully automated correction pipeline that leverages hand-crafted rules. Most of the corrections focused on the LOC entity type, as it was the entity type we were interested in.

The impact of these dataset corrections on metrics was impressive: F1 scores for LOC entities went from 89.03 to 93.79 (+4.7). However, the score improvements were not significant for the other entity types. The impact of fixing the dataset widely exceeded the gain we observed by tuning hyperparameters or the model architecture.

As illustrated by WikiNER, it is most of the time more effective to spend time improving your dataset before trying fancier neural architecture or performing a hyperparameter search.

As a rule of thumb, you should always check the dataset you plan to use to train a machine learning model. It can be as simple as diving into the data to get a better grasp of the task and to detect potential issues. Depending on your dataset, a more advanced exploratory data analysis may be needed. Here are a few things you may want to check:

- incorrect or incoherent annotations. Errors always slip through, but this is especially important in a multi-annotator setting, as annotation guidelines can be interpreted differently by annotators.
- corrupted images/texts/data.
- wrong data. Sometimes the data you're using is simply not what you expected.
- label distribution, to detect unbalanced dataset

Finally, once your model is trained, it's a good idea to have a look at the errors of your models (when the predictions differ from the targets). Besides allowing you to have a better understanding of when your model fails, it often reveals data quality issues that were not spotted before. 


# Bibliography

Caswell, Isaac, Julia Kreutzer, Lisa Wang, Ahsan Wahab, Daan van Esch, Nasanbayar Ulzii-Orshikh, Allahsera Tapo, et al. 2021. “Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets.” *ArXiv:2103.12028 [Cs]*, March. [http://arxiv.org/abs/2103.12028](http://arxiv.org/abs/2103.12028).

Nothman, Joel, Nicky Ringland, Will Radford, Tara Murphy, and James R. Curran. 2013. “Learning Multilingual Named Entity Recognition from Wikipedia.” *Artificial Intelligence* 194 (January): 151–75. [https://doi.org/10.1016/j.artint.2012.03.006](https://doi.org/10.1016/j.artint.2012.03.006).
