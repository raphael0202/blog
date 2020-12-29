---
title: "Prodigy, a must-have in the Data Scientist toolbox"
date: 2020-11-07T17:41:06+01:00
draft: false
categories:
- tool
tags:
- tool
- annotation
- data
---
If you sometimes find yourself annotating data for machine learning projects and you've never heard of [Prodigy](https://prodi.gy/), it's definitely a tool you would be interested in.

The project was initiated by the makers of spaCy - the well-known Python NLP library - after they realized that while supervised learning works well, [data collection was broken](https://explosion.ai/blog/supervised-learning-data-collection). To this day, many data collection projects still rely on Amazon Mechanical Turk, a crowdsourcing platform with low wages, questionable UX, and low incentives for quality. Given the major impact of data quality on supervised learning model performances, data collection is too important to be outsourced on Mechanical Turk. Whenever possible, annotations should be done in-house and at least partially by the scientist in charge of the research project. The latter ensures that any quality or annotation issues that could impact the training is noticed beforehand.

Prodigy aims at tackling these issues, with a modern annotation tool that allows generating datasets quickly and efficiently. The annotation tasks mainly cover NLP (text classification, named entity recognition, syntactic dependencies), but tasks in other ML fields were added progressively (object detection, image classification).

The UI and UX of the tool were carefully thought to make the annotation as fast as possible. Binary decisions (yes/no questions) were favored in the interface to speed up annotation and to avoid the mental overhead of a context switch. With keyboard shortcuts, we can annotate without touching the mouse, which also provides a speed boost.

Another really useful feature is the annotation history that lists previous annotations. When annotating quickly, annotation mistakes always occur (often just after the keystroke), and checking the last annotations is important to fix them. Undoing a previous annotation is really simple and can be done with a keyboard shortcut.

![Prodigy interface](/img/prodigy_interface.png)
*Binary classification task with 4 possible actions: accept, reject, ignore or undo the last annotation*

Another interesting aspect of Prodigy is that it's not a SaaS service but a Python framework that is launched locally through a CLI. This enables a great deal of configurability through [custom recipes](https://prodi.gy/docs/recipes) and the annotation of a private dataset that cannot be shared with a third-party.

The downside of such an approach is that it makes collaboration annotation harder: there is no way to split a dataset between several annotators or to compute inter-annotator agreement metrics. [Prodigy Scale](https://support.prodi.gy/t/prodigy-annotation-manager-update-prodigy-scale-prodigy-teams/805) may be an answer to that, but there isn't much info available on the current state of the project.

The annotation tool features an active learning mode that theoretically enables fewer annotations by using a *model in the loop.* The idea behind active learning is that the next examples to annotate are chosen by a model trained on the data already annotated. Usually, examples that the model is the most uncertain about are selected. By choosing the most informative examples to annotate, active learning aims at getting the same supervised learning performance with fewer examples. This is especially interesting for imbalanced classification problems, where samples from the majority class are much less useful than ones from the minority class.

In my experiments on imbalanced text classification problems in low-resource settings, active learning through Prodigy didn't deliver its promises. It was slightly better than the classic annotation strategy but beaten significantly by a strategy of oversampling minority class using pattern matching. Besides, we should be careful not to use active learning when generating a test set, as it biases the distribution: the classifier performance on the test set is no longer representative of its performance on the original distribution.

To me, another small caveat of Prodigy is that annotations are saved on a local SQLite DB. An export is required to get a JSONL file containing all annotations. I would appreciate the possibility of directly exporting the annotation in JSONL, as I rarely reuse previous annotations sessions.

Overall, Prodigy is an extremely useful tool in data science, sold as a lifetime license at 390$ (included with 12 months of upgrades). The price is a bit high for students and machine learning hobbyists, but it's worth it for ML teams.