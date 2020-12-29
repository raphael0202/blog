---
title: "How many layers of my BERT model should I freeze? ❄️"
date: 2020-12-04T15:56:37+01:00
draft: false
categories:
- experiment
tags:
- bert
- transformer
- freezing
---
Since the advent of the Transformer architecture (Vaswani et al. 2017) and of BERT models (Devlin et al. 2019), Transformer models have become ubiquitous in NLP, achieving SOTA results on most NLP datasets.

Before Sesame Street puppets flooded on ArXiv, the de-facto method to train an NLP model leveraged word embeddings pre-trained using Glove or word2vec. These word embeddings were used to initialize the first embedding layer of your model, and you just had to plug the rest of your architecture above this first layer.

Nowadays, it's increasingly common to use BERT-like language models trained on vast amounts of unlabeled text, and fine-tune the model on task-specific data. This change of paradigm led to a new subfield in NLP called BERTology which studies the ins and outs of BERT-like models.

Many hyper-parameters of BERT — such as the number of attention heads or the learning rate to use — have been studied, but few papers looked at how layer freezing affects performances.  Freezing layers means disabling gradient computation and backpropagation for the weights of these layers. This is a common technique in NLP or Computer Vision when working on small datasets as it usually reduces the chances of your model to overfit. Usually, models are frozen from bottom to top, as the first layers of neural networks are thought to encode universal features while the top layers are more task-specific. Another advantage of freezing is that it provides a nice speed boost and reduced memory usage during training.

Most machine learning teams in industry don't have access to high-quality and domain-specific datasets. As the costs of building custom datasets are high, these datasets are usually way smaller than those often used in academia. Learning how to train better models with fewer data is a major concern, especially in industry. While the effect of freezing on performance has been explored (Lee, Tang, and Lin 2019), the low-resource setting hasn't attracted much attention yet. Luckily, it's what this post is all about 😎

# 📝 Methods

To test how freezing affects performances when data is scarce, we will use the now popular [🤗 Transformers library](https://github.com/huggingface/transformers) from Hugging Face to train a RoBERTa model (base version) on two standard binary classification datasets from the GLUE benchmark: SST-2 (Stanford Sentiment Treebank) and CoLA (Corpus of Linguistic Acceptability).

SST-2 is a sentiment analysis task and CoLA a dataset containing sentences labeled with respect to their grammatical acceptability (i.e whether each sentence is grammatically correct or not). SST-2 and CoLA contain ~51% and ~69% of positive samples respectively in their validation set, which gives us a majority class baseline to compare to. We will use accuracy as the metric of evaluation for both datasets.

We will train models with a subset of the training dataset and evaluate on the whole validation dataset. For CoLA and SST-2, we generate subsets of size 50, 100, 200, 500, and 1000, by:

- taking care that the fraction of positive samples is the same as in the original train dataset (stratify splitting)
- generating the splits by increasing size (50, 100, 200, 500, 1000) and reusing all samples from the previous split. By doing so, we limit the variability coming from using different samples in each split.

To assess the impact of layer freezing on performances, 0, 2, 4, 8, 10, or 12 encoder layers were frozen. I also tested to only freeze the embedding layers (frozen layers = -1).

Similar to the original RoBERTa paper (Liu et al. 2019), I fine-tuned for 10 epochs with a linear warmup of 1 epoch followed by a linear decay to zero. As my computational budget was limited, I didn't perform any hyper-parameter search and stuck to the default values (`lr = 5e-5`, AdamW optimizer).

On both datasets, for each layer freezing value (-1, 0, 2, 4, 8, 10, 12) and for each train size (50, 100, 200, 500, 1000), 5 replicates were trained with a different random seed.

I decided to use the recent (and pretty awesome!) [🤗Datasets](https://github.com/huggingface/datasets) library — also developed by folks from Hugging Face — to easily fetch and process these two datasets:

```python
DATASET_MAP = {"sst2": ("glue", "sst2"), "cola": ("glue", "cola")}

def get_dataset(tokenizer, dataset_name: str, split: str, split_path: Optional[Path] = None):
    ds = datasets.load_dataset(*DATASET_MAP[dataset_name], split=split)
    ds = ds.shuffle(seed=42)

    if split_path is not None:
	      # split_path is a npy file containing indexes of samples to keep
        split_ids = set(np.load(split_path).tolist())
        ds = ds.filter(lambda idx: idx in split_ids, input_columns="idx")

    ds = ds.map(lambda e: tokenizer(e["sentence"], padding=False, truncation=True), batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds
```

The `get_dataset` function returns a `Dataset` object that integrates seamlessly with the Transformers library. The sentences are tokenized (and cached) on the fly using `Dataset.map`.

The training function is pretty simple, thanks to the use of the `Trainer` class that takes care of most of the boilerplate code for us: 

```python
def get_random_seed():
    return int.from_bytes(os.urandom(4), "big")

def train(
    output_dir: str,
    dataset_name: str,
    train_size: Optional[int] = None,
    freeze_layer_count: int = 0,
):
    # get_split_path returns the path of a npy file containing sample indexes
    # to include in the training set
    split_path = get_split_path(dataset_name, train_size) if train_size is not None else None
    args_dict = {
        "evaluation_strategy": "steps",
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "learning_rate": 5e-5,
        "num_train_epochs": 10,
        "logging_first_step": True,
        "save_total_limit": 1,
        "fp16": True,
        "dataloader_num_workers": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        # we need to generate a random seed manually, as otherwise 
        # the same constant random seed is used during training for each run
        "seed": get_random_seed(),
    }
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", return_dict=True)

    if freeze_layer_count:
	      # We freeze here the embeddings of the model
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = False

        if freeze_layer_count != -1:
	          # if freeze_layer_count == -1, we only freeze the embedding layer
	          # otherwise we freeze the first `freeze_layer_count` encoder layers
            for layer in model.roberta.encoder.layer[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_ds = get_dataset(tokenizer, dataset_name, split="train", split_path=split_path)
    val_ds = get_dataset(tokenizer, dataset_name, split="validation")

    epoch_steps = len(train_ds) / args_dict["per_device_train_batch_size"]
    args_dict["warmup_steps"] = math.ceil(epoch_steps)  # 1 epoch
    args_dict["logging_steps"] = max(1, math.ceil(epoch_steps * 0.5))  # 0.5 epoch
    args_dict["save_steps"] = args_dict["logging_steps"]

    training_args = TrainingArguments(output_dir=str(output_dir), **args_dict)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.train()
```

The complete code is also [available as a gist](https://gist.github.com/raphael0202/85580b29b27a27ddaae8d393f686f891).

# 🔎 Results

Here comes the most interesting part!

Below is displayed for both datasets the best accuracy obtained during training on the development set for all train sizes and all numbers of frozen layers. Each red dot on the plot represents a single run.

The complete results can also be found as tables at the end of the post.

![Results on SST-2](/img/freezing-bert/sst2.png)

_SST-2: Best accuracy during training on development set_

![Results on CoLA](/img/freezing-bert/cola.png)

_CoLA: Best accuracy during training on development set_

As expected, having more data always help: we gain +0.15 (0.75 → 0.90) in accuracy by increasing dataset size from 50 to 1000 for SST-2.

Freezing all (12) layers of the encoder is definitely not a good idea when dealing with limited data: there is always a significant performance drop, and the resulting models are barely better than a majority class classifier. Freezing 10 layers is also associated with degraded accuracy, even if the effect isn't as strong. Aside from 10- and 12-frozen layers, there are no clear impact of layer freezing on performances.

We also observe high variability in accuracy between runs with the same settings (but with different random seeds). This instability is known since the release of BERT. While catastrophic forgetting and the small sizes of datasets were first suspected as the causes of this instability, more recent work (Mosbach et al. 2020) suggests that optimization difficulties leading to vanishing gradients are the actual reasons. For both SST-2 and CoLA datasets, there are no clear evidence that freezing reduces these instabilities.

As a final note, the performance gap between SST-2 and CoLA with 50 examples is interesting: while best models for SST-2 classify 82% of samples correctly (with 51% of positive samples), best run are barely better than the majority class baseline with 70% of samples correctly classified. This is indicative of the inductive bias of these models, which are more prone to detect polarity than grammatical correctness.

# Conclusion

And that's all for this second post 😊!

The results of these experiments were surprising to me, I initially thought layer freezing would have a more visible impact on performances.

I hope you've enjoyed the post, feel free to share your feedbacks or your experience with layer freezing (my email address is available on the blog).

# Bibliography

Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. “BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding.” *ArXiv:1810.04805 [Cs]*, May. [http://arxiv.org/abs/1810.04805](http://arxiv.org/abs/1810.04805).

Lee, Jaejun, Raphael Tang, and Jimmy Lin. 2019. “What Would Elsa Do? Freezing Layers During Transformer Fine-Tuning.” *ArXiv:1911.03090 [Cs]*, November. [http://arxiv.org/abs/1911.03090](http://arxiv.org/abs/1911.03090).

Liu, Yinhan, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. “RoBERTa: A Robustly Optimized BERT Pretraining Approach.” *ArXiv:1907.11692 [Cs]*, July. [http://arxiv.org/abs/1907.11692](http://arxiv.org/abs/1907.11692).

Mosbach, Marius, Maksym Andriushchenko, and Dietrich Klakow. 2020. “On the Stability of Fine-Tuning BERT: Misconceptions, Explanations, and Strong Baselines.” *ArXiv:2006.04884 [Cs, Stat]*, October. [http://arxiv.org/abs/2006.04884](http://arxiv.org/abs/2006.04884).

Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. “Attention Is All You Need.” *ArXiv:1706.03762 [Cs]*, December. [http://arxiv.org/abs/1706.03762](http://arxiv.org/abs/1706.03762).

# Appendix

### SST-2

|train size  |frozen layers  |mean      |50%       |std        |min       |max       |
|------------|---------------|----------|----------|-----------|----------|----------|
|50          |-1             |0.612     |0.610     |0.0939     |0.490     |0.743     |
|            |0              |0.705     |0.752     |0.0847     |**0.600** |0.779     |
|            |2              |0.695     |0.709     |0.0713     |0.587     |0.775     |
|            |4              |**0.711** |**0.753** |0.1024     |0.557     |**0.823** |
|            |8              |0.529     |0.511     |0.0280     |0.509     |0.569     |
|            |10             |0.523     |0.509     |0.0276     |0.509     |0.572     |
|            |12             |0.514     |0.509     |**0.0112** |0.509     |0.534     |
|100         |-1             |0.861     |0.870     |0.0243     |0.818     |0.878     |
|            |0              |0.866     |0.870     |0.0177     |0.845     |0.884     |
|            |2              |**0.873** |**0.885** |0.0244     |0.832     |**0.893** |
|            |4              |0.866     |0.864     |**0.0118** |**0.855** |0.883     |
|            |8              |0.855     |0.866     |0.0279     |0.806     |0.873     |
|            |10             |0.673     |0.685     |0.0968     |0.509     |0.758     |
|            |12             |0.514     |0.509     |**0.0118** |0.509     |0.535     |
|200         |-1             |0.876     |0.876     |0.0094     |0.862     |0.888     |
|            |0              |0.875     |0.875     |0.0115     |0.863     |0.893     |
|            |2              |0.871     |0.877     |0.0203     |0.836     |0.885     |
|            |4              |0.884     |0.883     |0.0057     |**0.879** |0.894     |
|            |8              |**0.887** |**0.885** |0.0085     |**0.879** |**0.901** |
|            |10             |0.827     |0.827     |**0.0041** |0.821     |0.831     |
|            |12             |0.516     |0.509     |0.0169     |0.509     |0.547     |
|500         |-1             |0.896     |0.895     |0.0094     |0.886     |0.910     |
|            |0              |0.897     |0.900     |0.0106     |0.881     |0.910     |
|            |2              |**0.908** |**0.908** |0.0042     |**0.902** |**0.913** |
|            |4              |0.904     |0.903     |0.0033     |0.900     |0.909     |
|            |8              |0.905     |0.905     |0.0035     |0.901     |0.909     |
|            |10             |0.882     |0.884     |0.0073     |0.870     |0.889     |
|            |12             |0.509     |0.509     |**0.0000** |0.509     |0.509     |
|1000        |-1             |0.905     |0.905     |0.0015     |0.903     |0.907     |
|            |0              |0.905     |0.902     |0.0063     |0.899     |0.912     |
|            |2              |0.911     |0.911     |0.0063     |0.902     |**0.917** |
|            |4              |**0.913** |**0.912** |0.0022     |**0.910** |0.916     |
|            |8              |0.910     |0.907     |0.0044     |0.907     |0.915     |
|            |10             |0.891     |0.894     |0.0045     |0.886     |0.895     |
|            |12             |0.509     |0.509     |**0.0000** |0.509     |0.509     |


### CoLA

|train size  |frozen layers  |mean      |50%       |std        |min       |max       |
|------------|---------------|----------|----------|-----------|----------|----------|
|50          |-1             |0.694     |**0.694** |0.0047     |0.691     |0.702     |
|50          |0              |**0.695** |0.693     |0.0047     |0.691     |0.700     |
|50          |2              |**0.695** |0.693     |0.0057     |0.691     |**0.704** |
|50          |4              |0.691     |0.691     |0.0009     |0.691     |0.693     |
|50          |8              |0.691     |0.691     |**0.0000** |0.691     |0.691     |
|50          |10             |0.691     |0.691     |**0.0000** |0.691     |0.691     |
|50          |12             |0.691     |0.691     |**0.0000** |0.691     |0.691     |
|100         |-1             |0.720     |0.722     |0.0191     |0.698     |0.745     |
|100         |0              |**0.724** |0.723     |0.0234     |0.692     |0.757     |
|100         |2              |0.722     |0.728     |0.0123     |0.705     |0.735     |
|100         |4              |0.731     |0.727     |0.0099     |**0.722** |0.744     |
|100         |8              |0.732     |**0.736** |0.0221     |0.708     |**0.759** |
|100         |10             |0.691     |0.691     |**0.0000** |0.691     |0.691     |
|100         |12             |0.691     |0.691     |**0.0000** |0.691     |0.691     |
|200         |-1             |0.765     |0.764     |0.0112     |0.754     |0.779     |
|200         |0              |0.764     |0.761     |0.0078     |0.757     |0.777     |
|200         |2              |0.762     |0.761     |0.0077     |0.751     |0.772     |
|200         |4              |0.774     |0.773     |0.0049     |0.768     |0.781     |
|200         |8              |**0.782** |**0.784** |0.0058     |**0.773** |**0.789** |
|200         |10             |0.721     |0.719     |0.0194     |0.699     |0.744     |
|200         |12             |0.691     |0.691     |**0.0000** |0.691     |0.691     |
|500         |-1             |0.768     |0.786     |0.0431     |0.691     |0.791     |
|500         |0              |0.761     |0.779     |0.0397     |0.691     |0.784     |
|500         |2              |0.781     |0.787     |0.0187     |0.748     |0.796     |
|500         |4              |0.791     |0.791     |0.0053     |0.785     |0.796     |
|500         |8              |**0.792** |**0.793** |0.0049     |**0.787** |**0.797** |
|500         |10             |0.784     |0.786     |0.0062     |0.775     |0.792     |
|500         |12             |0.691     |0.691     |**0.0000** |0.691     |0.691     |
|1000        |-1             |0.774     |0.775     |0.0218     |0.739     |0.796     |
|1000        |0              |0.781     |0.785     |0.0186     |0.749     |0.796     |
|1000        |2              |0.794     |**0.797** |0.0056     |0.788     |0.798     |
|1000        |4              |0.795     |0.794     |0.0050     |0.790     |**0.802** |
|1000        |8              |0.795     |0.794     |0.0050     |0.790     |**0.802** |
|1000        |10             |**0.797** |**0.797** |0.0034     |**0.792** |0.801     |
|1000        |12             |0.691     |0.691     |**0.0000** |0.691     |0.691     |
