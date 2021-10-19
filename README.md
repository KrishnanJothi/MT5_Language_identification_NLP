# MT5_Language_identification_NLP

1. [ Introduction. ](#intro)<br />
     1.1 [ Text-to-Text Transfer Transformer (T5). ](#t5)<br />
     1.2 [ Multilingual T5. ](#mt5)
3. [ Usage tips. ](#usage)

<a name="intro"></a>
## 1. Introduction

<a name="t5"></a>
## 1.1 Text-to-Text Transfer Transformer (T5)

[T5](https://arxiv.org/pdf/1910.10683.pdf) is a pre-trained language model whose primary distinction is its use of a unified “text-to-text” format for all text-based NLP problems. 

This approach is natural for generative tasks where the task format requires the model to generate text conditioned on some input. It is more unusual for classification tasks, where T5 is trained to generate the literal text of the class label instead of a class index. The primary advantage of this approach is that it allows the use a single set of hyperparameters for effective fine-tuning on any downstream task.

T5 uses a basic [encoder-decoder Transformer](https://arxiv.org/pdf/1706.03762.pdf) architecture. T5 is pre-trained on C4 Common Crawl dataset using BERT-style masked language modeling “span-corruption” objective, where consecutive spans of input tokens are replaced with a mask token and the model is trained to reconstruct the masked-out tokens.

The authors trained 5 different size variants of T5: small model, base model, large model, and models with 3 billion and 11 billion parameters.

<a name="mt5"></a>
## 1.2 Multilingual T5

[MT5](https://arxiv.org/pdf/2010.11934.pdf) is a multilingual variant of T5 that was pre-trained on a new Common Crawl-based mC4 dataset covering 101 languages. MT5 pre-training uses suitable data sampling strategies to boost lower-resource languages, and to avoid over and under fitting of the model. Similar to T5, MT5 casts all the tasks into the text-to-text format.

Similar to T5, the authors trained 5 different size variants of MT5: small model, base model, large model, XL, and XXL model. The increase in parameter counts compared to the corresponding T5 model variants comes from the larger vocabulary used in mT5.

<a name="usage"></a>
## 2. Usage tips

sometext
