# MT5_Language_Identifier

1. [ Introduction. ](#intro)<br />
     1.1 [ Text-to-Text Transfer Transformer (T5). ](#t5)<br />
     1.2 [ Multilingual T5. ](#mt5)
2. [ Fine-tuning MT5. ](#finetune)<br />
     2.1 [ Data preparation. ](#dp)<br />
     2.2 [ Encoding configuration. ](#ec)<br />

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

|    | Input_text                                                                                                                                          | True_target   | Predicted   |
|---:|:----------------------------------------------------------------------------------------------------------------------------------------------------|:--------------|:------------|
|  0 | <idf.lang> andaroni circle ki membership muft nahi hai.                                                                                             | ur            | hi          |
|  1 | <idf.lang> aisa lag raha tha k ye hmaisha k liye hai.                                                                                               | ur            | hi          |
|  2 | <idf.lang> Brock no defiende a Hillary.                                                                                                             | es            | en          |
|  3 | <idf.lang> Huevos means balls.                                                                                                                      | en            | th          |
|  4 | <idf.lang> До свидания!                                                                                                                             | ru            | bg          |
|  5 | <idf.lang> Той прегърна широко лорд Джулиан.                                                                                                        | bg            | ru          |
|  6 | <idf.lang> Mujhe aik glass chocolate ka doodh chahiye                                                                                               | ur            | hi          |
|  7 | <idf.lang> (a) Променете всяко d или t в целта до c.                                                                                                | bg            | ru          |
|  8 | <idf.lang> aur aisa hi tha jaise wo ise mustarad kar rahi thi, kuch tareqon se, jis trha se isne apke sath bartao kiya, dosre nawase.               | ur            | hi          |
|  9 | <idf.lang> Кальдас де Мончик окружен лесами.                                                                                                        | ru            | bg          |
| 10 | <idf.lang> Yahan pr buht zyada IT workers han.                                                                                                      | ur            | hi          |
| 11 | <idf.lang> Cambridge'den.                                                                                                                           | tr            | en          |
| 12 | <idf.lang> Mosam kafi acha hai or barish wala hai.                                                                                                  | ur            | hi          |
| 13 | <idf.lang> Koleksiyonda on iki makale var.                                                                                                          | tr            | vi          |
| 14 | <idf.lang> خدا حافظ!                                                                                                                                | ur            | ar          |
| 15 | <idf.lang> Aik cheez jo M. Tesnaires nai khaya nahi kya, woh, Anglo-Saxon ka input hah.                                                             | ur            | hi          |
| 16 | <idf.lang> Mera ek Credit Union hai, jiske paas mai jaa sakta hu                                                                                    | hi            | ur          |
| 17 | <idf.lang> FAA ki traffic control k data ka tajzeea kiya gya tha.                                                                                   | ur            | hi          |
| 18 | <idf.lang> sasty aur dhokay wali masnoa'at.                                                                                                         | ur            | sw          |
| 19 | <idf.lang> Биеннале Венеции перенаселено.                                                                                                           | ru            | bg          |
| 20 | <idf.lang> Curanderas heilen oft caida de mollera.                                                                                                  | de            | bg          |
| 21 | <idf.lang> Бих избрал  I'll Fly Away.                                                                                                               | bg            | ru          |
| 22 | <idf.lang> Беги и кричи.                                                                                                                            | ru            | bg          |
| 23 | <idf.lang> अलविदा!                                                                                                                                  | hi            | ur          |
| 24 | <idf.lang> Давай продолжим разговор.                                                                                                                | ru            | bg          |
| 25 | <idf.lang> James Wilson Marshall ne kuch khas nahi kiya.                                                                                            | ur            | hi          |
| 26 | <idf.lang> Je choisirai I'll Fly Away                                                                                                               | fr            | en          |
| 27 | <idf.lang> Sielewi hoja hio.                                                                                                                        | sw            | ur          |
| 28 | <idf.lang> Ek insaan ne sare jawano ko mansik santulan diya                                                                                         | hi            | ur          |
| 29 | <idf.lang> Meri dadi hamasha apne bachpan ke baray mein baat kerne ki lye inkaar kar deti thein.                                                    | ur            | hi          |
| 30 | <idf.lang> aur hum, Las Vegas, NV mai aik adress par move hogaye hian, jaisa ke humne Washington mai kiya tah.                                      | ur            | hi          |
| 31 | <idf.lang> si...si..sikuota.                                                                                                                        | sw            | ur          |
| 32 | <idf.lang> Αντίο!                                                                                                                                   | el            | ar          |
| 33 | <idf.lang> Ye aik bura tasur daita hai agr ap karkanon ko ye dekhain k unki awazein nahi suni ja rahin.                                             | ur            | hi          |
| 34 | <idf.lang> Congress insadad e dehshat gardi ki nigran hai.                                                                                          | ur            | hi          |
| 35 | <idf.lang> Eugene Debs era de Indiana.                                                                                                              | es            | en          |
| 36 | <idf.lang> jaisa aap jante hai, is samuh me sadasyata un mitra evam purane chhatron ki hai jo varshik tor par $1,000 kanun school ke liye dete hai. | hi            | ur          |
| 37 | <idf.lang> Имах проблеми с Linux.                                                                                                                   | bg            | ru          |
| 38 | <idf.lang> Talmudic in samaanome se ek bhi nahi le jata.                                                                                            | hi            | ur          |
| 39 | <idf.lang> Режан кристально чиста.                                                                                                                  | ru            | bg          |
| 40 | <idf.lang> Aise mera usse picha choota tha                                                                                                          | hi            | ur          |
| 41 | <idf.lang> City hall terminal se buht dour hai.                                                                                                     | ur            | hi          |         |
| 42 | <idf.lang> 拦截处理不当。                                                                                                                            | zh            | zhur          |
| 43 | <idf.lang> James Cook bereiste alle Hawaii-Inseln.                                                                                                  | de            | en          |
| 44 | <idf.lang> Personne ne prie.                                                                                                                        | fr            | de          |
| 45 | <idf.lang> Biosfer 1000 mil genişledi.                                                                                                              | tr            | sw          |
| 46 | <idf.lang> Wolverstone hakucheka.                                                                                                                   | sw            | en          |
| 47 | <idf.lang> No soñé.                                                                                                                                 | es            | fr          |
| 48 | <idf.lang> 'Biosphere' ilikua.                                                                                                                      | sw            | ar          |
| 49 | <idf.lang> Sag Biegung.                                                                                                                             | de            | ar          |
| 50 | <idf.lang> Design ka mutali’aa kamiyab nae tha.                                                                                                     | ur            | hi          |

<a name="finetune"></a>
## 2. Fine-tuning MT5

[MT5-small](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/mt5/small?pli=1) is fine-tuned on a new task of predicting the language a given text is written in, using the [XNLI](https://github.com/facebookresearch/XNLI) dataset, which contains text in 15 languages. The XNLI 15-way parallel corpus consists of 15 tab-separated columns, each corresponding to one language as indicated by the column headers. The column headers, each representing a language is given below,

ar: Arabic<br />
bg: Bulgarian<br />
de: German<br />
el: Greek<br />
en: English<br />
es: Spanish<br />
fr: French<br />
hi: Hindi<br />
ru: Russian<br />
sw: Swahili<br />
th: Thai<br />
tr: Turkish<br />
ur: Urdu<br />
vi: Vietnamese<br />
zh: Chinese (Simplified)<br />

These column headers are used as the target text during fine-tuning. MT5 models are supported by [Hugging Face transformers](https://huggingface.co/transformers/model_doc/mt5.html) package, and the details about model evaluation and fine-tuning can be found in the documentation.

<a name="dp"></a>
## 2.1 Data Preparation

The [xnli  dataset](dataset/xnli15.tsv) is cleaned and then prepared as a two-column data frame, with the column headers 'input_text' and 'target_text'. Since MT5 is a text-to-text model, to specify which task the model should perform, a prefix text is added to the original input sequence before feeding it. The prefix helps the model better when fine-tuning it on multiple downstream tasks, e.g., machine translation between many languages. The prefix <idf.lang> is added as a special token to the tokenizer. As stated in the [documentation](https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.resize_token_embeddings), if the new number of tokens is not equal to the model.config.vocab_size, then resize the input token embeddings matrix of the model. A few of the prepared training samples are shown below,

| input_text                                                                                                | target_text   |
|:----------------------------------------------------------------------------------------------------------|:--------------|
| <idf.lang> सांप नदी सांपों  से भरा है।                                                                         | hi            |
| <idf.lang> Anaokulu öğrencilerinin taklit yapma konusunda o kadar fazla yardıma ihtiyaçları yok.          | tr            |
| <idf.lang> Важно показать пределы данных, или люди сделают плохие выводы, которые уничтожат исследование. | ru            |
| <idf.lang> Музеят е в близост до египетския музей.                                                        | bg            |
| <idf.lang> O Mungu kwa sababu jina jina tu nimelisahau lakini ni Amani ya Bunge                           | sw            |

<a name="ec"></a>
## 2.1 Encoding Configuration

[T5 paper](https://arxiv.org/pdf/1910.10683.pdf) (source) : " *There are some extra parameters in the decoder due to the encoder-decoder attention and there are also some computational costs in the attention layers that are* ***quadratic in the sequence lengths*** "

Since the input and target token id lengths are task-specific, the distribution of the token id lengths of the dataset needs to be first analyzed.

<p float="left">
  <img src="dataset/token_length_hist/ip_tokens_len.jpg" width="400" />
  <img src="dataset/token_length_hist/op_tokens_len.jpg" width="400" /> 
</p>

![alt text](dataset/token_length_hist/ip_tokens_len.jpg)  ![alt text](dataset/token_length_hist/op_tokens_len.jpg)

- The maximum input sequence length is set to 40.
- The maximum target sequence length is set to 3.
- truncation=True truncates the sequence to a maximum length specified by the max_length argument.
- padding='max_length' pads the sequence to a length specified by the max_length argument.
