Deep Learning for Text with PyTorch
===================================


Chapter 1: (Overview of NLP) NLP with RNNs in PyTorch

Lesson 1.1: Use Cases for Natural Language Processing (NLP)
- Learning Objectives:
  * Learner will be able to classify problems in terms of the different NLP use cases, e.g. Text Classification, Text Generation, Translation, Question Answering, Summarisation
  * Learner will be able to identify which deep learning techniques are appropriate for each NLP use case, e.g. transformers, RNNs, LSTMs, bag of words, embeddings
  * Learner will be able to describe the importance of train/validation/test split for NLP use cases

Lesson 1.2: Preparing Text/Word Data
- Learning Objectives:
  * Learner will understand the difference between character level representations, tokenising, bags of words and embeddings
  * Learner will be able to preprocess text at the character level (`torch.tensor`)
  * Learner will be able to generate a bag of words representation of a text (`torch.view`, `torch.LongTensor`)

Lesson 1.3: Word/Text Classification with a PyTorch RNN
- Learning Objectives:
  * Learner will be able to create and train an RNN in PyTorch that handles text data (extending `torch.nn.Module`, using ` nn.NLLLoss`, `torch.nn.LogSoftMax`, `torch.nn.Linear`)
  * Learner will understand how to initialise the recurrent hidden layer in an RNN
  * Learner will be able to evaluate the results of an RNN trained on text data (`matplotlib.pyplot.matshow`)
  * Learner will be able to use an RNN to make predictions on unseen text data (`torch.no_grad`, `torch.tensor.item`, `torch.topk`)
  * Learner will be able to save and load a PyTorch RNN (`torch.save`, `torch.load`, `torch.nn.Module.state_dict`, `torch.nn.Module.load_state_dict`)


Chapter 2: Text Classification with TorchText

Lesson 2.1: Accessing Large Datasets
- Learning Objectives:
  * Learner will be able to access Torchtext datasets (`torchtext.datasets`)
  * Learner will be able to split Torchtext datasets into train and test (`torchtext.datasets`)
  * Learner will be able to select appropriate Torchtext datasets for different NLP tasks (e.g. `torchtext.datasets.AG_NEWS`, `torchtext.datasets.IMDB` etc.)

Lesson 2.2: Creating a Text Classification model
- Learning Objectives:
  * Learner will be able to set up an appropriate text preprocessing pipeline (`torchtext.data.utils.get_tokenizer`, `torchtext.vocab.build_vocab_from_iterator`, `torch.utils.data.DataLoader`)
  * Learner will understand how to use Embeddings and EmbeddingBags in a PyTorch neural net (NN) (``torch.nn.Embedding`, `torch.nn.EmbeddingBag`)
  * Learner will be able to define an appropriate model for a embedded text classification model (extending `torch.nn.Module`)

Lesson 2.3: Training a Text Classification model
- Learning Objectives:
  * Learner will be able to define the functions necessary to train an embedded text classification model (`torch.nn.Module.train`)
  * Learner will be able to split the dataset and train a embedded text classification model (` torch.utils.data.dataset.random_split`)
  * Learner will understand the pros and cons of different loss functions and optimisation procedures (`torch.nn.CrossEntropyLoss`, `torch.optim.SGD`)
  * Learner will be able to save and load an embedded model (`torch.save`, `torch.load`, `torch.nn.Module.state_dict`, `torch.nn.Module.load_state_dict`)

Lesson 2.4: Evaluating a Text Classification model
- Learning Objectives:
  * Learner will be able to evaluate a text classification model accuracy (`torch.nn.Module.eval`, `torch.no_grad`)
  * Learner will understand the challenges of underfitting and overfitting a text classification model
  * Learner will be able to use the trained model to classify previously unseen data (`torch.nn.Tensor.argmax`, `torch.no_grad`)



Chapter 3: Building your own transformer in PyTorch for Text Generation

Lesson 3.1: Understanding Transformers
- Learning Objectives:
  * Learner will be able to describe the overall architecture of a transformer
  * Learner will understand attention mechanisms in transformers
  * Learner will be able to describe the role of positional encoding
  * Learner will understand the parallelisation benefits of transformers vs RNNs

Lesson 3.2: Creating a Transformer
- Learning Objectives:
  * Learner will be able to set up an appropriate text preprocessing pipeline for a Transformer, including batching (`torchtext.data.utils.get_tokenizer`, `torchtext.vocab.build_vocab_from_iterator`, `torch.utils.data.DataLoader`)
  * Learner will understand how to use TransformerEncoders and TransformerEncoderLayers (`torch.nn.TransformerEncoder`, `torch.nn.TransformerEncoderLayer`)
  * Learner will be able to define a transformer model in PyTorch for text generation (`torch.nn.Module`, `torch.nn.Linear`, `torch.nn.Dropout`, `torch.triu`)

Lesson 3.3: Training a Transformer
- Learning Objectives:
  * Learner will be able to define the functions necessary to train a transformer (`torch.optim.lr_scheduler.StepLR`, `torch.nn.utils.clip_grad_norm_`)
  * Learner will understand the range of hyperparameters for a transformer model
  * Learner will understand the pros and cons of different loss functions and optimisation procedures (`torch.nn.CrossEntropyLoss`, `torch.optim.SGD`)
  * Learner will be able to save and load a transformer model (`torch.save`, `torch.load`, `torch.nn.Module.state_dict`, `torch.nn.Module.load_state_dict`)

Lesson 3.4: Evaluating a Transformer
- Learning Objectives:
  * Learner will be able to evaluate a text generation transformer in terms of evolving loss values
  * Learner will be able to use the trained model to predict subsequent text for previously unseen sequences (`torch.softmax`, `torch.max`, `torchtext.vocab.lookup_token`)
  * Learner will understand how to adapt a text generation transformer to other NLP tasks



Chapter 4: Using Pretrained transformers from HuggingFace

Lesson 4.1: Accessing Datasets via HuggingFace
- Learning Objectives:
  * Learner will be able to access datasets via HuggingFace (`datasets.load_dataset`)
  * Learner will be able to select appropriate datasets from HuggingFace for particular NLP tasks
  * Learner will be able to split datasets into train and test using the HuggingFace Dataset module (`datasets.load_dataset.train_test_split`)

Lesson 4.2: Accessing Large Language Models (LLMs) via HuggingFace
- Learning Objectives:
  * Learner will be able to access multiple LLMs and tokenizers via the HuggingFace transformers module (`transformers`, `transformers.AutoTokenizer`, `transformers.AutoTokenizer.from_pretrained`)
  * Learner will be able to use an LLM on multiple NLP tasks to make predictions on unseen text data using pipelines (`transformers.pipeline`)
  * Learner will be able to evaluate LLM performance on multiple NLP tasks (`evaluate`, `evaluate.load`)

Lesson 4.3: Tuning Large Language Models (LLMs)
- Learning Objectives:
  * Learner will understand the importance of splitting the dataset for tuning
  * Learner will be able to set up an appropriate text preprocessing pipeline for tuning an LLM (`torch.utils.data.DataLoader`, `transformers.default_data_collator`)
  * Learner will be able to train an existing LLM on additional data (`transformers.Trainer.train`, `torch.optim.AdamW`, `accelerate.Accelerator`, `transformers.get_scheduler`)

Lesson 4.4: Evaluating Tuned LLMs
- Learning Objectives:
  * Learner will understand the concept of NLP model perplexity
  * Learner will be able to evaluate a tuned LLM in terms of of perplexity (`transformers.Trainer.evaluate`)
  * Learner will be able to use the tuned LLM to make predictions on unseen text data via a pipeline (`transformers.pipeline`)
  * Learner will be able to save and load their tuned LLM (`transformers.AutoModelForSeq2SeqLM.from_pretrained`, `transformers.AutoModelForSeq2SeqLM.save_pretrained`)
