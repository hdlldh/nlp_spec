# Natural Language Processing Specialization

Natural Language Processing (NLP) uses algorithms to understand and manipulate human language. This technology is one of the most broadly applied areas of machine learning. As AI continues to expand, so will the demand for professionals skilled at building models that analyze speech and language, uncover contextual patterns, and produce insights from text and audio.
This Specialization will equip you with the state-of-the-art deep learning techniques needed to build cutting-edge NLP systems. By the end of this Specialization, you will be ready to design NLP applications that perform question-answering and sentiment analysis, create tools to translate languages and summarize text, and even build chatbots.

This Specialization is for students of machine learning or artificial intelligence as well as software engineers looking for a deeper understanding of how NLP models work and how to apply them. Learners should have a working knowledge of machine learning, intermediate Python including experience with a deep learning framework (e.g., TensorFlow, Keras), as well as proficiency in calculus, linear algebra, and statistics. If you would like to brush up on these skills, we recommend the Deep Learning Specialization, offered by deeplearning.ai and taught by Andrew Ng.

This Specialization is designed and taught by two experts in NLP, machine learning, and deep learning. Younes Bensouda Mourri is an Instructor of AI at Stanford University who also helped build the Deep Learning Specialization. Łukasz Kaiser is a Staff Research Scientist at Google Brain and the co-author of Tensorflow, the Tensor2Tensor and Trax libraries, and the Transformer paper.

## Course 1: Classification and Vector Spaces in NLP

This is the first course of the Natural Language Processing Specialization.

**Week 1: [Logistic Regression for Sentiment Analysis of Tweets](1_classification_vector_spaces/Week_1/assignment/C1_W1_Assignment.ipynb)**

- Use a simple method to classify positive or negative sentiment in tweets
- Labs:
	- [Natural Language preprocessing](1_classification_vector_spaces/Week_1/labs/C1_W1_lecture_nb_01_preprocessing.ipynb)
	- [Visualzing word frequencies](1_classification_vector_spaces/Week_1/labs/C1_W1_lecture_nb_02_word_frequencies.ipynb)
	- [Visualizing tweets and Logistic Regression models](1_classification_vector_spaces/Week_1/labs/C1_W1_lecture_nb_03_logistic_regression_model.ipynb) 	

**Week 2: [Naïve Bayes for Sentiment Analysis of Tweets](1_classification_vector_spaces/Week_2/assignment/C1_W2_Assignment.ipynb)**

- Use a more advanced model for sentiment analysis
- Labs:
	- [Visualizing likelihoods and confidence ellipses](1_classification_vector_spaces/Week_2/labs/C1_W2_lecture_nb_01_visualizing_naive_bayes.ipynb)

**Week 3: [Vector Space Models](1_classification_vector_spaces/Week_3/assignment/C1_W3_Assignment.ipynb)**

- Use vector space models to discover relationships between words and use principal component analysis (PCA) to reduce the dimensionality of the vector space and visualize those relationships
- Labs:
	- [Linear algebra in Python with Numpy](1_classification_vector_spaces/Week_3/labs/C1_W3_lecture_nb_01_linear_algebra.ipynb)
	- [Manipulating word embeddings](1_classification_vector_spaces/Week_3/labs/C1_W3_lecture_nb_02_manipulating_word_embeddings.ipynb)
	- [Another explanation about PCA](1_classification_vector_spaces/Week_3/labs/C1_W3_lecture_nb_03_pca.ipynb)

**Week 4: [Word Embeddings and Locality Sensitive Hashing for Machine Translation](1_classification_vector_spaces/Week_4/assignment/C1_W4_Assignment.ipynb)**

- Write a simple English-to-French translation algorithm using pre-computed word embeddings and locality sensitive hashing to relate words via approximate k-nearest neighbors search
- Labs:
	- [Rotation matrices in R2](1_classification_vector_spaces/Week_4/labs/C1_W4_lecture_nb_01_vector_manipulation.ipynb)
	- [Hash functions and multiplanes](1_classification_vector_spaces/Week_4/labs/C1_W4_lecture_nb_02_hash_functions_and_multiplanes.ipynb)


## Course 2: Probabilistic Models in NLP

This is the second course of the Natural Language Processing Specialization.

**Week 1: [Auto-correct using Minimum Edit Distance](2_probabilistic_models/Week_1/assignment/C2_W1_Assignment.ipynb)**

- Create a simple auto-correct algorithm using minimum edit distance and dynamic programming
- Labs:
	- [Building the vocabulary](2_probabilistic_models/Week_1/labs/C2_W1_lecture_nb_01_building_the_vocabulary_model.ipynb)
	- [Candidates from edits](2_probabilistic_models/Week_1/labs/C2_W1_lecture_nb_02_candidates_from_edits.ipynb)

**Week 2: [Part-of-Speech (POS) Tagging](2_probabilistic_models/Week_2/assignment/C2_W2_Assignment.ipynb)**

- Apply the Viterbi algorithm for POS tagging, which is important for computational linguistics
- Labs:
	- [Working with text files](2_probabilistic_models/Week_2/labs/C2_W2_lecture_nb_1_strings_tags.ipynb)
	- [Working with tags and Numpy](2_probabilistic_models/Week_2/labs/C2_W2_lecture_nb_2_numpy.ipynb)

**Week 3: [N-gram Language Models](2_probabilistic_models/Week_3/assignment/C2_W3_Assignment.ipynb)**

- Write a better auto-complete algorithm using an N-gram model (similar models are used for translation, determining the author of a text, and speech recognition)
- Labs:
	- [Corpus preprocessing for N-grams](2_probabilistic_models/Week_3/labs/C2_W3_lecture_nb_01_corpus_preprocessing.ipynb)
	- [Building the language model](2_probabilistic_models/Week_3/labs/C2_W3_lecture_nb_02_building_the_language_model.ipynb)
	- [Language model generalization](2_probabilistic_models/Week_3/labs/C2_W3_lecture_nb_03_oov.ipynb)

**Week 4: [Word2Vec and Stochastic Gradient Descent](2_probabilistic_models/Week_4/assignment/C2_W4_Assignment.ipynb)**

- Write your own Word2Vec model that uses a neural network to compute word embeddings using a continuous bag-of-words model
- Labs:
	- [Data Preparation](2_probabilistic_models/Week_4/labs/C2_W4_lecture_nb_1_data_prep.ipynb)
	- [Intro to CBOW model](2_probabilistic_models/Week_4/labs/C2_W4_lecture_nb_2_intro_to_CBOW.ipynb)
	- [Training the CBOW model](2_probabilistic_models/Week_4/labs/C2_W4_lecture_nb_3_training_the_CBOW.ipynb)
	- [Word Embeddings hands-on](2_probabilistic_models/Week_4/labs/C2_W4_lecture_nb_4_word_embeddings_hands_on.ipynb)
	- [Word Embeddings step by step](2_probabilistic_models/Week_4/labs/C2_W4_lecture_nb_5_word_embeddings_step_by_step.ipynb)


## Course 3: Sequence Models in NLP

This is the third course in the Natural Language Processing Specialization.

**Week 1: [Sentiment with Neural Nets](3_sequence_models/Week_1/assignment/C3_W1_Assignment.ipynb)**

- Train a neural network with GLoVe word embeddings to perform sentiment analysis of tweets
- Labs:
	- Introduction to Trax
	- Classes and Subclasses
	- Data Generators

**Week 2: [Language Generation Models](3_sequence_models/Week_2/assignment/C3_W2_Assignment.ipynb)**

- Generate synthetic Shakespeare text using a Gated Recurrent Unit (GRU) language model
- Labs:
	- Hidden State Activation
	- Vanilla RNNS, GRUs and the scan function
	- Working with JAX Numpy and Calculating Perplexity
	- Creating a GRU model using Trax

**Week 3: [Named Entity Recognition (NER)](3_sequence_models/Week_3/assignment/C3_W3_Assignment.ipynb)**

- Train a recurrent neural network to perform NER using LSTMs with linear layers
- Labs:
	- Introduction to LSTM

**Week 4: [Siamese Networks](3_sequence_models/Week_4/assignment/C3_W4_Assignment.ipynb)**

- Use so-called ‘Siamese’ LSTM models to compare questions in a corpus and identify those that are worded differently but have the same meaning
- Labs:
	- Creating a Siamese Model using Trax
	- Modified Triplet Loss
	- Evaluate a Siamese Model

## Course 4: Attention Models in NLP

This is the fourth course in the Natural Language Processing Specialization.

**Week 1: [Neural Machine Translation with Attention](4_attention_models/Week_1/assignment/C4_W1_Assignment.ipynb)**

- Translate complete English sentences into French using an encoder/decoder attention model
- Labs:
	- Basic Attention
	- Scaled Dot-Product Attention
	- BLEU Score
	- Stack Semantics

**Week 2: [Summarization with Transformer Models](4_attention_models/Week_2/assignment/C4_W2_Assignment.ipynb)**

- Build a transformer model to summarize text
- Labs:
	- Attention
	- The Transformer Decoder

**Week 3: [Question-Answering with Transformer Models](4_attention_models/Week_3/assignment/C4_W3_Assignment.ipynb)**

- Use T5 and BERT models to perform question answering
- Labs:
	- SentencePiece and BPE
	- Question Answering with HuggingFace 

**Week 4: [Chatbots with a Reformer Model](4_attention_models/Week_4/assignment/C4_W4_Assignment.ipynb)**

- Build a chatbot using a reformer model
- Labs:
	- Reformer LSH
	- Revnet
