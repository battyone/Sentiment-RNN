# **Requirements** 
* Python (3.6.0)
* numpy (1.16.0)
* keras (2.2.4) 
* tensorflow (1.13.1)

# **Classification: sentiment analysis** 

## **Background**
This dataset has been extensively studied after it was published in
[2011](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf). [Multiple machine learning architectures](https://www.kaggle.com/c/word2vec-nlp-tutorial/data)
were also developed to improve accuracy.
This problem achieved upper bound accuracy however the dataset is
very useful as the gold standard for benchmarking different machine learning algorithms. 
Here, the dataset is used to understand activation functions.

## **Methods** 
### **Dataset** 
In this work, a preprocessed dataset available from [keras](https://keras.io/datasets/) was used.
This dataset consists of 50000 movie reviews with their label either positive or negative to indicate the sentiment. Dataset is divided into training (50%) and testing (50%) set,   which resulted in 25000 samples on each side. Training samples were further divided into training (90%) 
and validation (10%) datasets.

```
(data_train, value_train), (data_test, value_test) = imdb.load_data(path="imdb.npz", num_words=10000)
```
By restricting num_word parameter to 10000 (most frequently occuring words) in "imdb.load_data" function,
training and testing dataset consists with 5967841 and 5770105 words.
Among these, 9998 and 9951 are unique in training and testing dataset
(for detail please look any ipynb file in src directory).

Training and testing data contain 25000 samples with variable sequence length. 
Mean sequence length is around 240 with standard deviation 175 for training and 230 with 
standard deviation 170 for testing. Based on the distribution of sequence length, the mean 
sequence length is selected ~400, which is around one standard deviation above a mean as maxlen 
for padding. The data were converted into 2D array (matrix) using the following code to obtain an array of (25000, 400).

```
ar_data_train = sequence.pad_sequences(ar_data_train, maxlen=400)
```
### **Network Architecture** 
Recurrent neural network(RNN) was chosen for this classification. Multiple models with different
parameters for three architectures (SimpleRNN, LSTM, and GRU) listed below were tested:
* [Single](https://bit.ly/30MOnRq)
* [Bidirectional](https://bit.ly/32c3WSR)
* [Stack](https://bit.ly/2PgwJnv)

The 'rmsprop' was used as an optimizer.

## **Results** 
Here is a summary of multiple architectures with their accuracy in the test dataset (Table 1). 
The trend of loss functions for training and validation datasets within the 10 first epochs 
indicates the overfitted model.
Although they meet at some points in the course of model generation,
their convergence did not continue till the end.
The accuracy was ~87% for LSTM and GRU. As expected, SimpleRNN performed the worst among all.

|                      |  Single        | Bidirectional  | Stack       | 
|----------------------|----------------|----------------|-------------|
| RNN                  | 0.75/0.77      | 0.81/0.75      | 0.76/0.78   |
| LSTM                 | 0.87/0.87      | 0.87/0.87      | 0.86/0.87   |
| GRU                  | 0.87/0.87      | 0.86/0.87      | 0.86/0.87   |

Table 1: Accuracies in test dataset. Columns except first one represent an accuracy.
The reported accuracy was separated in which the first one is from
tanh activation and second is the results using sigmoid.

The primary goal of this experiment is to monitor the performance differences due to 
the activation functions. However, the performance of activation functions in the majority 
of cases is indistinguishable. The major improvement was observed for bidirectional RNN (tanh) 
over RNN (sigmoid) for which the accuracy is improved by 6%.
