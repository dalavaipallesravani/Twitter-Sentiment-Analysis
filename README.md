# Twitter-Sentiment-Analysis
Twitter Sentiment Analysis using Deep Learning for NLP - Built and evaluated RNN, LSTM, GRU and Bidirectional GRU models on tweet data

Overview:
This project performs sentiment analysis on Twitter data using deep learning models. The goal is to classify tweets into three categories: Positive, Neutral, and Negative.
The project demonstrated Natural Language Processing (NLP) techniques combined with sequence modeling using RNN-based architectures.

Dataset:
The dataset consists of Twitter posts labeled for sentiment. The data was separated into Training and Test data.
Labels: Positive, Neutral (including “Irrelevant”), and Negative.

Data Preprocessing:
1. Dropped rows not having any tweets using dropna
2. Cleaning the tweets to remove special characters, URL's, hastags and converted the text to lower case
3. Dropped the text which are blank spaces as this created issue with LSTM models. These blank texts were also picked up by Embedding layers causing the LSTM model to not learn and this was critical step to make LSTM model progress.
4. Separated the inputs and labels into separate dataframes
5. Tokenization on the input text(tweets) and converting to sequences
6. Label encoding the labels
7. Split the data into training and validation splits
8. Padding sequences to ensure uniform sequence length for all the inputs
9. Check for class imbalance and calculated class weights

Simple RNN Model Architecture:
1. Embedding Layer
2. SimpleRNN Layer
3. Dense Layer with activation as softmax
4. Model compile with Optimizer as Adam, loss as Sparse Categorical CrossEntropy
5. Model Performance: Training accuracy: ~75%, Validation accuracy: ~65%

Mutliple Simple RNN Layers, model architecture:
1. Embedding Layer
2. SimpleRNN Layer with return_sequences as True
3. SimpleRNN Layer with return_sequences as True
4. SimpleRNN Layer with return_sequences as True
5. SimpleRNN Layer
6. Dense Layer with activation as softmax
7. Model compile with Optimizer as Adam, loss as Sparse Categorical CrossEntropy
8. Model Fit with class weights
9. Model Performance: Training accuracy: ~35%, Validation accuracy: ~33%. Low performance due to vanishing gradients problem due to multiple RNN layers.

LSTM Model Architecture:
1. Embedding Layer with mask_zero as True.Without this parameter and without handling the blank texts like '', the model was not learning at all the loss was just the baseline loss log(1/3). After handling the blank texts and with mask_zero parameter the model started learning.
2. LSTM Layer with Dropout as 0.2. With recurrent dropout the performance was very slow hence was not able to use it.
3. Dense Layer with activation as softmax
4. Model compile with Optimizer as Adam, loss as Sparse Categorical CrossEntropy
5. Model fit with class weights
6. Model Performance: Training accuracy: ~96%, Validation accuracy: ~87%

GRU Model Architecture:
1. Embedding Layer with mask_zero as True.
2. GRU Layer with Dropout as 0.2
3. Dense Layer with activation as softmax
4. Model compile with Optimizer as Adam, loss as Sparse Categorical CrossEntropy
5. Model fit with class weights
6. Model Performance: Training accuracy: ~96%, Validation accuracy: ~86%

Bidirectional GRU Architecture:
1. Embedding Layer with mask_zero as True.
2. BidirectionalGRU Layer with Dropout as 0.2
3. Dense Layer with activation as softmax
4. Model compile with Optimizer as Adam, loss as Sparse Categorical CrossEntropy
5. Model fit with class weights and early stopping
6. Model Performance: Training accuracy: ~94%, Validation accuracy: ~87%

Test data predictions on Bidirectional GRU model:
Model performance: test accuracy : 95%
Confusion matrix and classification report generated
