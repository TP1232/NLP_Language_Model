# Neural Network Language Model
## Introduction
- A general package to learn neural network language models, including:
  + RNN (BiRNN), deep-RNN (deep-biRNN)
  + GRU (BiGRU), deep-GRU (deep-biGRU)
  + LSTM (BiLSTM), deep-LSTM (deep-biLSTM)

## Review-domain Language Model
We will train a neural network language model using reviews on Yelp dataset. Where:  
- Inputs are stored in the *dataset* folder:
  + Train_data (*./dataset/train.small.txt*): used for training a model
  + Dev_data (*./dataset/val.small.txt*): used for validating a model
  + Test_data (*./dataset/test.small.txt*): used for testing a model
- Outputs are saved in the *results* folder:
  + A trained-model file *lm.m*
  + A model-argument file *lm.args*
- Applications:
  + Text generation: generate a restaurant review 
    - P(qj)=P(w1,w2,...,wn)
  + Text recommendation: recommend next words given a context
    - P(wi|wi-1,...,w1)

## Repository's Structure
  1. model.py: main script to train the model
  2. predict.py: an interface to generate random text and recommend words given context
  3. utils/
    - core_nns.py: a script to design the neural network architecture
    - data_utils.py: a script to load data and convert to indices
  4. results/: a folder to save the trained models
    - *.m: a trained-model file
    - *.args: a model-argument file
  5. dataset/: a folder to save a tiny dataset
## Design workflow
1. How to train a language model
- Design a model architecture (mainly done in utils/core_nns.py)
- Load a dataset map to indices (mainly done in utils/data_utils.py)
- Feed data to the model for training (mainly done in model.py)
- Save the model into a model file (mainly done in model.py)
- Stop the training process (mainly done in model.py) when  
    + having no improvement on testing data or;
    + exceeding the maximum number of epochs
2. How to predict using a trained model
- Load saved files at the training phase
- Write the ``predict.py`` file to predict the next words given some initial words: P(wi|wi-1,...,w1) 
and generate random text.

## Project usage
1. Download this repository: git clone https://github.com/TP1232/NLP_Language_Model
2. Train the model:
  - Change the current directory to "assignment3"
  - Run this command:
    ```
    python model.py [--use_cuda]
    ```
  - See other parameters, this command:
    ```
    python model.py -h
    ```

## Project


1. Part 1 - Theory questions:  

    a. What is the purpose of *bptt_batch()*  and *repackage_hidden()* functions in *model.py*  
    
    b. Describe an overview procedure of the *train()*  function in *model.py*  
    
2. Part 2 - Coding 

    - DON'T TRAIN THE MODEL: In this project, we will focus on writing an inference of an LM using available 
    pre-trained models. **NOTE:** we are working with a pre-trained LM trained on a tiny dataset, which doesn't provide 
    good performance; thus, coding is more important than model's results
    
    - Before coding, open new terminal at the repository and run following commands to create a virtual environment 
    and install all required libraries:

    ```commandline
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    NOTE: Assume that you are using python3.8 as a default Python version 

    - Main methods
     
        a. **load_model()**: Load the saved argument file and the model file 
         
        b. **generate()**: Generate a random document starting from the start-of-sentence token *SOS* and 
        stopping when reaching the end-of-sentence token *EOS* by employing a greedy sampling technique  
        
        c. **recommend()**: Recommend next possible k words with their corresponding probabilities given some initial context 
        
    - Then, *app.py* file for locally serving models.
    
        c. **getgenerate()**: serve the **generate()** function
        
        d. **getrecommend()**: serve the **recommend()** function

