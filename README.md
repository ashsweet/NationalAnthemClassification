# NationalAnthemClassification
**Overview**
This project aims to classify national anthems into various thematic clusters using a combination of K-means clustering and the BERT (Bidirectional Encoder Representations from Transformers) model. The national anthems are first grouped into clusters based on their thematic content using K-means, and then a BERT model is fine-tuned to classify new anthems into these predefined clusters.

**Dataset**
The dataset consists of 190 national anthems, each translated into English. These anthems are clustered into thematic groups using K-means clustering. The clusters represent different themes identified in the anthems, such as:

Call for Action and Valor
Devotion and Loyalty
Fatherland, Blessings, and Divinity
Freedom and Resistance
Homeland and Patriotism
Liberty and Triumph
Love for Land and Heritage
Monarchy and Divine Protection
National Glory and Unity
Struggle and Sacrifice
Unity and Aspirational
Unity and Collective Strength
Unity and National Glory

**Methodology**
K-Means Clustering
Feature Extraction:

The text data of the national anthems was first preprocessed and vectorized. Techniques like TF-IDF (Term Frequency-Inverse Document Frequency) were used to convert the text into numerical features suitable for clustering.

Clustering:

K-means clustering was applied to the feature vectors to identify natural groupings in the data. The optimal number of clusters was determined through methods such as the elbow method and silhouette analysis.
Each anthem was assigned to a cluster, which represents a thematic grouping.

**Data Preprocessing**
Text Tokenization:

After clustering, the BERT tokenizer was used to tokenize the anthems. Each anthem was converted into a sequence of token IDs with a maximum length of 512 tokens.
Padding and Attention Masks:

Sequences were padded to ensure uniform input length. Attention masks were generated to differentiate between real tokens and padding tokens.
Handling Imbalanced Data
Given the imbalanced nature of the clusters, oversampling techniques were applied to ensure that the minority classes were sufficiently represented. The RandomOverSampler from the imbalanced-learn library was used to balance the dataset.

Modeling
Model Architecture:

A pre-trained BERT model (bert-base-uncased) was fine-tuned for the classification task. The model includes a classification head that maps BERT's output to the thematic clusters identified by K-means.
Training:

The model was trained for 10 epochs with a learning rate of 3e-5 and a batch size of 8.
Class weights were computed to further address class imbalance, and the AdamW optimizer was used with a cosine learning rate scheduler.
Evaluation
The model's performance was evaluated using the following metrics:

Accuracy: The overall correctness of the model's predictions.
Precision, Recall, F1-Score: These metrics were calculated for each class to assess the model's ability to correctly identify and classify each thematic cluster.

Results
The fine-tuned BERT model, following K-means clustering, achieved the following results:

Validation Accuracy: 87%
Precision, Recall, F1-Score: High scores across most clusters, indicating that the model is both accurate and reliable.
Classification Report:
                                    precision    recall  f1-score   support

         Call for action and Valor       1.00      1.00      1.00         8
              Devotion and loyalty       0.86      0.86      0.86         7
Fatherland, blessings and divinity       0.92      1.00      0.96        12
            Freedom and Resistance       0.90      0.82      0.86        11
           Homeland and Patriotism       0.89      1.00      0.94         8
               Liberty and Triumph       0.83      1.00      0.91         5
        Love for land and heritage       0.70      0.88      0.78         8
    Monarchy and divine protection       0.88      0.70      0.78        10
          National Glory and Unity       1.00      1.00      1.00         8
            Struggle and sacrifice       0.67      1.00      0.80         6
            Unity and Aspirational       1.00      0.71      0.83         7
     Unity and collective strength       0.67      0.44      0.53         9
          Unity and national glory       1.00      1.00      1.00         8

Usage
Installation
To use this project, clone the repository and install the required dependencies:

bash
Copy code
git clone https://github.com/ashsweet/NationalAnthemClassification.git
cd national-anthem-classification
pip install -r requirements.txt
Running the Model
Prepare the Data:

Ensure your dataset is in the correct format with the necessary columns (Anthem, Cluster_Label).
Run the Training Script:

Use the provided training script to fine-tune the BERT model:
bash
Copy code
python train_model.py
Evaluate the Model:

After training, evaluate the model's performance using the test set:
bash
Copy code
python evaluate_model.py
Hyperparameter Tuning
The model can be fine-tuned further by experimenting with different hyperparameters such as learning rate, batch size, and the number of epochs. A grid search or random search approach can be used to find the optimal hyperparameters.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request if you have any improvements or new features.

License
This project is licensed under the MIT License - see the LICENSE file for details.















