# Capstone-Project
Kaggle Competition - Shopee Price Match Guarantee

# Problem Statement

Shopee is the leading e-commerce platform in SEA and Taiwan with the largest item listing on their website in the region. Being an e-commerce website, sellers are able to list items, set prices, upload images with autonomy. This creates a potential problem, sellers can create duplicate item listings with no significant difference. This may be accidental, if sellers have the same supplier. It may also be intentional where the seller sells products at different prices or to create a false sense of competition. This is bad for the user experience for customers on their platform. Hence there is a business use case to detect for duplicate item listings.

In this notebook, we seek to detect for duplicate item listing on an e-commerce website from the product image and title only.

# Summary

To tackle this problem,

1) Generate image embeddings

We employ EfficientNetB3 model to get image embeddings. Different levels of re-training was conducted on the model.

2) Generate word embeddings

Text embeddings were created using Language-agnostic BERT Sentence Embedding. We also employ a TFIDF vectorizer to compare the results of the 2 model.

3) Use cuML Nearest Neighbors algorithm to detect for duplicates product listing by sorting by Cosine Distance

We get the 50 nearest neighbors of the observation in the feature embedding space, then set a cosine distance threshold to determine if a product is a duplicate.

4) Employ ensemble model

We use set union to combine predictions as well as combining embeddings.

# Results

|Model| Cosine   distance Threshold | F1   Score |   |   |
|------------------------------------------------------------------------|-----------------------------|------------|---|---|
| **Baseline**                                                               |                             |            |   |   |
| No predictions                                                         | -                           | 0.461      |   |   |
| Image   PHASH                                                          | -                           | 0.553      |   |   |
| **Image   Models**                                                         |                             |            |   |   |
| Default   EfficientNetB3                                               | 0.2                         | 0.665      |   |   |
| EfficientNetB3,   re-train last layer                                  | 0.2                         | 0.677      |   |   |
| EfficientNetB3,   re-train last layer and last residual block          | 0.25                        | 0.678      |   |   |
| **Text   Models**                                                          |                             |            |   |   |
| TFIDF   Vectorizer                                                     | 0.45                        | 0.648      |   |   |
| LaBSE Embeddings                                                                 | 0.2                         | 0.621      |   |   |
| **Ensemble Models**                                                        |                             |            |   |   |
| Union of Predictions (LaBSE +   EfficientNetB3 w retraining last block)                                        | -                           | 0.721      |   |   |
| Combined Embeddings (LaBSE +   EfficientNetB3 w retraining last block) | 0.35                        | 0.719      |   |   |
| Combined Embeddings (TFIDF +   EfficientNetB3 w retraining last block) | 0.25                        | 0.677      |   |   |


