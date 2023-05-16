# ISE244-Sentiment-and-weather-correlation

# Problem Definition
The impact of weather on human emotions and behavior is a widely researched area, and the use of social media data to analyze people’s sentiments towards the weather has become increasingly popular in recent years. In this project, I propose to investigate the correlation between weather and sentiment within social media data. The objective of the project is to understand how weather conditions affect the sentiments expressed on social media. The current research uses machine learning models like SVM, BERT, rule-based, and naive-bayes methods for sentimental analysis and poses the following challenges: 

- Handling multiple languages: Sentiment analysis across multiple languages poses unique challenges due to the variations in grammar, syntax, and cultural differences. 
- Irony and sarcasm: Irony and sarcasm can be difficult to detect in text, as they often involve the opposite meaning of what is actually being said.
- Ambiguity and subjectivity: Sentiment analysis can be difficult because language is often ambiguous and subjective. Words can have multiple meanings and can be interpreted differently depending on the context and the person reading them.

# Introduction
The weather’s impact on human emotions and behavior has been extensively researched, and in recent years, there has been an increase in interest in examining how people feel about the weather using data from social media.

Weather is an integral part of our daily lives and affects us in numerous ways. Our emotions and behavior are also closely tied to the weather conditions we experience. It is a well-known fact that weather can affect our daily routines, decision-making, and overall well-being. For instance, a sunny day may make us feel more positive and energized, while a gloomy day may make us feel sad and lethargic.

The weather has been shown in studies to have a variety of effects on our mood. Understanding how weather affects our emotions and behavior can assist us in developing strategies to improve our well-being, productivity, and, ultimately, quality of life.

Nowadays, Twitter, Facebook, and Instagram are used a lot. Social media platforms are a rich source of information about people’s sentiments and behaviors as people share their thoughts and comments without any restrictions or hesitation, making them an ideal tool to study the impact of weather on human emotions.

This project examines weather data and social media sentiment to see if there is any relationship between them. We will analyze tweets and their effects on individual moods and behaviors.

I would be fetching Twitter data from Twitter using their API and also fetching the weather of the place at the time the tweet originated. I would then use the roBERTa-base model, a transformer-based model trained on a dataset of approximately 58 million tweets and fine-tuned for sentiment analysis. I will then be using decision trees, random forests, or ensemble methods to determine the correlation between sentiment and weather in a particular location. The goal of this project is to use machine learning models to look into the relationship between weather and sentiment in social media data. By handling multiple languages, irony and sarcasm, ambiguity, and subjectivity in sentiment analysis, as well as using the roBERTa-base model, the project hopes to find out how weather affects how people feel on social media. The proposed method, which includes analyzing the data with decision trees, random forests, or ensemble methods, appears to be an effective way to reach the project objective.

Finally, the goal of this research is to improve our understanding of the complex relationship between mood and weather, as well as to inform strategies for promoting positive emotional health in various weather conditions.

# Data

Twitter Data - Tweets were collected over the period using the Twitter API. A filter was put in to get only tweets within the United States and the English language. We get features like ID, tweets, favourites, language, location, retweets, and many more. A filter was sent in the request to return only geo-tagged tweets in a bounding box that included tweets from San Jose and Bangalore. Additional filtering was included to remove any tweets, not within those cities and any languages that were not English. Only tweets where the coordinates were provided were used. 

Weather data - I have used the weatherapi website to fetch the weather details on a more frequent basis for which the most recent report available is the current weather. Useful information in these reports includes temperature, pressure, wind speed, weather condition (e.g. rain, snow, sunny, clear), time, and precipitation. Additional metrics such as humidity and solar radiation can be derived from this report. The climate is a composite of weather conditions taken over a period of time and is connected to the tweets based on the time. So, each tweet has corresponding weather features attached to it.

# Proposed Methodology
The implementation plan involves the following steps:

- Collecting social media data and weather data.
- Preprocessing the data to remove noise and irrelevant information.
- Performing sentiment analysis using the cardiffnlp/twitter-roberta-base-sentiment model.
- Analyzing the results to determine the correlation between weather conditions and sentiment expressed in social media data using machine learning models.

# Models

To find the correlation between weather and sentiment I would be using multiple machine models like decision tree, XGBOOST, Random Forest, Neural Networks, and Ensemble model. I will be applying all these techniques on two datasets i.e. tweets based on Bangalore and San Jose. In the end, I will be evaluating and comparing all the models using metrics like accuracy, precision, recall, and F1-Score. 
The models I have used are mainly used for classification as the main objective of this project is to classify the sentiments based on the weather patterns. 

### Decision Tree

The decision tree is used for classifying the data based on the criteria it finds the most important. It can also handle categorical data which is very helpful.
In this project, I am using a decision tree to find the sentiment based on weather details and find the weather condition based on the previous sentiment probability for both the datasets. 
This is one of the decision tree models which was used in predicting the weather conditions.
Then I also used grid search for hyperparameters tuning which gives a slightly better result when compared to the base model.

![image](https://github.com/ketanmalempati/ISE244-Sentiment-and-weather-correlation/assets/57043103/19f3b95d-745b-45ba-bcf9-d6b2dfcc9830)

### XGBoost

XGBoost, which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree (GBDT) machine learning library. It provides parallel tree boosting and is the leading machine learning library for regression, classification, and ranking problems. 

For using XGBOOST I had to convert the categorical data to numeric as they don’t work with categorical data. They are slower compared to decision trees as they have a much more complex architecture.

![image](https://github.com/ketanmalempati/ISE244-Sentiment-and-weather-correlation/assets/57043103/0403b30b-367b-48a3-a271-1bb54810e42f)

### Random Forest

Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. 

![image](https://github.com/ketanmalempati/ISE244-Sentiment-and-weather-correlation/assets/57043103/6ddf84df-5d27-4e09-8bfa-f9fff575eb79)

### Neural Networks

Neural networks, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs), are a subset of machine learning and are at the heart of deep learning algorithms. 

Artificial neural networks (ANNs) are comprised of a node layer, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network. 
I am using neural networks for this because it can find some unseen correlations in the data and can be easily trained and hyperparameter tuning could be easily done. I am using 4 layers, first a sequential layer and then followed by 3 dense layers. In the 2nd and 3rd layer I have used relu as the activation function and a sigmoid function for the last layer. I am also using binary cross entropy as the loss function and adam as the optimizer.

![image](https://github.com/ketanmalempati/ISE244-Sentiment-and-weather-correlation/assets/57043103/ac4f28aa-66b3-46af-854e-3b1e8a3bd1ba)

### Ensemble Model

Ensembling involves combining all the models and getting a better result from the output of all the models. Here I am using all the above models discussed above and combining them and using the output from those models and sending the ensemble model to get a better understanding of the data.


# RESULTS

Predicting weather conditions based on the sentiment

When looking at the results we can see that both the datasets have very similar scores. But the Bangalore dataset performs better compared to the the San Jose dataset in both models.

![image](https://github.com/ketanmalempati/ISE244-Sentiment-and-weather-correlation/assets/57043103/9006b5a8-2d40-460f-b9c4-273d54c62472)
