# Reddit Implicit Interest Classification


# Social Media Marketing


## Introduction

The social media giants of the world earn a significant amount of their revenue from advertising; specifically, advertising the right product or service to the right audience. They achieve this by matching an advertisement with users who have expressed interests that are complementary to the product or service being marketed. 

Whilst this has been incredibly successful, it often relies on users to explicitly express their interests. Typically in the form of subscribing to particular people, groups or communities. However with so many different things to subscribe to, it is unlikely a user has subscribed to everyone and everything they find interesting.

As such it stands to reason that if these social media companies could identify users who have not expressed an explicit interest in something but do have an implicit interest in that subject, they could increase the target audience of a marketing campaign without sacrificing conversion rates.

This would mean the company behind the marketing campaign could gain more customers and the social media companies could earn more revenue, from the exact same advert. Win win.

I wanted to see if I could use a classification model to predict whether a user has an implicit interest in a particular subject based on the interests they have explicitly expressed, to serve as a proof of concept for the business use case outlined above.

For this project I chose to use Reddit as the social media company because their users are anonymous and they offer an API which allows me to acquire the data without much fuss. I also chose the subreddit wallstreetbets as the target community a user is or isn’t interested in because I thought it was funny.


## Hypothesis

Can we predict whether a Reddit user has interacted or will interact with a particular subreddit (eg wallstreetbets) based on their previous interactions with other subreddits?

Where an interaction with a subreddit is defined as publishing a post or comment in that subreddit.


## Limitations & Assumptions

As with any experiment, there were a number of limitations discovered and assumptions made. 


### Limitations

- Reddit doesn’t provide data regarding which subreddits a user is subscribed to

- Sample of only ~6,500 users in our dataset

- Maximum of 200 unique interactions by each user (100 posts, 100 comments)

- Almost definitely bots amongst users


### Assumptions

- Users only subscribe to subreddits they interact with

- Our sample is representative of the entire Reddit community

- All users in our dataset are organic

- User behaviour and interests remain similar over time


## Data Mining

Included in the data that Reddit provides about a subreddit is a JSON object containing the posts made in that subreddit. Within the metadata of each post, they provide the username of the author of that post.

Likewise for a given post, the Reddit API provides a JSON object containing the comments made on that post. Within the metadata of each comment, they provide the username of the author of that comment.

In order to come up with a list of users, I wrote a script that would collect the usernames of the authors of the 100 most recent posts in a seed subreddit. Then for each of those posts, the script would collect the usernames of the authors of all of the comments on that post.

Below is an illustration of the tree-like fashion in which the usernames were collected. 

<img width="1005" alt="Screenshot 2022-03-14 at 14 34 20" src="https://user-images.githubusercontent.com/91144560/158194374-7fbb7eda-488a-4d68-9422-0f21498433c7.png">

Given that my target subreddit is to be wallstreetbets, I began by defining that as the seed subreddit in my script. In this way I was able to collect over 3000 unique users that would form the positive target class.

I then defined the subreddits “aww”, “food” & “science” as the seed subreddits to collect several thousand more unique users. Of those, the users who had never interacted with wallstreetbets would go on to form the negative target class. 

Once the two sets of users were obtained (positive & negative target classes), I queried the API to get the 100 most recent posts and 100 most recent comments published by each user. Note, for users who had published fewer than 100 posts or 100 comments, their entire post/comment history was collected.

Similar to the data provided regarding posts and comments in a particular subreddit, I was able to collect data regarding the posts and comments made by a particular user as a JSON object. Within the metadata of each post and comment was the subreddit in which that interaction occurred along with the score achieved, the number of awards earned and number of comments received. Note, metadata regarding number of comments received was only available if the original interaction was a post.

In this manner, I was able to collate the attributes of the recent unique interactions by each user.

Below is an example of data collected for the user OliveInvestor in the form of a Pandas DataFrame.

<img width="268" alt="Screenshot 2022-02-23 at 16 42 07" src="https://user-images.githubusercontent.com/91144560/158195133-aac1bebe-a871-4e14-a966-d62d1ba440b5.png">

Each row represents a single interaction. Interactions with a NaN value in the comments column are themselves comments. Posts that have received no comments have a value of 0 in the comments column.


## Feature Engineering

I believed the best format for the complete dataset to be as model-friendly as possible was to engineer the data into some sort of 3D dummified DataFrame with users as rows and subreddits as columns. 

Instead of just having values of 1 or 0 representing if a user has or has not interacted with a subreddit, if they have interacted with that subreddit I wanted the value to be a non-zero number representing the degree of their interactions with that subreddit.

This would require some feature engineering to condense the data collected into single values for each subreddit, for each user. I began by aggregating the interaction data for each user by subreddit, which condensed the data vertically. 

For each subreddit the following measures of interaction were calculated:

- The number of interactions

- The mean score achieved across those interactions

- The mean number of awards earned across those interactions

- The mean number of comments received across those interactions

Below is an example of the resulting aggregated DataFrame for the user OliveInvestor.

<img width="380" alt="Screenshot 2022-02-23 at 18 52 09" src="https://user-images.githubusercontent.com/91144560/158195384-17b2b9ec-e3dd-4d21-a99e-984ef067e6d9.png">

Having done this, I now had to condense the data horizontally by using the measures of interaction calculated above to define a single value for each subreddit, for each user. This required me to define an expression in order to calculate this value, which is shown below.
 
<img width="943" alt="Screenshot 2022-03-14 at 14 51 22" src="https://user-images.githubusercontent.com/91144560/158197767-ca2eeaad-4e63-4706-b5d0-02ab8e7c191b.png">

Where, for each user:

Vi = value for any given subreddit
Ni = number of interactions with the subreddit
Si = mean score achieved across all interactions with the subreddit
Ai = mean number of awards earned across all interactions with the subreddit
Ci = mean number of comments received across all interactions with the subreddit

However, I believe that a user is more interested in subreddits they have posted in than subreddits they have only commented in. Therefore values for subreddits that a user only commented in were half weighted.

Below is an example of the output after applying this methodology and transposing the resulting DataFrame, which will go on to serve as a single row in our complete dataset.

<img width="959" alt="Screenshot 2022-02-23 at 17 03 39" src="https://user-images.githubusercontent.com/91144560/158195485-7a69b978-f8db-447f-8d29-4c1dd0581c67.png">

Note that this is simply a representation of a single row, the complete dataset actually contains 6,480 rows and 25,609 columns.


## Data Cleaning

Given the manner in which the data has been mined and engineered, there wasn’t much cleaning required. However, the purpose of this project was to predict whether a user has interacted or will interact with the subreddit wallstreetbets based on their previous interactions with other subreddits. Namely, which other subreddits they have interacted with and the degree of those interactions.

As such, in order to maintain the integrity of the project, I had to control for users who have interacted with very many / few different subreddits and those with very high / low total values. This was done in order to ensure that these factors did not inadvertently influence the machine learning models.

Outliers were defined as those users who lay above the 75th percentile + 1.5 * IQR or below the 25th percentile - 1.5 * IQR 

Below are histograms illustrating the distributions of total values and number of subreddits for each user, with the outliers included.

<img width="1175" alt="Screenshot 2022-03-14 at 14 42 35" src="https://user-images.githubusercontent.com/91144560/158195866-c6c98072-adb2-4fa1-8c64-89d493c25174.png">

And here are histograms illustrating the distributions of total values and number of subreddits for each user, with the outliers removed.

<img width="1175" alt="Screenshot 2022-03-14 at 14 43 21" src="https://user-images.githubusercontent.com/91144560/158195989-3ae4dc38-2d57-4634-8681-6dd1417e3d28.png">

The dataset contained thousands of subreddits which only one or two users had interacted with. Now that the outliers had been removed, there were several empty columns. These were removed along with columns that now only had a single entry, as these would provide little or no value to the machine learning models. Following this, the dataset now contains 6,192 rows and 11,076 columns.


## Exploratory Data Analysis

Once the dataset had been cleaned, I wanted to investigate which subreddits would have the greatest differences between the target classes. This was in terms of both whether or not they had been interacted with and the degree to which they have been interacted with. This would go on to guide my feature selection as 11,075 predictors is quite a lot of predictors to have and I would like to narrow this down somewhat. From the bar charts below, you can see that some subreddits are among the most significant for both measures of difference.

Subreddits with the greatest population differences between target classes.

<img width="882" alt="Screenshot 2022-03-14 at 14 44 17" src="https://user-images.githubusercontent.com/91144560/158196195-501ee565-397f-4faa-b9a8-83b8f5fadd82.png">

Subreddits with the greatest total value differences between target classes.

<img width="872" alt="Screenshot 2022-03-14 at 14 45 18" src="https://user-images.githubusercontent.com/91144560/158196404-9764385e-5f83-4f80-b46f-f1d209004810.png">


## Feature Selection

Inspired by the EDA, I chose to narrow down the number of predictors by identifying the intersection between the subreddits that meet three different criteria. These were the 1000 most populated columns across the dataset (i.e. fewest 0 values), the 1000 columns with the greatest population differences between the target classes, and the 1000 columns with the greatest total value differences between the target classes. This resulted in a subset of 398 predictors from the original 11,075. This method of feature selection is illustrated below with a venn diagram.

<img width="463" alt="Screenshot 2022-03-10 at 21 25 12" src="https://user-images.githubusercontent.com/91144560/158196492-0ed129b2-8148-495a-8264-33d361ee0da8.png">

To go a step further I decided to create a second, narrower subset of predictors. To do this, of the 398 predictors in the initial subset, I selected those with a correlation to the target column of greater than +0.1 or less than -0.1. This resulted in a further subset of 14 predictors. Below you will find a Seaborn correlation heatmap illustrating the results of this approach.  

<img width="1111" alt="Screenshot 2022-03-14 at 14 46 39" src="https://user-images.githubusercontent.com/91144560/158196683-e0e4a192-5dc6-4a81-b0de-bdc651063a80.png">

Following the feature selection, I ended up with three sets of data that I will go on to use with each of the classification models: the entire dataset, the initial subset & the second narrower subset. 


## Modelling

Now that the data has been cleaned, I have to select which classification models to use and prepare the data for modelling. I settled on two core models, Decision Tree Classifier and Logistic Regression. 

I chose the Decision Tree because I expected some variant of this model to perform the best, but when it comes to evaluating feature significance the model does not allow me to identify which target class each predictor contributes to. 

On the other hand, whilst I did not expect a Logistic Regression variant to perform as well, I could extract the coefficients the model assigned to each predictor to determine which target class each contributes to. A predictor with a positive coefficient would contribute towards the positive target class and a predictor with a negative coefficient would contribute towards the negative target class. 

With the two core classification models selected, I decided to use various ensemble methods to introduce an element of noise into the underlying models in order to make them more robust. These were a standard bagging classifier, a bagging classifier with bootstrapped features, and a random forest. Below is an illustration of the models I selected to use and how they are related.

<img width="638" alt="Screenshot 2022-03-10 at 21 28 20" src="https://user-images.githubusercontent.com/91144560/158196796-80da18ee-4c14-43f8-8aab-057dc40b611e.png">

Now that the models had been selected, it was time to prepare the data for modelling. First, this involved the creation of a train-test split using an 80:20 ratio. Following the split, I used the training data to fit a standard scaler and in turn used this to transform both the training and testing sets. 

However, given that values of 0 represent the lack of interaction with a subreddit, I believed it to be important that these values should remain as 0s following the transformation. Fortunately, the StandardScaler function within Scikit-learn allowed me to do just that by setting the ‘with_mean’ parameter to False.

Finally, with each type of model I fit and tested them using each of the three sets of features. Namely, the entire set of features, the initial subset and the second, narrower subset. This resulted in a total of 18 different classification models.


## Analysis & Evaluation

The best performing model was a Random Forest classifier using 398 predictors. Below is the baseline accuracy and the test accuracy this model achieved.

- Baseline Accuracy:  51.8%

- Test Accuracy:      93.7%

As a general rule of thumb, an accuracy of between 80% and 90% is brilliant but an accuracy greater than 90% is suspicious. The very high accuracy achieved by the model causes concerns regarding overfitting and/or data leakage. The following is how I addressed those concerns.


### Overfitting

If the model was overfitting, there would be significant differences between the different accuracies.

- Cross Validation Accuracies: 		  94.1%, 93.2%, 94.1%, 93.8%, 93.7%

- Mean Cross Validation Accuracy: 	93.8%

- Test Accuracy: 			              93.7%

- Outlier Accuracy: 			          95.1%

As shown above, the individual cross validation accuracies exhibit only small differences between each other, as well as between the mean CV accuracy and the test accuracy. In addition, when testing with only the outlier users removed during the data cleaning stage, the model achieves an even higher accuracy score. Therefore, I can conclude that the model is not overfitting.


### Data Leakage

If there were data leakage, all models implemented would have achieved remarkably high accuracies.

The worst performing model was a Logistic Regression using 11,075 predictors. Below is the baseline accuracy and the test accuracy this model achieved.

- Baseline Accuracy: 	51.8%

- Test Accuracy: 	    70.3%

While this test accuracy score is respectable, it is not particularly remarkable. As such, I am comfortable concluding that there has been no data leakage either.


### Precision & Recall

Given the business case for this project, I believe that Precision and Recall are the most important metrics to evaluate the winning model. 

When trying to identify users with an implicit interest in a topic, I want to make sure that as many of those that are believed to be interested actually are. Otherwise one would be showing adverts to people who are unlikely to engage with them, which would be a waste of both time and money. This is represented by the precision.

On the other hand, I want to capture as many users with an implicit interest as possible. This would mean that the identified target audience successfully captures as many relevant users as possible, which would lead to higher conversion figures and revenue. This is represented by the recall.

Below are the precision and recall scores achieved by the best performing model. Of the two, I believe that precision is the more important metric, as it is ultimately a social media company’s ability to be precise about who they show a particular advert to that has fuelled their success.

- Precision:  0.97

- Recall:     0.91

Note, precision and recall have an inverse relationship. As one is increased, the other decreases. Fortunately, the model finds the optimal balance between the two by default. This relationship and the optimal point are illustrated by the Precision-Recall Curve below.

<img width="695" alt="Screenshot 2022-03-14 at 14 48 01" src="https://user-images.githubusercontent.com/91144560/158197008-80bcbae7-ea87-4dd5-b9c8-8b38648ad60c.png">


### Feature Importance & Class Contribution

Given that the Random Forest Classifier with 398 predictors was the best performing model, it is from this model I would like to extract the significance of each predictor. However, in order to identify which target class each predictor contributes to, I examined the positive and negative signs in front of the corresponding Logistic Regression coefficients and mapped them on to the feature importances from the Random Forest model.

While this is not an entirely accurate method to obtain class contribution, those predictors with the largest feature importances were also the predictors with the largest absolute coefficients. Therefore, I feel comfortable examining only the most significant features in this manner, as there may be some ambiguity with regards to the features that contribute very little. Those with very low feature importances would have coefficients of just below or just above 0, which would make identifying which class they contribute unreliable.  

Below is a horizontal bar chart illustrating subreddit Feature importance (Random Forest) and Class contribution (Logistic Regression). Bars with positive numerical values contribute to the positive target class ie user has interacted with wallstreetbets, and those with negative values contribute to the negative target class.

<img width="714" alt="Screenshot 2022-03-14 at 14 48 57" src="https://user-images.githubusercontent.com/91144560/158197234-2902495d-cdef-4289-a9b8-db679ba17970.png">

These results seem to suggest that the model can identify the negative target class more easily than the positive target class. As such, it seems to use a process of elimination more heavily than I’d like.


## Improvements

There are a number of improvements that I believe could be made to further enhance this project. These include:

- Apply a Grid Search to optimise the hyperparameters of the best performing model 

- Collect significantly more data

- Identify and remove bots

- Incorporate frequency of interactions & NLP into the feature engineering

- Experiment with more granular target classes ie post, comment, both, neither

- Use different target subreddits
