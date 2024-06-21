#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("Training.csv").dropna(axis = 1)
 
# Checking whether the dataset is balanced or not
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})
 
plt.figure(figsize = (18,8))
sns.barplot(x = "Disease", y = "Counts", data = temp_df)
plt.xticks(rotation=90)
plt.show()


# In[3]:


data = pd.read_csv("Training.csv").dropna(axis = 1)
 
# Checking whether the dataset is balanced or not
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})
 
plt.figure(figsize = (18,8))
sns.barplot(x = "Disease", y = "Counts", data = temp_df)
plt.xticks(rotation=90)
plt.show()


# In[4]:


encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])


# In[5]:


X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test =train_test_split(
  X, y, test_size = 0.2, random_state = 24)
 
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")


# In[6]:


def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))
 
# Initializing Models
models = {
    "SVC":SVC(),
    "Gaussian NB":GaussianNB(),
    "Random Forest":RandomForestClassifier(random_state=18)
}
 
# Producing cross validation score for the models
for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, X, y, cv = 10, 
                             n_jobs = -1, 
                             scoring = cv_scoring)
    print("=="*30)
    print(model_name)
    print(f"Scores: {scores}")
    print(f"Mean Score: {np.mean(scores)}")


# In[7]:


svm_model = SVC()
svm_model.fit(X_train, y_train)
preds = svm_model.predict(X_test)
 
print(f"Accuracy on train data by SVM Classifier\
: {accuracy_score(y_train, svm_model.predict(X_train))*100}")
 
print(f"Accuracy on test data by SVM Classifier\
: {accuracy_score(y_test, preds)*100}")
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for SVM Classifier on Test Data")
plt.show()
 
# Training and testing Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
preds = nb_model.predict(X_test)
print(f"Accuracy on train data by Naive Bayes Classifier\
: {accuracy_score(y_train, nb_model.predict(X_train))*100}")
 
print(f"Accuracy on test data by Naive Bayes Classifier\
: {accuracy_score(y_test, preds)*100}")
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Naive Bayes Classifier on Test Data")
plt.show()
 
# Training and testing Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_test)
print(f"Accuracy on train data by Random Forest Classifier\
: {accuracy_score(y_train, rf_model.predict(X_train))*100}")
 
print(f"Accuracy on test data by Random Forest Classifier\
: {accuracy_score(y_test, preds)*100}")
 
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Random Forest Classifier on Test Data")
plt.show()


# In[8]:


final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)
 
# Reading the test data
test_data = pd.read_csv("Testing.csv").dropna(axis=1)
 
test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])
 
# Making prediction by take mode of predictions 
# made by all the classifiers
svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X)
 
final_preds = [mode([i,j,k])[0][0] for i,j,
               k in zip(svm_preds, nb_preds, rf_preds)]
 
print(f"Accuracy on Test dataset by the combined model\
: {accuracy_score(test_Y, final_preds)*100}")
 
cf_matrix = confusion_matrix(test_Y, final_preds)
plt.figure(figsize=(12,8))
 
sns.heatmap(cf_matrix, annot = True)
plt.title("Confusion Matrix for Combined Model on Test Dataset")
plt.show()


# In[9]:


symptoms = X.columns.values
 
# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index
 
data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":encoder.classes_
}
 
# Defining the Function
# Input: string containing symptoms separated by commas
# Output: Generated predictions by models
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
     
    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
         
    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
     
    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
     
    # making final prediction by taking mode of all predictions
    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction":final_prediction
    }
    return predictions
 
# Testing the function
print(predictDisease("Joint Pain,Stomach Pain,Acidity"))


# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
data = pd.read_csv("Testing.csv").dropna(axis=1)

# Checking whether the dataset is balanced or not
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})

plt.figure(figsize=(18, 8))
sns.barplot(x="Disease", y="Counts", data=temp_df)
plt.xticks(rotation=90)
plt.show()

# Encode the target variable
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

# Split data into train and test sets
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24)

# Initialize the SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Make predictions on the test set
preds = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, preds)
print("Accuracy:", accuracy)


# In[ ]:





# In[11]:


from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision, recall, and F1-score for SVM predictions
precision_svm = precision_score(test_Y, svm_preds, average='weighted')
recall_svm = recall_score(test_Y, svm_preds, average='weighted')
f1_score_svm = f1_score(test_Y, svm_preds, average='weighted')
print("For SVM model:")

print("Recall:", recall_svm)

print()

# Calculate precision, recall, and F1-score for Naive Bayes predictions
precision_nb = precision_score(test_Y, nb_preds, average='weighted')
recall_nb = recall_score(test_Y, nb_preds, average='weighted')
f1_score_nb = f1_score(test_Y, nb_preds, average='weighted')

print("For Naive Bayes model:")
print("Recall:", recall_nb)

print()

# Calculate precision, recall, and F1-score for Random Forest predictions
precision_rf = precision_score(test_Y, rf_preds, average='weighted')
recall_rf = recall_score(test_Y, rf_preds, average='weighted')
f1_score_rf = f1_score(test_Y, rf_preds, average='weighted')

print("For Random Forest model:")
print("Recall:", recall_rf)

print()

# Calculate precision, recall, and F1-score for final predictions
precision_final = precision_score(test_Y, final_preds, average='weighted')
recall_final = recall_score(test_Y, final_preds, average='weighted')
f1_score_final = f1_score(test_Y, final_preds, average='weighted')

print("For Final model:")
print("Recall:", recall_final)


# In[12]:


import matplotlib.pyplot as plt

# List of model names
models = ['SVM', 'Naive Bayes', 'Random Forest', 'Final Combined']

# Corresponding F1-scores
f1_scores = [f1_score_svm, f1_score_nb, f1_score_rf, f1_score_final]

# Create bar plot
plt.figure(figsize=(10, 6))
plt.bar(models, f1_scores, color=['blue', 'orange', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('F1-score')
plt.title('Comparison of F1-score for Different Models')
plt.ylim(0, 1)  # Set y-axis limit between 0 and 1
plt.show()


# In[ ]:





# In[13]:


from sklearn.metrics import precision_score

# Calculate precision for SVM predictions
precision_svm = precision_score(test_Y, svm_preds, average='weighted')

# Calculate precision for Naive Bayes predictions
precision_nb = precision_score(test_Y, nb_preds, average='weighted')

# Calculate precision for Random Forest predictions
precision_rf = precision_score(test_Y, rf_preds, average='weighted')

# Calculate precision for final combined predictions
precision_final = precision_score(test_Y, final_preds, average='weighted')

print("Precision for SVM model:", precision_svm)
print("Precision for Naive Bayes model:", precision_nb)
print("Precision for Random Forest model:", precision_rf)
print("Precision for Final Combined model:", precision_final)


# In[14]:


import matplotlib.pyplot as plt

# List of model names
models = ['SVM', 'Naive Bayes', 'Random Forest', 'Final Combined']

# Corresponding precision scores
precision_scores = [precision_svm, precision_nb, precision_rf, precision_final]

# Create bar plot
plt.figure(figsize=(10, 6))
plt.bar(models, precision_scores, color=['blue', 'orange', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Precision')
plt.title('Comparison of Precision for Different Models')
plt.ylim(0, 1)  # Set y-axis limit between 0 and 1
plt.show()


# In[ ]:





# In[15]:


import numpy as np

def dice_index(y_true, y_pred):
    intersection = np.sum(y_true * y_pred, axis=0)
    cardinality_true = np.sum(y_true, axis=0)
    cardinality_pred = np.sum(y_pred, axis=0)
    dice = (2. * intersection) / (cardinality_true + cardinality_pred)
    return dice

# Calculate Sørensen-Dice index for SVM predictions
dice_svm = dice_index(test_Y, svm_preds)

# Calculate Sørensen-Dice index for Naive Bayes predictions
dice_nb = dice_index(test_Y, nb_preds)

# Calculate Sørensen-Dice index for Random Forest predictions
dice_rf = dice_index(test_Y, rf_preds)

# Calculate Sørensen-Dice index for final combined predictions
dice_final = dice_index(test_Y, final_preds)

print("Sørensen-Dice Index for SVM model:", np.mean(dice_svm))
print("Sørensen-Dice Index for Naive Bayes model:", np.mean(dice_nb))
print("Sørensen-Dice Index for Random Forest model:", np.mean(dice_rf))
print("Sørensen-Dice Index for Final Combined model:", np.mean(dice_final))


# In[ ]:





# In[ ]:





# In[16]:


def jaccard_index(y_true, y_pred):
    intersection = np.sum(y_true * y_pred, axis=0)
    union = np.sum((y_true + y_pred) > 0, axis=0)
    jaccard = intersection / union
    return jaccard

# Calculate Jaccard index for SVM predictions
jaccard_svm = jaccard_index(test_Y, svm_preds)

# Calculate Jaccard index for Naive Bayes predictions
jaccard_nb = jaccard_index(test_Y, nb_preds)

# Calculate Jaccard index for Random Forest predictions
jaccard_rf = jaccard_index(test_Y, rf_preds)

# Calculate Jaccard index for final combined predictions
jaccard_final = jaccard_index(test_Y, final_preds)

print("Jaccard Index for SVM model:", np.mean(jaccard_svm))
print("Jaccard Index for Naive Bayes model:", np.mean(jaccard_nb))
print("Jaccard Index for Random Forest model:", np.mean(jaccard_rf))
print("Jaccard Index for Final Combined model:", np.mean(jaccard_final))


# In[17]:


tn_svm, fp_svm, fn_svm, tp_svm = confusion_matrix(test_Y, svm_preds).ravel()
tn_nb, fp_nb, fn_nb, tp_nb = confusion_matrix(test_Y, nb_preds).ravel()
tn_rf, fp_rf, fn_rf, tp_rf = confusion_matrix(test_Y, rf_preds).ravel()
tn_final, fp_final, fn_final, tp_final = confusion_matrix(test_Y, final_preds).ravel()

# Calculate specificity for each model
specificity_svm = tn_svm / (tn_svm + fp_svm)
specificity_nb = tn_nb / (tn_nb + fp_nb)
specificity_rf = tn_rf / (tn_rf + fp_rf)
specificity_final = tn_final / (tn_final + fp_final)

print("Specificity for SVM model:", specificity_svm)
print("Specificity for Naive Bayes model:", specificity_nb)
print("Specificity for Random Forest model:", specificity_rf)
print("Specificity for Final Combined model:", specificity_final)

# Create a bar plot to compare specificity of different models
models = ['SVM', 'Naive Bayes', 'Random Forest', 'Final Combined']
specificities = [specificity_svm, specificity_nb, specificity_rf, specificity_final]

plt.figure(figsize=(10, 6))
plt.bar(models, specificities, color=['blue', 'orange', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Specificity')
plt.title('Comparison of Specificity for Different Models')
plt.ylim(0, 1)  # Set y-axis limit between 0 and 1
plt.show()


# In[18]:


from sklearn.metrics import confusion_matrix

# Compute confusion matrix for each model
cm_svm = confusion_matrix(test_Y, svm_preds)
cm_nb = confusion_matrix(test_Y, nb_preds)
cm_rf = confusion_matrix(test_Y, rf_preds)
cm_final = confusion_matrix(test_Y, final_preds)

# Extract TN, FP, FN, TP from each confusion matrix
tn_svm, fp_svm, fn_svm, tp_svm = cm_svm[0, 0], cm_svm[0, 1], cm_svm[1, 0], cm_svm[1, 1]
tn_nb, fp_nb, fn_nb, tp_nb = cm_nb[0, 0], cm_nb[0, 1], cm_nb[1, 0], cm_nb[1, 1]
tn_rf, fp_rf, fn_rf, tp_rf = cm_rf[0, 0], cm_rf[0, 1], cm_rf[1, 0], cm_rf[1, 1]
tn_final, fp_final, fn_final, tp_final = cm_final[0, 0], cm_final[0, 1], cm_final[1, 0], cm_final[1, 1]

# Calculate specificity for each model
specificity_svm = tn_svm / (tn_svm + fp_svm)
specificity_nb = tn_nb / (tn_nb + fp_nb)
specificity_rf = tn_rf / (tn_rf + fp_rf)
specificity_final = tn_final / (tn_final + fp_final)

print("Specificity for SVM model:", specificity_svm)
print("Specificity for Naive Bayes model:", specificity_nb)
print("Specificity for Random Forest model:", specificity_rf)
print("Specificity for Final Combined model:", specificity_final)

# Create a bar plot to compare specificity of different models
models = ['SVM', 'Naive Bayes', 'Random Forest', 'Final Combined']
specificities = [specificity_svm, specificity_nb, specificity_rf, specificity_final]

plt.figure(figsize=(10, 6))
plt.bar(models, specificities, color=['blue', 'orange', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Specificity')
plt.title('Comparison of Specificity for Different Models')
plt.ylim(0, 1)  # Set y-axis limit between 0 and 1
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




