#=================================================
#   ML_HW_Q2_Probability_theorem_and_Bayes_la
#   Bayes Theorem
#   Foad Moslem (foad.moslem@gmail.com) - Researcher | Aerodynamics
#   Using Python 3.9.16
#=================================================

# %clear
# %reset -f

#%%============================
""" Import Libraries """
# import pandas as pd # Import Pandas Library
# import numpy as np # Import Numpy Library
# from scipy.stats import multivariate_normal

#%%============================
""" Load Dataset """
# Import The Dataset (covid19.csv) CSV File Into Python Using Pandas
import pandas as pd # Import Pandas Library
dataset = pd.read_csv("./covid19.csv") # Read dataset's csv file
print(dataset) # Present data to console
dataset.head() # preview the top 5 rows of dataset
dataset.info() # view summary of dataset

#%%============================
""" Segregate the dataset into categorical and numerical variables """
# Find categorical & numerical variables
categorical = [var for var in dataset.columns if dataset[var].dtype=='O']
numerical = [var for var in dataset.columns if dataset[var].dtype!='O']
print('There are {} categorical variables.\nThe categorical variables are:\n{}'.format(len(categorical), categorical))
print('There are {} numerical variables.\nThe numerical variables are:\n{}'.format(len(numerical), numerical))

#%%============================
""" Check Missing Values """
dataset[categorical].isnull().sum() # check missing values in categorical variables
dataset[numerical].isnull().sum() # check missing values in numerical variables

#%%============================
# """ New Dataset 0 """
# """ Drop All Columns with NaN Values [New Dataset 0: df0] """
# df0 = dataset.dropna(axis=1).reset_index(drop=True) # Reset index after drop
# df0.to_csv(r'.\df0.csv', index=False) # Save new dataset
# print(df0)
# df = df0

#%%============================
# """ New Dataset 1 """
# """ Drop All Rows with NaN Values [New Dataset 1: df1] """
# df1 = dataset.dropna().reset_index(drop=True) # Reset index after drop
# df1.to_csv(r'.\df1.csv', index=False) # Save new dataset
# print(df1)
# df = df1

#%%============================
""" New Dataset 2 """
""" Drop Rows Based on NaN Percentage in each Column [New Dataset 2: df2] """

# Delete columns based on percentage of NaN values in columns
perc = 20.0 # Acceptable percentage of NaN values in each column
min_count = int(((100-perc)/100)*dataset.shape[0] + 1) # columns which contains less than min_count number of non-NaN values.
dff = dataset.dropna(axis=1, thresh=min_count).reset_index(drop=True) # Drop columns which contain missing value

# Drop Rows of new dataset with NaN Values
df2 = dff.dropna().reset_index(drop=True) # Reset index after drop
df2.to_csv(r'.\df2.csv', index=False) # Save new dataset
print(df2)
df = df2

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%============================
""" Declare feature vector and target variable """
df_features = df.drop(['ID', 'target'], axis=1) # Declare features (Numerical)
df_target = df['target'] # Declare target
print(df_features)
print(df_target)

#%%============================
""" Split Dataset Into Features/Target and Train/Test Sets """
# 1- Using Pandas
df_features_train = df_features.sample(frac=0.7, random_state=42) # Return a random sample of features_train
df_features_test = df_features.drop(df_features_train.index) # Drop specified lables from rows or columns

df_target_train = df_target.sample(frac=0.7, random_state=42) # Return a random sample of target_train
df_target_test = df_target.drop(df_target_train.index) # Drop specified lables from rows or columns

# 2- Using scikit-learn
# from sklearn.model_selection import train_test_split # Create both the training and testings sets in a one-liner by passing to train_test_split()
# df_features_train, df_features_test, df_target_train, df_target_test = train_test_split(df_features, df_target, test_size=0.3, shuffle=True, random_state=42)

print(f"No. of training examples: {df_features_train.shape[0]}")
print(f"No. of testing examples: {df_features_test.shape[0]}")


#%%============================
""" Make a Gaussian multivariate PDF for the Training dataset """
import numpy as np # Import Numpy Library
from scipy.stats import multivariate_normal

# Estimate the mean and covariance matrix from the training dataset
df_mu_features_train = np.mean(df_features_train, axis=0) # find out the estimated mean with respect columns
df_cov_features_train = np.cov(df_features_train.T) # find out covariance with respect columns

# Define a multivariate Gaussian PDF using the estimated mean and covariance
df_mvn_features_train = multivariate_normal(mean = df_mu_features_train, cov = df_cov_features_train)

# Evaluate the PDF at some test points to check if it looks reasonable
df_pdf_vals_features = df_mvn_features_train.pdf(df_features_test)
print("PDF values at test points:\n", df_pdf_vals_features)

#%%============================
""" Maximum Likelihood Estimation """
""" Implementing a Gaussian classifier with a uniform probability 
    distribution assumption for all classes using and calculate 
    this classifier accuracy with the Train and the Test dataset. """

class GaussianClassifier():
    
    def fit(self, X_train, y_train): # self: represents the instance of class
        self.classes = np.unique(y_train) # unique: Find the unique elements of an array
        self.mean = []
        self.var = []
        self.prior = []
        for c in self.classes:
            X_c = X_train[y_train == c]
            self.mean.append(X_c.mean(axis=0))
            self.var.append(X_c.var(axis=0))
            self.prior.append(X_c.shape[0] / X_train.shape[0])
    
    def predict(self, X_test):
        probs = np.zeros((X_test.shape[0], len(self.classes)))
        for i, c in enumerate(self.classes): # Enumerate is a built-in function in python that allows you to keep track of the number of iterations (loops) in a loop.
            probs[:,i] = self.calculate_prob(X_test, self.mean[i], self.var[i], self.prior[i])
            return self.classes[np.argmax(probs, axis=1)]
    
    def calculate_prob(self, X, mean, var, prior):
        exponent = np.exp(-((X-mean)**2 / (2 * var)))
        return np.prod(exponent, axis=1) * prior
    
    def calculate_accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

# Train Gaussian classifier
clf = GaussianClassifier()
clf.fit(df_features_train, df_target_train)

# Predict using train and test dataset
df_target_train_pred = clf.predict(df_features_train)
df_target_test_pred = clf.predict(df_features_test)
print('Predicted:', df_target_train_pred)
print('True:', df_target_test_pred)

# Calculate accuracy
train_accuracy = clf.calculate_accuracy(df_target_train, df_target_train_pred)
test_accuracy = clf.calculate_accuracy(df_target_test, df_target_test_pred)
print("Train accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)
