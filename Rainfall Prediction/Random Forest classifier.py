import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

#Load data
url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
df.head()  #Display the first few rows of the dataset
df.count() #Check for missing values in each column and data count each feature to remove those with too many missing values

df = df.dropna() #Drop rows with missing values
df.info() #Check the data types of each column and ensure they are appropriate for modeling

df.columns #Check the column names to identify categorical and numerical features

df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'}) # to avoid data leackage doing this to make sure that the model is not using future data to predict the target variable
# Example in data set we have RainTomorrow target and few features like temp3pm, Humidity3pm, WindSpeed3pm which are recorded at 3pm and can be used to predict RainTomorrow but 
# what if we want to predict early morning then model will assume future vlaue and can be bais with prediction.


# Now need to select locations in australia because each location has different weather pattern and we want to make sure that our model is not biased towards one location and can generalize well to other locations.
# We can select few locations and filter the data accordingly.

df = df[df['Location'].isin(['Melbourne','MelbourneAirport','Watsonia',])] # Filter the data for selected locations
df.info() # Check the data after filtering to ensure we have enough data for modeling

# Creating season wise function which replicated by months
def get_season(month):
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'
    
# convert date column to datetime format and extract month and season
# removing date feature and maping it with season feature because date feature is not useful for modeling and can lead to overfitting as it has too many unique values and can capture noise in the data.
df['Date'] = pd.to_datetime(df['Date'])
df['Season'] = df['Date'].apply(get_season)
df = df.drop(columns=['Date']) # Drop the date and month columns as they are not useful for modeling    

#Check how balanced our target is
X = df.drop(columns=['RainToday'], axis=1) # Features RainToday is droping from input becuase it is target value.
y = df['RainToday'] # Target

X.value_counts() # Check the distribution of the target variable to see if it is balanced or imbalanced
y.value_counts() # Check the distribution of the target variable to see if it is balanced or imbalanced

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify= y, random_state= 42)

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['string','object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Define separate transformers for both feature types
numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# now need to combine these transformers into a single preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# a pipeline by combining the preprocessing with a Random Forest classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# Use StratifiedKFold for cross-validation to maintain the distribution of the target variable in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Instantiate and fit GridSearchCV to the pipeline
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Display your model's estimated score
test_score = grid_search.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))

y_pred = grid_search.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


# Extract the feature importances
feature_importances = grid_search.best_estimator_['classifier'] .feature_importances_

# Combine numeric and categorical feature names
feature_names = numerical_features + list(grid_search.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

N = 20  # Change this number to display more or fewer features
top_features = importance_df.head(N)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
plt.title(f'Top {N} Most Important Features in predicting whether it will rain today')
plt.xlabel('Importance Score')
plt.show()