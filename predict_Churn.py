import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Loads data into a DataFrame from a string filepath.
    """
    df = pd.read_csv(filepath, index_col='customerID')
    return df


def make_predictions(trainingdata,testdata):

    trainingdata.rename(columns = {'TotalCharges_by_tenure':'charge_per_tenure'}, inplace = True)
    trainingdata.drop('TotalCharges_by_tenure_log', axis=1, inplace=True)
    trainingdata.rename(columns = {'Churn':'target'}, inplace = True)
    features = trainingdata.drop('target', axis=1)  
    
    tpot_data = trainingdata.copy()
    
    training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

    # Average CV score on the training set was: 0.8017091741983411
    exported_pipeline = RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=.5, \
                                               min_samples_leaf=8, min_samples_split=4, n_estimators=100)
    # Fix random state in exported estimator
    if hasattr(exported_pipeline, 'random_state'):
        setattr(exported_pipeline, 'random_state', 3)

    #This fits our training data
    exported_pipeline.fit(training_features, training_target)
    
    #This fits our test data 
    testing_features = testdata
    for row in range(len(testing_features)):
        results = exported_pipeline.predict_proba(testing_features)
        prob = (results[row][0]*100).round(decimals = 2)
        print(f'There is a {prob} probability for customer {testing_features.index[row]} to Churn' )


if __name__ == "__main__":
    trainingdata = load_data('../week2/prepped_churned_data.csv')
    testdata = load_data('new_churn_data.csv')
    print('predictions:')
    predictions = make_predictions(trainingdata,testdata)
    
