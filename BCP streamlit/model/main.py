import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import pickle


def get_clean_data():
    data=pd.read_csv('data/data.csv')
    
    data = data.drop(['Unnamed: 32','id'],axis=1)

    data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
    return data

def create_model(data):
    X=data.drop(['diagnosis'],axis=1)
    y=data['diagnosis']

    # split the data — no scaling needed, RandomForest is a tree-based
    # model and is invariant to feature scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Hyperparameter tuning was done once via GridSearchCV (kept here for reference) ---
    # param_grid = {
    #     'n_estimators': [100, 200],
    #     'max_depth': [None, 10, 20],
    #     'min_samples_split': [2, 5],
    #     'min_samples_leaf': [1, 2]
    # }
    # rf = RandomForestClassifier(random_state=42)
    # grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,scoring='accuracy',n_jobs=-1,verbose=2)
    # grid_search.fit(X_train,y_train)
    # model = grid_search.best_estimator_
    # print('Best parameters found:',grid_search.best_params_)

    # Best params found by the search above — used directly now to skip
    # re-running the full grid search on every training run.
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    model.fit(X_train,y_train)

    #test 
    y_pred = model.predict(X_test)
    print('Accuracy of our model:',accuracy_score(y_test,y_pred))
    print('Classification report:\n',classification_report(y_test,y_pred))

    # 5-fold cross-validation on the full dataset — gives a more robust,
    # less-lucky/unlucky estimate than a single 80/20 split. Good number
    # to actually quote in your review since it's averaged over 5 splits.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f'Cross-validated accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})')

    #feature importance
    feature_names = data.drop(['diagnosis'],axis=1).columns
    importances = pd.Series(model.feature_importances_,index=feature_names).sort_values(ascending=False)
    print('Feature importances:\n',importances)

    return model




def main():
    data = get_clean_data()
    model = create_model(data)

    with open('model/model.pkl','wb')as f:
        pickle.dump(model,f)
    
if __name__ == '__main__':
    main()