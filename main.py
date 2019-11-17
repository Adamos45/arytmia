from symbols_enums import FeatureNames
from feature_extraction import prepare_data, read_data
from train_models import train_models


def main():
    files = [FeatureNames.mvsk, FeatureNames.db5,
             FeatureNames.db6, FeatureNames.diffs]

    prepare_data(files)
    data = read_data(files)
    train_models(data, files)
    
    model = svm.SVC(C=0.8, kernel='rbf', degree=4, gamma='auto',
                        coef0=0.0, shrinking=True, probability=False, tol=0.001,
                        cache_size=200, class_weight='balanced', verbose=False,
                        max_iter=-1, decision_function_shape='ovo', random_state=1)
     
    model = DecisionTreeClassifier(class_weight='balanced')

if __name__ == "__main__":
    main()
