import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#GridSearchCV - performs hyperparameter tuning 
#StratifiedKFold - ensures equal class distribution in cross-validation
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

#StandardScaler - used for feature scaling 
#LabelEncoder - converts categorical labels into numeric form
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load dataset 
df = pd.read_csv("/Users/sec23cb069/Downloads/Iris.csv").drop(columns=["Id"], errors="ignore")

# Encode target
le = LabelEncoder()
# fit() - learns all unique classes, transform() - converts them into numbers
df["Species"] = le.fit_transform(df["Species"])

# Split features & target
# X - input features (sepal & petal measurements) 
# y - target class (species)
X = df.drop("Species", axis=1)
y = df["Species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=15, stratify=y
)

# Feature Scaling
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

#Function to train and evaluate
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train) #training
    y_pred = model.predict(X_test) #testing

    print(f"\n{model_name}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    report = classification_report(y_test, y_pred, output_dict=True)
    
    return [model_name, report['accuracy'], report['weighted avg']['precision'],
            report['weighted avg']['recall'], report['weighted avg']['f1-score']] #list

# Models 
models = [
    ("Naive Bayes", GaussianNB()),

    ("Logistic Regression", LogisticRegression()),

    ("Tuned Logistic Regression", GridSearchCV(
        LogisticRegression(), 
        param_grid={"penalty":["l1","l2"], "C":[0.1,1,10], "solver":["liblinear","saga"]},
        cv=StratifiedKFold(5), scoring='accuracy', n_jobs=-1)),

    ("SVM", SVC()),

    ("Tuned SVM", GridSearchCV(
        SVC(),
        param_grid={"kernel":["linear","rbf"], "C":[0.1,1,10], "gamma":["scale","auto"]},
        cv=5, scoring='accuracy', n_jobs=-1)),

    ("KNN", KNeighborsClassifier(n_neighbors=5)),

    ("Decision Tree", DecisionTreeClassifier(random_state=42)),

    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42))
]

results = []
for name, model in models:
    res = evaluate_model(model, X_train, y_train, X_test, y_test, name)
    results.append(res)

#Comparison report
summary = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
print("\nClassification Report")
print(summary)
