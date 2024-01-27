from flask import Flask, render_template
from distutils.log import debug
from fileinput import filename
from flask import *
import pandas as pd
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np  # linear algebra
import matplotlib.pyplot as plt  # visualization
import seaborn as sns  # visualization

app = Flask(__name__)

s_data = None


def clean_data(df):
    df.dropna(inplace=True)
    # Aykırı değerleri kontrol et ve düzelt
    numeric_columns = df.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].apply(lambda x: lower_bound if x < lower_bound else (
            upper_bound if x > upper_bound else x))

    # Tekrar eden değerleri kontrol et
    df.drop_duplicates(inplace=True)

    return df


def save_confusion_matrix(y_test, y_pred, filename="confusion_matrix.png"):
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(conf_mat, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'static/{filename}')
    plt.close()


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        f = request.files['file']
        file_contents = f.read().decode('utf-8')
        # Check if the file is empty
        if not file_contents.strip():
            return "The uploaded file is empty", 400
        df = pd.read_csv(StringIO(file_contents))
        # Ön işleme işlemlerini gerçekleştir
        df_cleaned = clean_data(df)
        # Veri setinin sütun adlarını al
        column_names = df_cleaned.columns.tolist()

        cleaned_data_string = df_cleaned.to_csv(
            index=False)  # DataFrame'i string'e çevirme
        # Veri setini string olarak sakla
        return render_template('index.html',  data=df_cleaned.head(10).to_html(),  column_names=column_names, selected_data=cleaned_data_string)

    return render_template('index.html', data=None)


@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':
        selected_features = request.form.getlist('features')
        selected_target = request.form.getlist('target')[0]
        selected_model = request.form.getlist('model')[0]
        selected_test_size = request.form['traintest']
        print("selected", selected_test_size)
        # Gizli inputtan veriyi alma
        cleaned_data_string = request.form['selected_data']
        # String'i DataFrame'e dönüştürme
        df_cleaned = pd.read_csv(StringIO(cleaned_data_string))
        target_column = df_cleaned[selected_target]
        label_encoder = LabelEncoder()
        target_column = label_encoder.fit_transform(target_column)

        X = df_cleaned[selected_features]
        y = df_cleaned[selected_target]
        if selected_model == 'RF':
            model = RandomForestClassifier()
        elif selected_model == 'DT':
            model = DecisionTreeClassifier()
        elif selected_model == 'SVM':
            model = SVC()
        elif selected_model == 'LR':
            model = LogisticRegression()
        elif selected_model == 'KNN':
            model = KNeighborsClassifier()
        else:
            return "Invalid Model Selection", 400

        if 'kfold' in selected_test_size:
            k = int(selected_test_size.split('-')[1])
            if k > 1:
                cv = StratifiedKFold(
                    n_splits=k) if 'Stratified' in selected_model else KFold(n_splits=k)
                scores = cross_val_score(model, X, y, cv=cv)
                avg_score = np.mean(scores)
                # Sonuçları train.html'e gönder
                return render_template('train.html', cv_scores=scores, avg_score=avg_score, selected_model=selected_model)
            else:
                return "Invalid K-fold value", 400
        else:

            test_size = float(selected_test_size)
            # Modeli eğitme ve performans metriklerini hesaplama
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Sonuçları train.html'e gönder ve grafikleri kaydet
            # save_confusion_matrix(y_test, y_pred)

            return render_template('train.html', accuracy=accuracy, class_report=class_report, conf_matrix=conf_matrix, selected_features=selected_features, selected_target=selected_target, selected_model=selected_model, confusion_matrix_url='/static/confusion_matrix.png')

    return render_template('train.html', selected_features=None, selected_target=None, selected_model=None, selected_performance=None)


if __name__ == '__main__':
    app.run(debug=True)
