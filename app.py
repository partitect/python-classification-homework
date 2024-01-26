from flask import Flask, render_template
from distutils.log import debug
from fileinput import filename
from flask import *
import pandas as pd
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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


def svm_classification(X, y, size):
    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=size, random_state=42)

    # Destek Vektör Makineleri modelini oluştur
    model = SVC(kernel='linear', C=1)

    # Modeli eğit
    model.fit(X_train, y_train)

    # Test seti üzerinde tahmin yap
    y_pred = model.predict(X_test)

    # Doğruluk (accuracy) değerini hesapla
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


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
        # Veri setinin özet bilgilerini al
        data_summary = df_cleaned.describe(include='all').to_html()
        # Veri setinin sütun adlarını al
        column_names = df_cleaned.columns.tolist()
        # Veri setindeki ilk 10 satırı döndür
        selected_data = file_contents
        return render_template('index.html',  data=df_cleaned.head(10).to_html(), summary=data_summary, column_names=column_names, selected_features=None, selected_data=selected_data)

    return render_template('index.html', data=None, summary=None)


@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':
        selected_features = request.form.getlist('features')
        selected_target = request.form.getlist('target')
        selected_model = request.form.getlist('model')
        selected_performance = request.form.getlist('traintest')
        # Veri setini yükle
        f = request.files['selected_data']
        file_contents = f.read().decode('utf-8')
        df = pd.read_csv(StringIO(file_contents))
        # Ön işleme işlemlerini gerçekleştir
        df_cleaned = clean_data(df)
        return render_template('train.html',  selected_features=selected_features, selected_target=selected_target, selected_model=selected_model, selected_performance=selected_performance, selected_data=df_cleaned)

    return render_template('train.html', selected_features=None, selected_target=None, selected_model=None, selected_performance=None)


if __name__ == '__main__':
    app.run(debug=True)
