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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm  # statsmodels kütüphanesini ekleyin

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

        categorical_features = df_cleaned[selected_features].select_dtypes(include=['object']).columns.tolist()
        for col in categorical_features:
            df_cleaned[col] = LabelEncoder().fit_transform(df_cleaned[col])
            
        # Hedef sütunu sayısallaştır
        df_cleaned[selected_target] = LabelEncoder().fit_transform(df_cleaned[selected_target])
        
        X = df_cleaned[selected_features]
        y = df_cleaned[selected_target]
        if selected_model == 'Random Forests':
            model = RandomForestClassifier()
        elif selected_model == 'Decision Trees':
            model = DecisionTreeClassifier()
        elif selected_model == 'Support Vector Machines':
            model = SVC()
        elif selected_model == 'Lojistik Regresyon':
            model = LogisticRegression()
        elif selected_model == 'K-Nearest Neighbors':
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
                return render_template('train.html',selected_features=selected_features, selected_target=selected_target, cv_scores=scores, avg_score=avg_score, selected_model=selected_model,selected_performance=selected_test_size)
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
            # Grafikleri oluştur ve kaydet
            conf_matrix_path = f'static/conf_matrix_{selected_model}.png'
            accuracy_path = f'static/accuracy_{selected_model}.png'
            class_report_precision_path = f'static/class_report_precision_{selected_model}.png'
            report = classification_report(y_test, y_pred, output_dict=True)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix ({selected_model})')
            plt.savefig(conf_matrix_path)
            plt.close()

            # Doğruluk grafiği
            plt.figure(figsize=(6, 4))
            sns.barplot(x=[selected_model], y=[accuracy])
            plt.title('Accuracy Score')
            plt.ylim(0, 1)
            plt.savefig(accuracy_path)
            plt.close()

            # Sınıf etiketlerini al (genel puanları hariç tut)
            class_labels = [key for key in report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]

            # Sınıflandırma raporu grafiği için sadece sınıfların 'precision' değerlerini kullan
            plt.figure(figsize=(10, 6))
            sns.barplot(x=class_labels, y=[report[key]['precision'] for key in class_labels])
            plt.title('Classification Report - Precision')
            plt.savefig(class_report_precision_path)
            plt.close()

            # Sonuçları train.html'e gönder ve grafikleri kaydet
            # save_confusion_matrix(y_test, y_pred)
            print("conf_matrix_path", conf_matrix_path)
            return render_template('train.html', accuracy=accuracy, class_report=class_report, conf_matrix=conf_matrix, selected_features=selected_features, selected_target=selected_target, selected_model=selected_model, conf_matrix_path=conf_matrix_path,accuracy_path=accuracy_path,class_report_precision_path=class_report_precision_path,selected_performance=selected_test_size)

    return render_template('train.html', selected_features=None, selected_target=None, selected_model=None, selected_performance=None)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Formdan verileri al
        selected_features = request.form.getlist('features')
        selected_target = request.form.getlist('target')[0]
        selected_model = request.form.getlist('model')[0]
        selected_split_ratio = request.form['split_ratio']
        
        # Gizli inputtan veriyi alma ve DataFrame'e dönüştürme
        cleaned_data_string = request.form['selected_data']
        df_cleaned = pd.read_csv(StringIO(cleaned_data_string))

        # Kategorik özellikleri sayısallaştır
        categorical_features = df_cleaned[selected_features].select_dtypes(include=['object']).columns.tolist()
        for col in categorical_features:
            df_cleaned[col] = LabelEncoder().fit_transform(df_cleaned[col])
        
        # Hedef sütunu sayısallaştır
        df_cleaned[selected_target] = LabelEncoder().fit_transform(df_cleaned[selected_target])

        # Veri setini özellikler ve hedef olarak ayır
        X = df_cleaned[selected_features]
        y = df_cleaned[selected_target]

        # Model seçimi
        if selected_model == 'Lineer Regresyon':
            model = LinearRegression()
        elif selected_model == 'Ridge Regresyonu':
            model = Ridge()
        elif selected_model == 'Lasso Regresyonu':
            model = Lasso()
        elif selected_model == 'Karar Ağacı Regresyonu':
            model = DecisionTreeRegressor()
        elif selected_model == 'Rastgele Orman Regresyonu':
            model = RandomForestRegressor()
        # Buraya diğer regresyon modellerinizin seçimi eklenebilir
        else:
            return "Invalid Model Selection", 400

        # Veri setini eğitim ve test setlerine ayırma
        test_size = float(selected_split_ratio)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Modeli eğit ve tahmin yap
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Performans metriklerini hesapla
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        scatter_path = f'static/scatter_{selected_model}.png'
        error_path = f'static/error_{selected_model}.png'

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred)
        plt.xlabel('Gerçek Değerler')
        plt.ylabel('Tahmin Edilen Değerler')
        plt.title('Gerçek Değerler vs Tahmin Edilen Değerler')
        plt.savefig(scatter_path)
        plt.close()

        # Hata Grafiği
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.residplot(x=y_pred, y=residuals, lowess=True, color="g")
        plt.xlabel('Tahmin Edilen Değerler')
        plt.ylabel('Kalanlar (Residuals)')
        plt.title('Hata Grafiği')
        plt.savefig(error_path)
        plt.close()
        # Performans metriklerini ve tahminleri train.html'e gönder
        return render_template('predict.html', mse=mse, r2=r2, y_pred=y_pred, selected_features=selected_features, selected_target=selected_target, selected_model=selected_model,selected_performance=test_size , selected_split_ratio=selected_split_ratio, scatter_path=scatter_path, error_path=error_path)

    return render_template('predict.html', selected_features=None, selected_target=None, selected_model=None, selected_performance=None)


if __name__ == '__main__':
    app.run(debug=True)
