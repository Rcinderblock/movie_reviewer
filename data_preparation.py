import os
import re
import nltk
import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
# Скачать необходимые пакеты NLTK
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(data_dir, labeled=True):
    texts = []
    labels = []
    ratings = []

    label_dirs = ['pos', 'neg'] if labeled else ['pos', 'neg', 'unsup']

    for label in label_dirs:
        dir_path = os.path.join(data_dir, label)
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(dir_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    texts.append(file.read())

                if labeled:
                    # Извлечение рейтинга из имени файла
                    try:
                        _, rating_str = filename.split('_')
                        rating = int(rating_str.replace('.txt', ''))
                        ratings.append(rating)
                    except ValueError:
                        # Если не удалось извлечь рейтинг, пропустить файл
                        continue

                    # Определение метки
                    if label == 'pos':
                        labels.append(1)
                    elif label == 'neg':
                        labels.append(0)
                else:
                    # Для несупервизируемого набора метка не нужна
                    pass
    return texts, labels if labeled else texts, ratings if labeled else None


def preprocess_text(text):
    # Удаление HTML-тегов
    text = re.sub(r'<.*?>', '', text)
    # Удаление специальных символов и чисел
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Приведение к нижнему регистру
    text = text.lower()
    # Токенизация
    tokens = text.split()
    # Удаление стоп-слов
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Лемматизация
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


def prepare_data():
    # Пути к обучающим и тестовым данным
    train_dir = 'C:\\Users\\Roman\\PycharmProjects\\GreenAtom\\train'  # Путь относительно корня проекта
    test_dir = 'C:\\Users\\Roman\\PycharmProjects\\GreenAtom\\test'

    # Загрузка обучающих данных
    print("Загрузка обучающих данных...")
    X_train_texts, y_train_labels, y_train_ratings = load_data(train_dir, labeled=True)

    # Предобработка обучающих текстов
    print("Предобработка обучающих данных...")
    X_train = [preprocess_text(text) for text in X_train_texts]

    # Загрузка тестовых данных
    print("Загрузка тестовых данных...")
    X_test_texts, y_test_labels, y_test_ratings = load_data(test_dir, labeled=True)

    # Предобработка тестовых текстов
    print("Предобработка тестовых данных...")
    X_test = [preprocess_text(text) for text in X_test_texts]

    # Разделение данных на обучающую и валидационную выборки (опционально)
    # В данном случае, train и test уже разделены, поэтому можем использовать их напрямую

    return X_train, y_train_labels, y_train_ratings, X_test, y_test_labels, y_test_ratings


if __name__ == "__main__":
    X_train, y_train_labels, y_train_ratings, X_test, y_test_labels, y_test_ratings = prepare_data()

    # Векторизация текста
    print("Векторизация текста...")
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Обучение модели классификации
    print("Обучение классификатора...")
    classifier = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    classifier.fit(X_train_tfidf, y_train_labels)
    y_pred = classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test_labels, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Обучение модели регрессии
    print("Обучение регрессора...")

    import lightgbm as lgb

    lgb_regressor = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=8,
        num_leaves=31,
        min_data_in_leaf=20,
        learning_rate=0.1,
        random_state=42
    )
    lgb_regressor.fit(X_train_tfidf, y_train_ratings)

    y_pred_reg_lgb = lgb_regressor.predict(X_test_tfidf)
    mse_lgb = mean_squared_error(y_test_ratings, y_pred_reg_lgb)
    print(f'LightGBM Regressor MSE: {mse_lgb:.2f}')

    # Сохранение моделей и векторизатора
    print("Сохранение моделей...")
    joblib.dump(classifier, 'classifier_model.pkl')
    joblib.dump(lgb_regressor, 'regressor_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    print("Все модели успешно сохранены.")