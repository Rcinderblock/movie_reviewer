from django.shortcuts import render
from django.views import View
from .forms import ReviewForm
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Определите путь к моделям относительно файла views.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Загрузка обученных моделей и векторизатора
classifier_path = os.path.join(BASE_DIR, 'classifier_model.pkl')
regressor_path = os.path.join(BASE_DIR, 'regressor_model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'vectorizer.pkl')

classifier = joblib.load(classifier_path)
regressor = joblib.load(regressor_path)
vectorizer = joblib.load(vectorizer_path)


def classify_review(text):
    # Предобработка текста должна быть такой же, как при обучении
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

    def preprocess_text(text):
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        tokens = text.split()
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    processed_text = preprocess_text(text)
    tfidf_vector = vectorizer.transform([processed_text])

    classification = classifier.predict(tfidf_vector)[0]
    status = 'Positive' if classification == 1 else 'Negative'

    predicted_rating = regressor.predict(tfidf_vector)[0]
    predicted_rating = int(round(predicted_rating))
    predicted_rating = max(1, min(10, predicted_rating))

    return status, predicted_rating


class HomeView(View):
    def get(self, request):
        form = ReviewForm()
        return render(request, 'reviews/home.html', {'form': form})

    def post(self, request):
        form = ReviewForm(request.POST)
        review_text = None
        status = None
        rating = None

        if form.is_valid():
            review_text = form.cleaned_data['review_text']
            status, rating = classify_review(review_text)

        return render(request, 'reviews/home.html', {
            'form': form,
            'review': review_text,
            'status': status,
            'rating': rating,
        })