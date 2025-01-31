"""This module contains the sentiment analysis library calls and methods for that"""
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_textblob(text):
    """Analyze sentiment using TextBlob."""
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

def analyze_vader(text):
    """Analyze sentiment using Vader."""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores
