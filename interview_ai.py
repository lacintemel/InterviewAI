import random
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
from keybert import KeyBERT
import nltk
import json
import os
from datetime import datetime

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# Load sentiment analysis and NER pipelines
sentiment_analyzer = pipeline('sentiment-analysis')
ner_pipeline = pipeline('ner', grouped_entities=True)
# Load KeyBERT for keyword extraction
kw_model = KeyBERT(model)

# Sample questions and expected answers
question_set = {
    "Can you tell me about yourself?": {
        "answers": [
            "I am a motivated individual with a background in computer engineering.",
            "I have a strong passion for technology and have worked on several software projects.",
            "My experience includes internships in data science and web development."
        ],
        "rating": 0,
        "times_asked": 0
    },
    "What are your strengths?": {
        "answers": [
            "I am detail-oriented, a good problem solver, and a team player.",
            "I am creative, hardworking, and adapt quickly to new environments.",
            "I have strong analytical skills and communicate effectively."
        ],
        "rating": 0,
        "times_asked": 0
    },
    "Why do you want this job?": {
        "answers": [
            "I am passionate about this field and I want to contribute to your company's growth.",
            "This position aligns with my career goals and offers great learning opportunities.",
            "I admire your company's culture and want to be part of your innovative team."
        ],
        "rating": 0,
        "times_asked": 0
    },
    "What are your weaknesses?": {
        "answers": [
            "Sometimes I focus too much on details, but I'm working on improving that.",
            "I can be a perfectionist, but I am learning to balance quality and efficiency.",
            "I tend to be shy in new groups, but I make an effort to connect with colleagues."
        ],
        "rating": 0,
        "times_asked": 0
    },
    "Where do you see yourself in 5 years?": {
        "answers": [
            "I see myself in a leadership position where I can guide teams and contribute to impactful projects.",
            "I hope to have advanced my skills and be managing larger projects.",
            "I want to be recognized as an expert in my field and mentor others."
        ],
        "rating": 0,
        "times_asked": 0
    },
    "Describe a challenging project you worked on.": {
        "answers": [
            "I worked on a machine learning project with tight deadlines, requiring teamwork and creative problem-solving.",
            "I developed a web application under resource constraints, learning to prioritize tasks and communicate clearly."
        ],
        "rating": 0,
        "times_asked": 0
    },
    "How do you handle stress and pressure?": {
        "answers": [
            "I stay organized, break tasks into smaller steps, and maintain a positive attitude.",
            "I use time management techniques and seek support from my team when needed."
        ],
        "rating": 0,
        "times_asked": 0
    },
    "Tell me about a time you showed leadership.": {
        "answers": [
            "I led a student project team, delegating tasks and motivating members to achieve our goals.",
            "I took initiative during a crisis, coordinating efforts and ensuring clear communication."
        ],
        "rating": 0,
        "times_asked": 0
    }
}

# Keywords for evaluation (expanded version)
keyword_map = {
    "strengths": [
        "detail-oriented", "problem solver", "team player", "creative", "hardworking", "analytical",
        "adaptable", "communicate", "leadership", "organized", "efficient", "motivated", "collaborative",
        "initiative", "reliable", "focused", "dedicated", "resilient", "independent"],
    "weaknesses": [
        "perfectionist", "focus too much", "impatient", "shy", "overthink", "self-critical", "delegation",
        "public speaking", "disorganized", "procrastinate", "detail-obsessed", "multitasking", "insecure"],
    "goals": [
        "leadership", "team", "projects", "growth", "mentor", "expert", "manager", "skills", "career",
        "promotion", "responsibility", "certification", "entrepreneurship", "long-term", "achievement"],
    "motivation": [
        "passionate", "interested", "career", "opportunity", "culture", "learning", "innovation", "values",
        "mission", "impact", "challenge", "recognition", "collaboration", "vision", "drive"],
    "introduction": [
        "background", "experience", "skills", "internship", "education", "projects", "technology", "software",
        "development", "programming", "university", "degree", "certification", "journey", "role", "training"],
    "stress": [
        "organized", "time management", "positive attitude", "support", "break tasks", "prioritize",
        "resilience", "calm", "focus", "breathe", "routine", "deadlines", "communication", "balance"],
    "leadership": [
        "delegating", "motivating", "initiative", "coordinating", "communication", "teamwork",
        "vision", "decision-making", "responsibility", "supportive", "mentor", "collaborate", "strategic", "goal-setting"]
}


def log_answer(question, answer, rating, log_file="cevaplar_log.json"):
    log_path = os.path.join(os.path.dirname(__file__), log_file)
    # Load existing log or start new
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                data = []
    else:
        data = []
    # Append new entry
    data.append({
        "soru": question,
        "cevap": answer,
        "rating": rating,
        "timestamp": str(datetime.now())
    })
    # Save back
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_next_question():
    # Sort questions by rating and times asked
    available_questions = [(q, data) for q, data in question_set.items() if data["times_asked"] < 10]
    if not available_questions:
        print("\n🎉 Congratulations! You've completed all questions!")
        return None
    
    # Sort by rating (descending) and times_asked (ascending)
    available_questions.sort(key=lambda x: (-x[1]["rating"], x[1]["times_asked"]))
    
    # Select from top 3 questions randomly to add some variety
    top_questions = available_questions[:min(3, len(available_questions))]
    selected_question = random.choice(top_questions)[0]
    question_set[selected_question]["times_asked"] += 1
    return selected_question

def ask_question():
    question = get_next_question()
    if not question:
        return False
    
    print("\n📝 Interviewer: " + question)
    user_answer = input("Your answer: ")
    log_answer(question, user_answer, question_set[question]["rating"])
    score_answer(question, user_answer)
    
    # Ask for question rating
    while True:
        rating = input("\nDid you find this question helpful? (yes/no): ").lower().strip()
        if rating in ['yes', 'no']:
            question_set[question]["rating"] += 1 if rating == 'yes' else -1
            break
        print("Please answer with 'yes' or 'no'")
    
    return True

def score_answer(question, answer):
    expected_answers = question_set[question]["answers"]
    embeddings1 = model.encode(expected_answers, convert_to_tensor=True)
    embeddings2 = model.encode(answer, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
    score = float(similarity[0][0]) * 100
    
    # Sentiment analysis
    sentiment = sentiment_analyzer(answer)[0]
    sentiment_score = sentiment['score']
    sentiment_label = sentiment['label']
    
    print("\n🎯 AI Feedback:")
    if score >= 80:
        print("🌟 Outstanding! Your answer is perfectly aligned with what interviewers look for!")
    elif score >= 60:
        print("✨ Great job! You're definitely on the right track!")
    elif score >= 40:
        print("💫 Good effort! Let's add a bit more detail to make it even better!")
    else:
        print("💡 Keep going! Here are some tips to improve your answer:")

    print(f"\n😊 Sentiment Analysis:")
    if sentiment_label == 'POSITIVE':
        if sentiment_score > 0.9:
            print("🎉 Amazing enthusiasm! Your positivity really shines through!")
        elif sentiment_score > 0.7:
            print("😃 Great confidence! Your positive attitude is clear!")
        else:
            print("🙂 Good tone! A bit more enthusiasm would make it even better!")
    else:
        if sentiment_score > 0.7:
            print("🤔 Let's add more positivity to your response!")
        else:
            print("😊 Try to be more optimistic in your answer!")

    # Named Entity Recognition (hidden from user)
    entities = ner_pipeline(answer)
    # KeyBERT keyword extraction (hidden from user)
    keybert_keywords = kw_model.extract_keywords(answer, top_n=5)
    # Keyword checking (hidden from user)
    matched_keywords = check_keywords(answer)

    # Get relevant keywords for the question type
    relevant_keywords = []
    for topic, keywords in keyword_map.items():
        if any(topic in q.lower() for q in [question]):
            relevant_keywords.extend(keywords)

    # Answer length feedback
    word_count = len(answer.split())
    print("\n💫 Suggestions:")
    if word_count < 10:
        print("📝 Your answer is quite brief. Let's add more details to make it shine!")
    elif word_count < 30:
        print("📚 Perfect length! You've provided a concise yet informative answer!")
    elif word_count < 50:
        print("📖 Excellent detail! Your answer is comprehensive and well-structured!")
    else:
        print("📋 Your answer is quite detailed. Make sure all points are relevant!")

    if not matched_keywords and relevant_keywords:
        print("🔑 Try to incorporate these powerful keywords in your answer:")
        # Show up to 5 example keywords
        example_keywords = random.sample(relevant_keywords, min(5, len(relevant_keywords)))
        for kw in example_keywords:
            print(f"  ✨ {kw}")
    
    if sentiment_label != 'POSITIVE' or sentiment_score < 0.7:
        print("😊 Add more positivity and confidence to your response!")
    
    if score < 60:
        print("🎯 Here's an example of how to structure your answer:")
        # Show an example answer structure
        if expected_answers:
            example = random.choice(expected_answers)
            # Show first 100 characters of example
            print(f"  💡 \"{example[:100]}...\"")

def check_keywords(answer):
    matched = []
    answer_lower = answer.lower()
    for topic, keywords in keyword_map.items():
        for kw in keywords:
            if kw in answer_lower:
                matched.append(kw)
    return matched

def run_interview():
    print("👋 Welcome to AI-Powered Mock Interview Assistant!")
    print("Press Ctrl+C to exit at any time.")
    print("\nEach question will be asked up to 10 times.")
    print("Your feedback helps improve the question selection!")
    
    try:
        while ask_question():
            pass
    except KeyboardInterrupt:
        print("\n\nThank you for using the AI-Powered Mock Interview Assistant! Good luck with your real interviews! 🚀")

if __name__ == "__main__":
    run_interview()
