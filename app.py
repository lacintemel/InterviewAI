import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import random
import sys
from io import StringIO
from interview_ai import (
    question_set,
    model,
    sentiment_analyzer,
    ner_pipeline,
    kw_model,
    keyword_map,
    score_answer,
    log_answer
)
import torch
from sentence_transformers import util

class InterviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Interview Assistant")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.current_question = None
        self.interview_started = False
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create widgets
        self.create_widgets()
        
        # Start with the first question
        self.start_interview()
    
    def create_widgets(self):
        # Question display
        self.question_frame = ttk.LabelFrame(self.main_frame, text="Current Question", padding="10")
        self.question_frame.pack(fill=tk.X, pady=5)
        
        self.question_label = ttk.Label(self.question_frame, text="", wraplength=700)
        self.question_label.pack(fill=tk.X)
        
        # Answer input
        self.answer_frame = ttk.LabelFrame(self.main_frame, text="Your Answer", padding="10")
        self.answer_frame.pack(fill=tk.X, pady=5)
        
        self.answer_text = scrolledtext.ScrolledText(self.answer_frame, height=4, wrap=tk.WORD)
        self.answer_text.pack(fill=tk.X)
        
        # Submit button
        self.submit_button = ttk.Button(self.answer_frame, text="Submit Answer", command=self.submit_answer)
        self.submit_button.pack(pady=5)
        
        # Rating display
        self.rating_frame = ttk.LabelFrame(self.main_frame, text="Overall Rating", padding="10")
        self.rating_frame.pack(fill=tk.X, pady=5)
        
        self.rating_label = ttk.Label(self.rating_frame, text="", font=('Helvetica', 14, 'bold'))
        self.rating_label.pack(fill=tk.X)
        
        # Feedback display
        self.feedback_frame = ttk.LabelFrame(self.main_frame, text="AI Feedback", padding="10")
        self.feedback_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.feedback_text = scrolledtext.ScrolledText(self.feedback_frame, wrap=tk.WORD)
        self.feedback_text.pack(fill=tk.BOTH, expand=True)
        
        # Chat history
        self.chat_frame = ttk.LabelFrame(self.main_frame, text="Chat History", padding="10")
        self.chat_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.chat_text = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD)
        self.chat_text.pack(fill=tk.BOTH, expand=True)
    
    def start_interview(self):
        self.interview_started = True
        self.get_next_question()
    
    def get_next_question(self):
        available_questions = [(q, data) for q, data in question_set.items() if data["times_asked"] < 10]
        if not available_questions:
            self.question_label.config(text="🎉 Congratulations! You've completed all questions!")
            self.submit_button.config(state=tk.DISABLED)
            return
        
        available_questions.sort(key=lambda x: (-x[1]["rating"], x[1]["times_asked"]))
        top_questions = available_questions[:min(3, len(available_questions))]
        self.current_question = random.choice(top_questions)[0]
        question_set[self.current_question]["times_asked"] += 1
        
        self.question_label.config(text=f"🤖 Interviewer: {self.current_question}")
        self.add_to_chat("Interviewer", self.current_question)
    
    def calculate_rating(self, answer):
        # Calculate semantic similarity
        expected_answers = question_set[self.current_question]["answers"]
        embeddings1 = model.encode(expected_answers, convert_to_tensor=True)
        embeddings2 = model.encode(answer, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
        semantic_score = float(similarity[0][0]) * 100

        # Get sentiment
        sentiment = sentiment_analyzer(answer)[0]
        sentiment_score = sentiment['score']
        sentiment_label = sentiment['label']

        # Calculate word count
        word_count = len(answer.split())

        # Calculate keyword matches
        matched_keywords = []
        answer_lower = answer.lower()
        for topic, keywords in keyword_map.items():
            if any(topic in q.lower() for q in [self.current_question]):
                for kw in keywords:
                    if kw in answer_lower:
                        matched_keywords.append(kw)

        # Calculate total score (0-100)
        total_score = 0

        # 1. Semantic Similarity (0-40)
        if semantic_score >= 80:
            total_score += 40
        elif semantic_score >= 60:
            total_score += 30
        elif semantic_score >= 40:
            total_score += 20
        else:
            total_score += 10

        # 2. Sentiment Score (0-20)
        if sentiment_label == 'POSITIVE':
            if sentiment_score >= 0.9:
                total_score += 20
            elif sentiment_score >= 0.7:
                total_score += 15
            else:
                total_score += 10
        else:
            total_score += 5

        # 3. Keyword Match (0-20)
        keyword_count = len(matched_keywords)
        if keyword_count >= 3:
            total_score += 20
        elif keyword_count == 2:
            total_score += 15
        elif keyword_count == 1:
            total_score += 10
        else:
            total_score += 5

        # 4. Word Count / Detail (0-20)
        if word_count >= 50:
            total_score += 20
        elif word_count >= 30:
            total_score += 15
        elif word_count >= 10:
            total_score += 10
        else:
            total_score += 5

        return total_score

    def submit_answer(self):
        answer = self.answer_text.get("1.0", tk.END).strip()
        if not answer:
            return
        
        # Add answer to chat
        self.add_to_chat("You", answer)
        
        # Clear answer text
        self.answer_text.delete("1.0", tk.END)
        
        # Log and score the answer
        log_answer(self.current_question, answer, question_set[self.current_question]["rating"])
        
        # Get feedback in a separate thread to keep UI responsive
        threading.Thread(target=self.process_feedback, args=(answer,), daemon=True).start()
    
    def process_feedback(self, answer):
        # Calculate rating
        total_score = self.calculate_rating(answer)
        
        # Update rating display
        rating_text = f"Overall Rating: {total_score}/100 ({round(total_score/10, 1)}/10)"
        self.root.after(0, lambda: self.rating_label.config(text=rating_text))
        
        # Capture the output of score_answer
        old_stdout = sys.stdout
        feedback_output = StringIO()
        sys.stdout = feedback_output
        
        # Score the answer
        score_answer(self.current_question, answer)
        
        # Get the feedback text
        feedback_text = feedback_output.getvalue()
        sys.stdout = old_stdout
        
        # Update the feedback in the GUI
        self.root.after(0, lambda: self.add_feedback(feedback_text))
        
        # Get next question
        self.root.after(0, self.get_next_question)
    
    def add_to_chat(self, sender, message):
        self.chat_text.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_text.see(tk.END)
    
    def add_feedback(self, feedback):
        self.feedback_text.delete("1.0", tk.END)  # Clear previous feedback
        self.feedback_text.insert(tk.END, feedback)
        self.feedback_text.see(tk.END)

def main():
    root = tk.Tk()
    app = InterviewApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 