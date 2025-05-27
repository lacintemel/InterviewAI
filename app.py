import customtkinter as ctk
import threading
import random
import sys
from io import StringIO
from datetime import datetime
import json
import os
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

# Set appearance mode and default color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class HistoryItem(ctk.CTkFrame):
    def __init__(self, parent, title, timestamp, command):
        super().__init__(parent, fg_color="transparent")
        
        self.title_label = ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(size=13),
            text_color="#ececf1",
            anchor="w"
        )
        self.title_label.pack(fill="x", padx=10, pady=(5, 0))
        
        self.timestamp_label = ctk.CTkLabel(
            self,
            text=timestamp,
            font=ctk.CTkFont(size=11),
            text_color="#8e8ea0",
            anchor="w"
        )
        self.timestamp_label.pack(fill="x", padx=10, pady=(0, 5))
        
        self.bind("<Button-1>", lambda e: command())
        self.title_label.bind("<Button-1>", lambda e: command())
        self.timestamp_label.bind("<Button-1>", lambda e: command())

class ChatBubble(ctk.CTkFrame):
    def __init__(self, parent, sender, message, is_user=False):
        super().__init__(parent, fg_color="transparent")
        
        # Create container for the bubble
        self.bubble_container = ctk.CTkFrame(
            self,
            fg_color=("#2b2c2f" if is_user else "#343541"),
            corner_radius=15
        )
        self.bubble_container.pack(
            anchor="e" if is_user else "w",
            padx=(60, 10) if is_user else (10, 60),
            pady=5,
            fill="x"
        )
        
        # Sender label
        self.sender_label = ctk.CTkLabel(
            self.bubble_container,
            text=sender,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#8e8ea0",
            anchor="w"
        )
        self.sender_label.pack(fill="x", padx=15, pady=(10, 5))
        
        # Message label
        self.message_label = ctk.CTkLabel(
            self.bubble_container,
            text=message,
            font=ctk.CTkFont(size=14),
            text_color="#ececf1",
            wraplength=600,
            justify="left",
            anchor="w"
        )
        self.message_label.pack(fill="x", padx=15, pady=(0, 10))

class InterviewApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Interview AI")
        self.root.geometry("1400x800")  # Increased width for sidebar
        
        # Configure grid
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        self.current_question = None
        self.interview_started = False
        self.conversations = []
        self.current_conversation = None
        
        # Sidebar
        self.sidebar = ctk.CTkFrame(
            self.root,
            fg_color="#202123",
            width=260,
            corner_radius=0
        )
        self.sidebar.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.sidebar.grid_propagate(False)
        
        # New chat button
        self.new_chat_button = ctk.CTkButton(
            self.sidebar,
            text="New Chat",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#19c37d",
            hover_color="#15a06a",
            height=40,
            command=self.start_new_chat
        )
        self.new_chat_button.pack(fill="x", padx=15, pady=15)
        
        # History container
        self.history_container = ctk.CTkScrollableFrame(
            self.sidebar,
            fg_color="transparent",
            label_text="History"
        )
        self.history_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Header
        self.header = ctk.CTkFrame(
            self.root,
            fg_color="#343541",
            height=80,
            corner_radius=0
        )
        self.header.grid(row=0, column=1, sticky="ew")
        self.header.grid_columnconfigure(0, weight=1)
        
        self.title = ctk.CTkLabel(
            self.header,
            text="Interview AI",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#ececf1"
        )
        self.title.grid(row=0, column=0, pady=20)
        
        # Chat area
        self.chat_frame = ctk.CTkScrollableFrame(
            self.root,
            fg_color="#202123",
            corner_radius=0
        )
        self.chat_frame.grid(row=1, column=1, sticky="nsew", padx=20, pady=(0, 20))
        
        # Input area
        self.input_frame = ctk.CTkFrame(
            self.root,
            fg_color="#343541",
            height=100,
            corner_radius=0
        )
        self.input_frame.grid(row=2, column=1, sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)
        
        # Input container
        self.input_container = ctk.CTkFrame(
            self.input_frame,
            fg_color="#40414f",
            corner_radius=15
        )
        self.input_container.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        self.input_container.grid_columnconfigure(0, weight=1)
        
        # Text input
        self.answer_var = ctk.StringVar()
        self.answer_entry = ctk.CTkEntry(
            self.input_container,
            textvariable=self.answer_var,
            font=ctk.CTkFont(size=14),
            fg_color="#40414f",
            border_width=0,
            height=40,
            placeholder_text="Type your answer here..."
        )
        self.answer_entry.grid(row=0, column=0, padx=15, pady=15, sticky="ew")
        self.answer_entry.bind('<Return>', lambda e: self.submit_answer())
        
        # Send button
        self.send_button = ctk.CTkButton(
            self.input_container,
            text="Send",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#19c37d",
            hover_color="#15a06a",
            height=40,
            width=100,
            command=self.submit_answer
        )
        self.send_button.grid(row=0, column=1, padx=(0, 15), pady=15)
        
        # Load conversations
        self.load_conversations()
        
        # Start new chat if no conversations exist
        if not self.conversations:
            self.start_new_chat()
    
    def start_new_chat(self):
        # Clear chat area
        for widget in self.chat_frame.winfo_children():
            widget.destroy()
        
        # Create new conversation
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        conversation = {
            "id": len(self.conversations),
            "title": f"Interview {len(self.conversations) + 1}",
            "timestamp": timestamp,
            "messages": []
        }
        self.conversations.append(conversation)
        self.current_conversation = conversation
        
        # Add to history
        self.add_to_history(conversation)
        
        # Start interview
        self.interview_started = True
        self.get_next_question()
        
        # Save conversations
        self.save_conversations()
    
    def add_to_history(self, conversation):
        item = HistoryItem(
            self.history_container,
            conversation["title"],
            conversation["timestamp"],
            lambda: self.load_conversation(conversation["id"])
        )
        item.pack(fill="x", pady=2)
    
    def load_conversation(self, conversation_id):
        # Find conversation
        conversation = next((c for c in self.conversations if c["id"] == conversation_id), None)
        if not conversation:
            return
        
        # Clear chat area
        for widget in self.chat_frame.winfo_children():
            widget.destroy()
        
        # Set current conversation
        self.current_conversation = conversation
        
        # Load messages
        for message in conversation["messages"]:
            self.add_bubble(message["sender"], message["text"], message["is_user"])
    
    def add_bubble(self, sender, message, is_user=False):
        bubble = ChatBubble(self.chat_frame, sender, message, is_user)
        bubble.pack(fill="x", pady=2)
        self.root.after(100, self.scroll_to_bottom)
        
        # Save to conversation
        if self.current_conversation:
            self.current_conversation["messages"].append({
                "sender": sender,
                "text": message,
                "is_user": is_user
            })
            self.save_conversations()
    
    def scroll_to_bottom(self):
        self.chat_frame._parent_canvas.yview_moveto(1.0)
    
    def get_next_question(self):
        available_questions = [(q, data) for q, data in question_set.items() if data["times_asked"] < 10]
        if not available_questions:
            self.add_bubble("AI", "🎉 Congratulations! You've completed all questions!", is_user=False)
            self.send_button.configure(state="disabled")
            self.answer_entry.configure(state="disabled")
            return
        
        available_questions.sort(key=lambda x: (-x[1]["rating"], x[1]["times_asked"]))
        top_questions = available_questions[:min(3, len(available_questions))]
        self.current_question = random.choice(top_questions)[0]
        question_set[self.current_question]["times_asked"] += 1
        self.add_bubble("AI", self.current_question, is_user=False)
    
    def calculate_rating(self, answer):
        expected_answers = question_set[self.current_question]["answers"]
        embeddings1 = model.encode(expected_answers, convert_to_tensor=True)
        embeddings2 = model.encode(answer, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
        semantic_score = float(similarity[0][0]) * 100
        sentiment = sentiment_analyzer(answer)[0]
        sentiment_score = sentiment['score']
        sentiment_label = sentiment['label']
        word_count = len(answer.split())
        matched_keywords = []
        answer_lower = answer.lower()
        for topic, keywords in keyword_map.items():
            if any(topic in q.lower() for q in [self.current_question]):
                for kw in keywords:
                    if kw in answer_lower:
                        matched_keywords.append(kw)
        total_score = 0
        if semantic_score >= 80:
            total_score += 40
        elif semantic_score >= 60:
            total_score += 30
        elif semantic_score >= 40:
            total_score += 20
        else:
            total_score += 10
        if sentiment_label == 'POSITIVE':
            if sentiment_score >= 0.9:
                total_score += 20
            elif sentiment_score >= 0.7:
                total_score += 15
            else:
                total_score += 10
        else:
            total_score += 5
        keyword_count = len(matched_keywords)
        if keyword_count >= 3:
            total_score += 20
        elif keyword_count == 2:
            total_score += 15
        elif keyword_count == 1:
            total_score += 10
        else:
            total_score += 5
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
        answer = self.answer_var.get().strip()
        if not answer:
            return
        self.add_bubble("You", answer, is_user=True)
        self.answer_var.set("")
        log_answer(self.current_question, answer, question_set[self.current_question]["rating"])
        threading.Thread(target=self.process_feedback, args=(answer,), daemon=True).start()
    
    def process_feedback(self, answer):
        total_score = self.calculate_rating(answer)
        rating_text = f"Overall Rating: {total_score}/100 ({round(total_score/10, 1)}/10)"
        old_stdout = sys.stdout
        feedback_output = StringIO()
        sys.stdout = feedback_output
        score_answer(self.current_question, answer)
        feedback_text = feedback_output.getvalue()
        sys.stdout = old_stdout
        self.root.after(0, lambda: self.add_bubble("AI", rating_text + "\n" + feedback_text, is_user=False))
        self.root.after(0, self.get_next_question)
    
    def save_conversations(self):
        try:
            with open("conversations.json", "w", encoding="utf-8") as f:
                json.dump(self.conversations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving conversations: {e}")
    
    def load_conversations(self):
        try:
            if os.path.exists("conversations.json"):
                with open("conversations.json", "r", encoding="utf-8") as f:
                    self.conversations = json.load(f)
                    for conversation in self.conversations:
                        self.add_to_history(conversation)
        except Exception as e:
            print(f"Error loading conversations: {e}")

def main():
    app = InterviewApp()
    app.root.mainloop()

if __name__ == "__main__":
    main() 