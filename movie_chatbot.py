import os
import re
import pandas as pd
import streamlit as st
from retrieval_fixed import recommend_movies  # your existing recommender

st.set_page_config(page_title="🎬 Movie Chatbot", page_icon="🎬")
st.title("🎬 Conversational Movie Recommender")

#helper functions for small talk
SMALL_TALK = ("hi", "hello", "hey", "sup", "yo")

def is_small_talk(text: str) -> bool:
    t = text.strip().lower()
    return t in SMALL_TALK or t.startswith("hi ") or t.startswith("hello")

def clean_query(text: str) -> str:
    # remove common filler words
    t = text.lower()
    t = re.sub(r"\b(i want|i would like|can i get|can you recommend|movie|movies|film|show me|find me)\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or text 

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey! Tell me what you're in the mood for (genre, year, actor, vibe)."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#main 
user_input = st.chat_input("Describe the movie...")
if user_input:
    # user query
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    #in case of small talk like "hi"
    if is_small_talk(user_input):
        bot_reply = "Hi 👋 Tell me something like “2000s romcom”, “heist movie”, or “sci-fi with aliens”."
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    else:
        # clean query
        cleaned_query = clean_query(user_input)

        # run main api
        results = recommend_movies(query=cleaned_query, top_n=5)

        #reply
        if results is None or results.empty:
            bot_reply = (
                f"I tried looking for **{cleaned_query}** but didn’t find a close match.\n"
                "Try adding a genre, actor, or decade."
            )
        else:
            lines = [f"Here are some that match **{cleaned_query}**:"]
            for _, row in results.iterrows():
                title = row["title"]
                year = str(row["release_date"])[:4] if "release_date" in row and pd.notna(row["release_date"]) else "N/A"
                rating = row["vote_average"]
                runtime = int(row["runtime"]) if pd.notna(row["runtime"]) else None
                genres = row["genres"]
                runtime_str = f" | ⏱ {runtime} min" if runtime else ""
                lines.append(f"- **{title}** ({year}) — ⭐ {rating:.1f}{runtime_str} | 🎭 {genres}")
            bot_reply = "\n".join(lines)

        with st.chat_message("assistant"):
            st.markdown(bot_reply)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
