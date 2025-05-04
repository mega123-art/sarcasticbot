# sarcasticbot
don't talk with him he is bad boy
# 🧠 Chatbot using Groq API

This is a customizable and persistent chatbot built with the [Groq API](https://groq.com/), supporting real-time context, personality traits, and streaming responses. It stores chat history and adapts responses based on user-defined rules.

## ✨ Features

- 💬 Chat using `llama3-70b-8192` model from Groq
- 🧠 Maintains conversation history (`chatbot.json`)
- 🕒 Includes real-time date and time context
- 🤖 Customizable assistant personality (funny, sarcastic, emotional)
- 🌐 Responds in the user's language
- ⚡ Streamed responses for better UX


## ⚙️ Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/groq-chatbot.git
   cd groq-chatbot
   pip install -r requirements.txt
   python chatbot.py


# update your env file
-GROQ_API_KEY=your_groq_api_key
-USERNAME=YourName
-ASSISTANT_NAME=YourBot


