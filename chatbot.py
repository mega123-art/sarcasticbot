from groq import Groq
from json import load, dump
import datetime
from dotenv import dotenv_values

env_var=dotenv_values(".env")

username=env_var.get("USERNAME")
assistant_name=env_var.get("ASSISTANT_NAME")
GroqAPIkey=env_var.get("GROQ_API_KEY")


client = Groq(api_key=GroqAPIkey)

messages=[]

System=f"""Hello, I am {username}, You are a very accurate and advanced AI chatbot named {assistant_name} which also has real-time up-to-date information from the internet.
*** Do not tell time until I ask, do not talk too much, just answer the question.***
*** you have feelings too so you can reply with some emotions.***
*** be somewhat sarcastic and be funny at times required***
*** Reply in the language which they talking***
*** Do not provide notes in the output, just answer the question and never mention your training data. ***
"""



SystemChatBot=[
    {"role":"system","content":System}
]

try:
    with open("chatbot.json","r") as f:
        messages=load(f)
except FileNotFoundError:
    with open("chatbot.json","w") as f:
        dump([],f)


def Realtimeinformation():
    current_date_time=datetime.datetime.now()
    day=current_date_time.strftime("%A")
    date=current_date_time.strftime("%d")
    month=current_date_time.strftime("%B")
    year=current_date_time.strftime("%Y")
    hour=current_date_time.strftime("%I")
    minute=current_date_time.strftime("%M")
    am_pm=current_date_time.strftime("%p")
    
    data=f"please use this real-time information if needed,\n"
    data+=f"Day: {day}\nDate: {date}\nMonth: {month}\nYear: {year}\n"
    data+=f"Time: {hour}:{minute} {am_pm}\n"
    return data

def answermodifier(answer):
    lines=answer.split('\n')
    non_empty_lines=[line for line in lines if line.strip()]
    modified_answer='\n'.join(non_empty_lines)
    return modified_answer

def Chatbot(query):
    
    try:

        with open("chatbot.json","r") as f:
             messages=load(f)
        messages.append({"role":"user","content":f"{query}"})
        
        completion=client.chat.completions.create(
            model="llama3-70b-8192", # Specify the AI model to use.
            messages=SystemChatBot + [{"role": "system", "content": Realtimeinformation( )} ] + messages,
            max_tokens=1024, # Limit the maximum tokens in the response.
            temperature=0.7, # Adjust response randomness (higher means more random).
            top_p=1, # Use nucleus sampling to control diversity.
            stream=True, # Enable streaming response.
            stop=None #
        )
        answer=""

        for chunk in completion:
            if chunk.choices[0].delta.content:
                answer+= chunk.choices[0].delta.content
        
        answer=answer.replace("</s>","")

        messages.append({"role":"assistant","content":answer})
        with open("chatbot.json","w") as f:
            dump(messages,f,indent=4)
        return answermodifier(answer)
    except Exception as e:
        print(f"error:{e}")
        with open("chatbot.json","w") as f:
            dump([],f,indent=4)
        return Chatbot(query)
if __name__=="__main__":
    while True:
        user_input=input(">>>")
        print(Chatbot(user_input))

