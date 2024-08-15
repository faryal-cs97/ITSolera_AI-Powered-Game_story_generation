import nltk #text processing
nltk.download('punkt') #tokenize paragraph into sentence or words
import streamlit as st #creating web app 
import pandas as pd #data manipulation
import re #regular expression that search for pattern within text
from nltk.tokenize import sent_tokenize #split paragraph into sentences
from transformers import GPT2LMHeadModel, GPT2Tokenizer 
#gpt2headmodel is a pretrained lm generate text and understand language
#gpt2tokenizer prepared text to be processed by GPT2 model

# Load pre-trained GPT-2 model and tokenizer that generate text based on the input prompt
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name) #prepares text so that model can understand
model = GPT2LMHeadModel.from_pretrained(model_name) #model generates text
model.eval() #used for tasks like generate text without learning anything new

# Data preprocessing function
# load some text converts to lower case
#cleans it up remove punctuation and special char 
#splits text into sentences and return it
#try n except if an error occured it shows an error msg and return empty list
def preprocess_text(text): #clean and split text into sentences
    try: #handling errors without crashing the program
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation and special charachter remove those that arent letters
        sentences = sent_tokenize(text)  # Sentence tokenization 
        return sentences #hello world, how r u to ["hello world","how are u"]
    except Exception as e: #catches any errors in try block without crashing program
        st.error(f"Error preprocessing text: {e}")
        return []

# Function to generate story

def generate_story(prompt, max_length=200, temperature=0.7, top_k=50, top_p=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors='pt') #This line converts the prompt (text) into a format (tokens) that the model can understand.
    output = model.generate( #This is where the actual story generation happens.
        input_ids, #(the encoded prompt) and uses it to generate a longer piece of tex
        max_length=max_length, #he story won't be longer than this number of tokens.
        temperature=temperature, #Adjusts randomness in word selection.
        top_k=top_k, #limits the selection to the top 50 most likely next words.
        top_p=top_p, #Selects from a pool of words that together have a cumulative probability of 0.9.
        pad_token_id=tokenizer.eos_token_id, #Uses a special token to fill in if the text is shorter than max_length.
        attention_mask=None,#It can be used to focus on certain parts of the input, but it's not needed here.
        num_return_sequences=1, #he model generates only one story.
        early_stopping=True, #The model stops generating if it feels the story is complete before reaching max_length.
        no_repeat_ngram_size=2 #Prevents the model from repeating the same sequence of 2 words, making the story more coherent.
    )
    story = tokenizer.decode(output[0], skip_special_tokens=True) #This line converts the generated text (which is in token format) back into human-readable text.
    #removes any special tokens (like padding or end-of-sentence markers) from the output.
    return story #inally, the function returns the generated story as a string.

# Function to generate dialogue
#Splits the dialogue into separate lines or sentences
#eturns the generated dialogue as a list of individual sentences or lines, making it easy to read or display as a conversation.
def generate_dialogue(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(
        input_ids, #(encoded prompt) and generates a dialogue based on it.
        max_length=max_length, #Limits the length of the dialogue.
        temperature=temperature, #Adjusts randomness in word selection.
        top_k=top_k, #Limits the word selection to the top 50 most likely next words.
        top_p=top_p, #Chooses from a pool of words that together have a cumulative probability of 0.9
        pad_token_id=tokenizer.eos_token_id, #Uses a special token to fill in if the text is shorter than max_length.
        attention_mask=None, 
        do_sample=True, #Ensures the model samples words randomly (within the constraints of temperature, top_k, and top_p), making the output less predictable.
        repetition_penalty=repetition_penalty #Penalizes the model for repeating the same words, encouraging more varied dialogue.
    )
    dialogue = tokenizer.decode(output[0], skip_special_tokens=True) #This line converts the generated text (in token format) back into human-readable text.
    dialogue_lines = re.split(r'(?<=[.!?])\s+', dialogue) #removes any special tokens (like padding or end-of-sentence markers) from the output.
    #This line splits the dialogue into individual lines or sentences.
    return dialogue_lines #Finally, the function returns the generated dialogue as a list of sentences or lines.

# Streamlit app
#to create a title for a web app
st.title('AI-Powered Game_Story Generation using GPT-2')


# Upload file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    text_data = df['message'].tolist() #retrieve data from msg column to a list

    # Preprocess text
    preprocessed_stories = [preprocess_text(story) for story in text_data] #variable story to preprocess data in msg
    
    # Generate story
    prompt = st.text_input("Enter a prompt for story generation", "In a distant galaxy, a brave warrior")
    if st.button("Generate Story"):
        story = generate_story(prompt)
        st.write("Generated Story:")
        st.write(story)
    
    # Generate dialogue
    dialogue_prompt = st.text_input("Enter a prompt for dialogue generation", "Hero: Where are we heading?")
    if st.button("Generate Dialogue"):
        dialogue = generate_dialogue(dialogue_prompt)
        st.write("Generated Dialogue:")
        for line in dialogue:
            st.write(line)
