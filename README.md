# ITSolera_AI-Powered-Game_story_generation


AI-Powered Game Story Generator

Overview
This repository contains an AI-powered game story generator built with Streamlit. The application leverages advanced natural language processing models GPT-2 to generate engaging and unique game stories based on user inputs.

Features

Interactive UI: Built with Streamlit for a user-friendly interface.
AI Story Generation: Uses a language model GPT-2 to create game narratives based on input parameters.
Customizable Inputs: Allow users to specify story and dialogue for story generation.
Real-Time Output: Generates and displays stories in real-time as users interact with the application.

Installation
To set up and run the AI-powered game story generator locally, follow these steps:

Prerequisites
Python 3.12.0
Virtual environment (recommended)
Setup

Create a Virtual Environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

Install Dependencies:

bash
Copy code
pip install -r requirements.txt

Run the Application:

bash
Copy code
streamlit run app.py
This will open the application in your default web browser.

Usage

Enter Story Parameters:

Specify themes in the input fields on the Streamlit app.

Generate Story:

Click the "Generate Story" button to see the AI-generated narrative based on your inputs.

Enter Dialogue Parameters:

Specify themes in the input fields on the Streamlit app.

Generate Dialogue:

Click the "Generate Dialogue" button to see the AI-generated narrative based on your inputs.


Explore Results:

Review and interact with the generated story in the app.
Files
app.py: Main Streamlit application script.
requirements.txt: List of Python packages required for the project.
model/: Directory containing pre-trained models GPT-2 and scripts for story generation.

