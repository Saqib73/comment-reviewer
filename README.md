🧠 Comment Classification Tool

A machine-learning tool that automatically classifies social media or product comments into useful categories — helping teams respond better, faster, and more professionally.

⭐ What This Tool Does

Online comments come in all forms — appreciation, criticism, hate, spam, even personal emotions. This tool helps you automatically sort them into 8 meaningful categories:

Praise – Positive feedback

Support – Encouraging messages

Constructive Criticism – Helpful negative feedback

Hate/Abuse – Insults, trolling, toxic remarks

Threat – Warnings or threatening content

Emotional – Personal feelings, memories

Irrelevant/Spam – Promotions, bots, unrelated content

Question/Suggestion – Queries or new ideas

This makes it easier to
✅ Respond better
✅ Handle hate safely
✅ Improve customer engagement
✅ Filter spam automatically

📁 Dataset

The project uses a balanced dataset of 160 labeled comments
→ 20 comments for each of the 8 categories
→ Stored in project_data.csv

You can add more comments anytime to improve the model.

🛠️ Tech Stack

Python 3.8+

scikit-learn (Logistic Regression, SVM)

NLTK (tokenization, stopwords, lemmatization)

pandas

Streamlit (web app)

Plotly (visualizations)

🚀 How to Use
1. Install Dependencies
pip install -r requirements.txt

2. Train the Model
python train_model.py


This will generate:

model.pkl

vectorizer.pkl

label_encoder.json

3. Run the Web App
streamlit run app.py


Features in the app:

1. Classify a single comment

2. Upload CSV/JSON for batch classification

View charts and category statistics

Get auto-generated response templates

4. Command-Line Use

Classify one comment:

python classify_comments.py --text "Great work!"



📂 Project Structure
content-reviewer/
│── project_data.csv
│── train_model.py
│── classify_comments.py
│── preprocessing.py
│── response_templates.py
│── app.py
│── model.pkl
│── vectorizer.pkl
│── label_encoder.json
│── requirements.txt
│── README.md