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
1. Clone the Repository
   https://github.com/Saqib73/comment-reviewer

2. Install Dependencies
pip install -r requirements.txt

3. Train the Model
python train_model.py


This will generate:

model.pkl

vectorizer.pkl

label_encoder.json

4. Run the Web App
streamlit run app.py


Features in the app:

1. Classify a single comment

2. Upload CSV/JSON for batch classification

View charts and category statistics

Get auto-generated response templates

4. Command-Line Use

Classify one comment:

python classify_comments.py --text "Great work!"


📊 Example Results

Here are some example classifications to show how the model works:

Example 1: Praise
Input: "Amazing work! The animation looks stunning."
Output:
- Category: Praise
- Confidence: 0.92 (92%)
- Suggested Response: "Thank you so much for your kind words! We're thrilled that you enjoyed it. Your support means the world to us! 🙏"

Example 2: Constructive Criticism
Input: "The animation is okay but feels a bit slow at some parts."
Output:
- Category: Constructive Criticism
- Confidence: 0.85 (85%)
- Suggested Response: "Thank you for your thoughtful feedback! We really appreciate you taking the time to share your perspective. We'll definitely take your suggestions into consideration for future content. Your input helps us grow! 📝"

Example 3: Hate/Abuse
Input: "This is trash, stop posting."
Output:
- Category: Hate/Abuse
- Confidence: 0.88 (88%)
- Suggested Response: "We're sorry to hear that our content didn't resonate with you. We're always open to constructive feedback if you'd like to share specific concerns. We aim to create content that everyone can enjoy. 🙏"

Example 4: Question/Suggestion
Input: "Can you make one about space exploration?"
Output:
- Category: Question/Suggestion
- Confidence: 0.91 (91%)
- Suggested Response: "Great question! We appreciate your interest. We'll definitely consider your suggestion for future content. Thank you for the idea! 💡"

Example 5: Threat
Input: "I'll report you if this continues."
Output:
- Category: Threat
- Confidence: 0.89 (89%)
- Suggested Response: "We take all concerns seriously. If you believe our content violates any platform guidelines, please report it through the official channels. We're committed to following all community standards. Thank you for bringing this to our attention. 📋"

Example 6: Emotional
Input: "This reminded me of my childhood days."
Output:
- Category: Emotional
- Confidence: 0.87 (87%)
- Suggested Response: "We're deeply touched that our content resonated with you on such a personal level. Thank you for sharing that with us. It's moments like these that remind us why we create. 💙"

Example 7: Support
Input: "Keep going, you're improving every day!"
Output:
- Category: Support
- Confidence: 0.90 (90%)
- Suggested Response: "Thank you for your encouragement! Your support keeps us going. We'll keep creating and improving! 💪"

Example 8: Irrelevant/Spam
Input: "Follow me for instant followers!"
Output:
- Category: Irrelevant/Spam
- Confidence: 0.93 (93%)
- Suggested Response: "Thank you for your comment. We focus on maintaining a space for meaningful discussions about our content. If you have feedback about our work, we'd love to hear it! 🎯"

📈 Model Performance

The model achieves high accuracy in distinguishing between different comment types:
- Overall Accuracy: 80-90%+ (depending on training)
- Best Performance: Praise, Support, and Spam detection
- Key Strength: Effectively separates Constructive Criticism from Hate/Abuse

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
