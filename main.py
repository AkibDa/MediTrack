from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import date, time
import os
from key import API_KEY

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
app.secret_key = 'qwertyuiopasdfghjklzxcvbnm'  # Change this for production

# -----------------------------
# Data Loading and Helper Functions
# -----------------------------

def load_data():
    df = pd.read_csv('archive/dataset.csv').fillna('none')
    df_precaution = pd.read_csv('archive/symptom_precaution.csv', index_col=0)
    df_description = pd.read_csv('archive/symptom_Description.csv', index_col=0)
    return df, df_precaution, df_description

def get_disease_info(disease, df_precaution, df_description):
    info = {}

    # --- Description lookup ---
    if "Disease" in df_description.columns:
        desc_row = df_description[df_description["Disease"].str.lower() == disease.lower()]
        if not desc_row.empty:
            info['description'] = desc_row['Description'].values[0]
        else:
            info['description'] = "No description available."
    else:
        if disease in df_description.index:
            info['description'] = df_description.loc[disease]['Description']
        else:
            info['description'] = "No description available."

    # --- Precaution lookup ---
    if "Disease" in df_precaution.columns:
        prec_row = df_precaution[df_precaution["Disease"].str.lower() == disease.lower()]
        if not prec_row.empty:
            precautions = [
                prec_row[col].values[0] 
                for col in prec_row.columns if col.startswith("Precaution")
            ]
            precautions = [p for p in precautions if isinstance(p, str) and p.strip()]
            info['precautions'] = precautions
        else:
            info['precautions'] = []
    else:
        if disease in df_precaution.index:
            precautions = [p for p in df_precaution.loc[disease] if pd.notna(p)]
            info['precautions'] = precautions
        else:
            info['precautions'] = []

    return info


def encode_symptoms(row, symptom_columns, all_symptoms):
    row_syms = set(
        str(row[col]).strip().lower().replace(" ", "_")
        for col in symptom_columns
        if row[col] != 'none'
    )
    return [1 if sym in row_syms else 0 for sym in all_symptoms]

def predict_disease(symptoms_list, df, df_precaution, df_description):
    symptom_columns = [col for col in df.columns if str(col).startswith("Symptom_")]
    all_symptoms = sorted(set(
        str(sym).strip().lower().replace(" ", "_")
        for col in symptom_columns
        for sym in df[col].unique()
        if sym != 'none'
    ))

    input_vector = [1 if symptom in symptoms_list else 0 for symptom in all_symptoms]
    if sum(input_vector) < 1:
        return None, "Please select at least one valid symptom."

    X = df.apply(lambda row: encode_symptoms(row, symptom_columns, all_symptoms),
                 axis=1, result_type='expand')
    X.columns = all_symptoms
    y = df['Disease']

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    input_df = pd.DataFrame([input_vector], columns=all_symptoms)
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    top_indices = np.argsort(probabilities)[-3:][::-1]
    results = []
    for rank_idx, cls_idx in enumerate(top_indices, start=1):
        disease = classes[cls_idx]
        conf = probabilities[cls_idx]
        disease_info = get_disease_info(disease, df_precaution, df_description)
        results.append({
            'rank': rank_idx,
            'disease': disease,
            'confidence': f"{conf * 100:.2f}%",
            **disease_info
        })

    return results, None

# -----------------------------
# Doctors Database
# -----------------------------

DISEASE_SPECIALIST_MAP = {
    'asthma': 'Pulmonologist',
    'pneumonia': 'Pulmonologist',
    'covid': 'Infectious Disease',
    'flu': 'General Physician',
    'common_cold': 'General Physician',
    'diabetes': 'Endocrinologist',
    'hypertension': 'Cardiologist',
    'heart': 'Cardiologist',
    'migraine': 'Neurologist',
    'stroke': 'Neurologist',
    'depression': 'Psychiatrist',
    'anxiety': 'Psychiatrist',
    'dermatitis': 'Dermatologist',
    'acne': 'Dermatologist',
    'arthritis': 'Rheumatologist',
    'tb': 'Pulmonologist',
    'thyroid': 'Endocrinologist',
}

DOCTORS_DB = pd.DataFrame([
    {"name": "CityCare Hospital", "specialization": "General Physician", "city": "Kolkata", "phone": "033-4000-1000"},
    {"name": "Apollo Multi-Speciality", "specialization": "Cardiologist", "city": "Kolkata", "phone": "033-3500-2222"},
    {"name": "Fortis Health", "specialization": "Neurologist", "city": "Kolkata", "phone": "033-3344-7788"},
    {"name": "MedLife Clinic", "specialization": "Pulmonologist", "city": "Delhi", "phone": "011-4100-9090"},
    {"name": "Wellness Point", "specialization": "Endocrinologist", "city": "Delhi", "phone": "011-2255-6677"},
    {"name": "Health+ Clinic", "specialization": "Dermatologist", "city": "Mumbai", "phone": "022-4400-8800"},
    {"name": "Sunrise Hospital", "specialization": "General Physician", "city": "Bengaluru", "phone": "080-4455-9900"},
    {"name": "Care & Cure", "specialization": "Pulmonologist", "city": "Hyderabad", "phone": "040-6677-7788"},
])

# -----------------------------
# Doctors CSV (Optional)
# -----------------------------

DOCTOR_CSV = "doctor_database.csv"   

def load_doctors():
    try:
        if os.path.exists(DOCTOR_CSV):
            df = pd.read_csv(DOCTOR_CSV)
            return df
        return pd.DataFrame(columns=["doctor_name", "specialization", "city", "phone_number"])
    except Exception:
        return pd.DataFrame(columns=["doctor_name", "specialization", "city", "phone_number"])

def save_doctors(df):
    df.to_csv(DOCTOR_CSV, index=False)

# -----------------------------
# Reminders
# -----------------------------

REMINDERS_CSV = "reminders.csv"

def load_reminders():
    if os.path.exists(REMINDERS_CSV):
        return pd.read_csv(REMINDERS_CSV)
    return pd.DataFrame(columns=["when_date", "when_time", "type", "text"])

def save_reminders(df):
    df.to_csv(REMINDERS_CSV, index=False)

# -----------------------------
# Chatbot Helper
# -----------------------------

def get_medicine_info(med_name):
    return f"{med_name.title()} is a common medicine. (Demo info, integrate API here)"

import google.generativeai as genai

# Gemini API set koro (API_KEY replace koro tomar key diye)
genai.configure(api_key="API_KEY")

# Model initialize
model = genai.GenerativeModel("gemini-1.5-flash")

def chatbot_reply(user_input):
    user_input = user_input.lower()

    # Medicine Info
    if "medicine" in user_input or "drug" in user_input:
        med_name = user_input.replace("medicine", "").replace("drug", "").strip()
        if med_name:
            return get_medicine_info(med_name)   # <-- ekhane tomar DB function use hobe
        else:
            return "Please tell me the medicine name you want to know about."

    # Doctor Info
    elif "doctor" in user_input:
        for spec in DOCTORS_DB["specialization"].unique():
            if spec.lower() in user_input:
                filtered = DOCTORS_DB[DOCTORS_DB["specialization"].str.lower() == spec.lower()]
                if not filtered.empty:
                    return "Here are some doctors:\n" + "\n".join(
                        [f"{row['name']} ({row['city']}) - {row['phone']}" for _, row in filtered.iterrows()]
                    )
                else:
                    return f"Sorry, I couldn't find any {spec} doctors."
        return "Please tell me the specialization of the doctor you are looking for."

    # Default → Gemini AI
    else:
        try:
            response = model.generate_content(user_input)
            return response.text
        except Exception as e:
            return f"Error with Gemini API: {e}"

# -----------------------------
# Initialize Data
# -----------------------------

def initialize_data():
    try:
        df, df_precaution, df_description = load_data()
        app.df = df
        app.df_precaution = df_precaution
        app.df_description = df_description
        
        symptom_columns = [col for col in df.columns if str(col).startswith("Symptom_")]
        app.ALL_SYMPTOMS = sorted(set(
            str(sym).strip().lower().replace(" ", "_")
            for col in symptom_columns
            for sym in df[col].unique()
            if sym != 'none'
        ))
    except Exception as e:
        app.logger.error(f"Failed to load data: {e}")
        raise

with app.app_context():
    initialize_data()

# -----------------------------
# Routes
# -----------------------------

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/symptom_checker', methods=['GET', 'POST'])
def symptom_checker():
    if request.method == 'POST':
        selected_symptoms = request.form.get("symptoms", "").split(",")
        selected_symptoms = [s.strip() for s in selected_symptoms if s.strip()]
        results, error = predict_disease(selected_symptoms, app.df, app.df_precaution, app.df_description)
        
        if error:
            return render_template('symptom_checker.html', 
                                 symptoms=app.ALL_SYMPTOMS,
                                 error=error)
        else:
            session['last_predictions'] = results
            session['last_selected_symptoms'] = selected_symptoms
            session['last_disease_for_doctors'] = results[0]['disease']
            
            return render_template('symptom_checker.html',
                                symptoms=app.ALL_SYMPTOMS,
                                results=results,
                                selected_symptoms=selected_symptoms)
    
    return render_template('symptom_checker.html', symptoms=app.ALL_SYMPTOMS)

@app.route('/doctors', methods=['GET', 'POST'])
def doctors():
    suggested_spec = "General Physician"
    last_dis = session.get('last_disease_for_doctors', '')
    
    if last_dis:
        key = last_dis.strip().lower().replace(" ", "_")
        for k, spec in DISEASE_SPECIALIST_MAP.items():
            if k in key:
                suggested_spec = spec
                break
    
    if request.method == 'POST':
        city = request.form.get('city', 'Kolkata')
        use_suggested = request.form.get('use_suggested', 'false') == 'true'
        spec = suggested_spec if use_suggested else request.form.get('specialization', 'General Physician')
    else:
        city = 'Kolkata'
        spec = suggested_spec
    
    filtered = DOCTORS_DB[(DOCTORS_DB['city'] == city) & (DOCTORS_DB['specialization'] == spec)]
    if filtered.empty:
        filtered = DOCTORS_DB[DOCTORS_DB['city'] == city]
    
    cities = sorted(DOCTORS_DB['city'].unique())
    specializations = sorted(DOCTORS_DB['specialization'].unique())
    
    return render_template('findoc.html',
                         doctors=filtered.to_dict('records'),
                         cities=cities,
                         specializations=specializations,
                         suggested_spec=suggested_spec,
                         selected_city=city,
                         selected_spec=spec)

@app.route('/reminders', methods=['GET', 'POST'])
def reminders():
    reminders_df = load_reminders()
    
    if request.method == 'POST':
        if 'add_reminder' in request.form:
            r_date = request.form['reminder_date']
            r_time = request.form['reminder_time']
            r_type = request.form['reminder_type']
            r_text = request.form['reminder_text']
            
            new_row = {
                "when_date": r_date,
                "when_time": r_time,
                "type": r_type,
                "text": r_text.strip()
            }
            
            reminders_df = pd.concat(
                [reminders_df, pd.DataFrame([new_row])],
                ignore_index=True
            )
            save_reminders(reminders_df)
        
        elif 'delete_reminder' in request.form:
            del_index = int(request.form['delete_index'])
            reminders_df = reminders_df.drop(reminders_df.index[del_index]).reset_index(drop=True)
            save_reminders(reminders_df)
    
    return render_template('remind.html', reminders=reminders_df.to_dict('records'))

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot_page():
    if 'chat' not in session:
        session['chat'] = []
    
    if request.method == 'POST':
        user_msg = request.form.get('user_message', '').strip()
        if user_msg:
            session['chat'].append(('You', user_msg))
            reply = chatbot_reply(user_msg)
            session['chat'].append(('Assistant', reply))
            session.modified = True
    
    return render_template('chatbot.html', chat=session.get('chat', []))

@app.route('/explore')
def explore():
    return render_template('explore.html')

# -----------------------------
# Auth Routes
# -----------------------------

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if email and password:
            session['user'] = email
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Please fill in all fields!', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        
        if not all([username, email, password1, password2]):
            flash('Please fill in all fields!', 'error')
        elif password1 != password2:
            flash('Passwords do not match!', 'error')
        elif len(password1) < 6:
            flash('Password must be at least 6 characters!', 'error')
        else:
            session['user'] = email
            flash('Account created successfully!', 'success')
            return redirect(url_for('index'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=7000)