from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import date, time
from dotenv import load_dotenv
import os

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

# Configuration
DOCTORS_DB = "doctor_database_1.csv"
df = pd.read_csv(DOCTORS_DB) if os.path.exists(DOCTORS_DB) else pd.DataFrame(columns=["name", "specialization", "city", "phone", "rating"])
# Default Cities
DEFAULT_CITIES = ["Kolkata", "Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai"]

# Disease to Specialist Mapping
DISEASE_SPECIALIST_MAP = {
    'asthma': 'Pulmonologist',
    'pneumonia': 'Pulmonologist',
    'covid': 'Infectious Disease',
    'flu': 'General Physician',
    'common_cold': 'General Physician',
    'diabetes': 'Endocrinologist',
    'hypertension': 'Cardiologist',
    'heart_disease': 'Cardiologist',
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

# Initialize Doctor Database - Modified to work with your CSV
def init_doctor_db():
    if not os.path.exists(DOCTORS_DB):
        # Create a DataFrame with the expected columns if file doesn't exist
        pd.DataFrame(columns=["name", "specialization", "city", "phone", "rating"]).to_csv(DOCTORS_DB, index=False)
    else:
        # Ensure existing CSV has the required columns
        df = pd.read_csv(DOCTORS_DB)
        required_cols = ["name", "specialization", "city", "phone"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV file is missing required column: {col}")
        
        # Add rating column if missing
        if "rating" not in df.columns:
            df["rating"] = 0.0
            df.to_csv(DOCTORS_DB, index=False)

# Data Loading Functions - Modified for your CSV
def load_doctors():
    """Load doctor data from your CSV file"""
    try:
        if os.path.exists(DOCTORS_DB):
            df = pd.read_csv(DOCTORS_DB)
            # Ensure required columns exist
            required_cols = ["name", "specialization", "city", "phone"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Add rating if missing
            if "rating" not in df.columns:
                df["rating"] = 0.0
                
            return df
        return pd.DataFrame(columns=["name", "specialization", "city", "phone", "rating"])
    except Exception as e:
        print(f"Error loading doctors: {str(e)}")
        return pd.DataFrame(columns=["name", "specialization", "city", "phone", "rating"])

def get_unique_cities():
    """Get all unique cities from the doctor database, including defaults"""
    try:
        # Load the doctor data
        df = load_doctors()
        
        # Get unique cities from the database
        db_cities = df['city'].dropna().unique().tolist()
        
        # Combine with default cities and remove duplicates
        all_cities = list(set(db_cities + DEFAULT_CITIES))
        
        # Sort alphabetically and return
        return sorted(all_cities)
    
    except Exception as e:
        print(f"Error getting cities: {str(e)}")
        # Return just the default cities if there's an error
        return sorted(DEFAULT_CITIES)


def get_unique_specializations():
    """Get all unique specializations from the doctor database"""
    try:
        # Load the doctor data
        df = load_doctors()
        
        # Get unique specializations, remove empty/NA values
        specializations = df['specialization'].dropna().unique().tolist()
        
        # Also include all specializations from disease mapping
        disease_specializations = list(set(DISEASE_SPECIALIST_MAP.values()))
        
        # Combine and remove duplicates
        all_specializations = list(set(specializations + disease_specializations))
        
        # Sort alphabetically and return
        return sorted(all_specializations)
    
    except Exception as e:
        print(f"Error getting specializations: {str(e)}")
        # Return just the specializations from disease mapping if error
        return sorted(list(set(DISEASE_SPECIALIST_MAP.values())))

def save_doctors(df):
    """Save doctor data to your CSV"""
    try:
        df.to_csv(DOCTORS_DB, index=False)
    except Exception as e:
        print(f"Error saving doctors: {str(e)}")

# Configuration and data loading functions remain the same...

# Rename the search function to avoid conflict
def search_doctors(city=None, specialization=None, min_rating=None):
    """Search doctors with flexible matching"""
    df = load_doctors()
    
    # Convert all to lowercase for case-insensitive search
    df['city_lower'] = df['city'].str.lower()
    df['spec_lower'] = df['specialization'].str.lower()
    
    # Apply filters
    if city:
        df = df[df['city_lower'] == city.lower()]
    if specialization:
        df = df[df['spec_lower'] == specialization.lower()]
    if min_rating:
        df = df[df['rating'] >= float(min_rating)]
    
    # Clean up
    df = df.drop(columns=['city_lower', 'spec_lower'])
    
    return df.sort_values(['rating', 'name'], ascending=[False, True])

# Update the route to use the renamed function
@app.route('/find-doctors', methods=['GET', 'POST'])
def find_doctors_route():  # Renamed from find_doctors
    if request.method == 'POST':
        specialization = request.form.get('specialization', '').strip()
        
        doctors_df = search_doctors(specialization=specialization)
        doctors = doctors_df.to_dict('records')
        return render_template('findoc.html', 
                            doctors=doctors,
                            specializations=get_unique_specializations())
    
    return render_template('findoc.html',
                         specializations=get_unique_specializations())

@app.route('/doctors', methods=['GET', 'POST'])
def doctors():
    init_doctor_db()
    
    # Get form data with proper defaults
    city = request.form.get('city', '').strip()
    disease = request.form.get('disease', '').strip().lower()
    specialization = request.form.get('specialization', '').strip()
    use_suggested = request.form.get('use_suggested', 'false') == 'true'
    
    # Auto-suggest specialization if disease is selected
    suggested_spec = ""
    if disease and use_suggested:
        suggested_spec = DISEASE_SPECIALIST_MAP.get(disease, "")
        if suggested_spec:
            specialization = suggested_spec
    
    # Perform search if any criteria provided
    doctors = []
    search_performed = bool(city or specialization or disease)
    
    if search_performed:
        doctors_df = search_doctors(
            city=city if city else None,
            specialization=specialization if specialization else None
        )
        doctors = doctors_df.to_dict('records')
    
    # Prepare context with all needed variables
    context = {
        'cities': get_unique_cities(),
        'disease_specialist_map': DISEASE_SPECIALIST_MAP,
        'specializations': get_unique_specializations(),
        'doctors': doctors,
        'selected_city': city,
        'selected_spec': specialization,
        'suggested_spec': suggested_spec,
        'search_performed': search_performed
    }
    
    return render_template('findoc.html', **context)

# The add_doctor route remains the same...

@app.route('/doctors/add', methods=['POST'])
def add_doctor():
    try:
        data = {
            'name': request.form['name'],
            'specialization': request.form['specialization'],
            'city': request.form['city'],
            'phone': request.form['phone'],
            'rating': float(request.form.get('rating', 0))
        }
        df = load_doctors()
        new_entry = pd.DataFrame([data])
        updated = pd.concat([df, new_entry], ignore_index=True)
        save_doctors(updated)
        return redirect(url_for('doctors'))
    except Exception as e:
        return f"Error adding doctor: {str(e)}", 400

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

# Load environment variables from .env file
load_dotenv()

# Get Gemini API Key from environment
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("❌ Gemini API Key not found. Please set GEMINI_API_KEY in your .env file.")

# Configure Gemini
genai.configure(api_key=API_KEY)

# Model initialize
model = genai.GenerativeModel("gemini-1.5-flash")

def chatbot_reply(user_input):
    user_input = user_input.lower()

    # Medicine Info
    if "medicine" in user_input or "drug" in user_input:
        med_name = user_input.replace("medicine", "").replace("drug", "").strip()
        if med_name:
            return get_medicine_info(med_name)
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
    app.run(debug=True, port=3000)