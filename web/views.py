from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from .models import Reminder
from . import db
import json

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
  if request.method == 'POST':
    reminder = request.form.get('reminder')
    if len(reminder) < 1:
      flash("Reminder is too short!", category='error')
    else:
      new_reminder = Reminder(data=reminder, user_id=current_user.id)
      db.session.add(new_reminder)
      db.session.commit()
      flash("Reminder added!", category='success')
  reminders = Reminder.query.filter_by(user_id=current_user.id).all()
  if not reminders:
    flash("No reminders found.", category='info')
  else:
    flash(f"{len(reminders)} reminders found.", category='info')
    
  return render_template('home.html', user=current_user)

@views.route('/delete_reminder', methods=['POST'])
@login_required
def delete_reminder():
  try:
    reminder = json.loads(request.data)
    reminder_id = reminder['reminderId']
    reminder = Reminder.query.get(reminder_id)
    if reminder and reminder.user_id == current_user.id:
      db.session.delete(reminder)
      db.session.commit()
      flash("Reminder deleted!", category='success')
    else:
      flash("Reminder not found or access denied!", category='error')
  except Exception as e:
    flash("Error deleting reminder!", category='error')
  
  return jsonify({})
