from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func

class Reminder(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  data = db.Column(db.String(10000))
  date = db.Column(db.DateTime(timezone=True), default=func.now())
  type = db.Column(db.String(50))
  user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
  

class User(db.Model, UserMixin):
  id = db.Column(db.Integer, primary_key=True)
  email = db.Column(db.String(150), unique=True)
  username = db.Column(db.String(150))
  password = db.Column(db.String(150))
  reminders = db.relationship('Reminder', backref='user', lazy=True)