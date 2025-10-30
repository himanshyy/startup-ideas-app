from datetime import datetime
from . import db

class StartupIdea(db.Model):
    __tablename__ = 'startup_ideas'

    id = db.Column(db.Integer, primary_key=True)
    idea_text = db.Column(db.Text, nullable=False)
    ai_potential = db.Column(db.Integer)
    uniqueness = db.Column(db.Integer)
    risk = db.Column(db.Integer)
    innovation = db.Column(db.Integer)
    market_feasibility = db.Column(db.Integer)
    tech_complexity = db.Column(db.Integer)
    market_chance = db.Column(db.Integer)

    # 🆕 New fields for latest analysis
    investor_readiness = db.Column(db.Integer)
    buzz_index = db.Column(db.Integer)
    success_probability = db.Column(db.Integer)
    investor_summary = db.Column(db.Text)
    market_trend = db.Column(db.String(255))

    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<StartupIdea {self.id} - {self.idea_text[:30]}>"
