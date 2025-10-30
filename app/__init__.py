from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate  # ✅ Added for migrations
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize SQLAlchemy
db = SQLAlchemy()
migrate = Migrate()  # ✅ Added migrate instance

def create_app():
    app = Flask(__name__)

    # 🔹 Database Configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Initialize database and migrations
    db.init_app(app)
    migrate.init_app(app, db)  # ✅ Enables flask db commands

    # Import models AFTER db init
    from .models import StartupIdea  

    # Register Blueprints
    from .routes import main
    app.register_blueprint(main)

    # Debug info (optional)
    print("✅ Connected to Database:", app.config['SQLALCHEMY_DATABASE_URI'])

    return app
