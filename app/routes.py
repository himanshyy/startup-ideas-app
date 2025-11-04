from flask import Blueprint, render_template, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
import requests
import json
import re
import logging
import numpy as np
from . import db
from .models import StartupIdea

# -------------------------------------------------------------------
# Blueprint & Logging
# -------------------------------------------------------------------
main = Blueprint("main", __name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Environment Variables
# -------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HF_API_TOKEN")

if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY missing in environment variables.")
if not SERPAPI_KEY:
    logger.warning("⚠️ SERPAPI_KEY missing in environment variables.")
if not HF_API_KEY:
    logger.warning("⚠️ HF_API_KEY missing in environment variables.")

# -------------------------------------------------------------------
# Model Config
# -------------------------------------------------------------------
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.0-flash"

HF_MODEL_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# -------------------------------------------------------------------
# Embedding Function (External API → Light)
# -------------------------------------------------------------------
def get_embedding(text):
    """Generate lightweight embeddings via Hugging Face API."""
    try:
        response = requests.post(
            HF_MODEL_URL,
            headers=HF_HEADERS,
            json={"inputs": text},
            timeout=15
        )
        if response.status_code != 200:
            logger.error(f"Hugging Face API Error: {response.text}")
            return []
        result = response.json()
        # API sometimes wraps the embedding in a nested list
        return result[0] if isinstance(result[0], list) else result
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return []

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2:
        return 0.0
    a, b = np.array(vec1), np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def clamp(n, smallest=0, largest=100):
    return max(smallest, min(n, largest))

def compute_market_feasibility(competitor_count, avg_similarity, top_recent_ratio=0.0):
    base = 70
    comp_penalty = clamp(competitor_count * 5, 0, 50)
    sim_penalty = (avg_similarity / 100.0) * 25
    recency_boost = (top_recent_ratio * 30.0)
    score = base - comp_penalty - sim_penalty + recency_boost
    return int(clamp(round(score), 0, 100))

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@main.route("/")
def index():
    return render_template("index.html")

# -------------------------------------------------------------------
# Analyze Idea Route
# -------------------------------------------------------------------
@main.route("/analyze", methods=["POST"])
def analyze_idea():
    payload = request.get_json() or {}
    user_idea = (payload.get("idea") or "").strip()

    if not user_idea:
        return jsonify({"error": "No idea provided"}), 400

    serp_results, competitors = [], []
    competitor_count = avg_similarity = top_recent_ratio = 0

    # 🔍 STEP 1 — SERP API Search
    serpapi_url = (
        f"https://serpapi.com/search.json?"
        f"engine=google&q={requests.utils.quote(user_idea + ' startup 2025')}"
        f"&api_key={SERPAPI_KEY}"
    )

    try:
        resp = requests.get(serpapi_url, timeout=10)
        serp_data = resp.json()
    except Exception as e:
        logger.error(f"SERP API Error: {e}")
        serp_data = {}

    if serp_data.get("organic_results"):
        organic = serp_data["organic_results"][:10]
        competitor_count = len(organic)
        similarities, recent_count = [], 0

        for item in organic:
            title = item.get("title", "")
            snippet = item.get("snippet", "") or ""
            link = item.get("link", "#")

            try:
                emb_idea = get_embedding(user_idea)
                emb_text = get_embedding(title + " " + snippet)
                sim = cosine_similarity(emb_idea, emb_text)
                sim_pct = round(sim * 100, 2)
            except Exception:
                sim_pct = 0.0

            similarities.append(sim_pct)
            serp_results.append({
                "title": title,
                "snippet": snippet,
                "link": link,
                "similarity": sim_pct
            })

            competitors.append({
                "name": title.split(" – ")[0].strip(),
                "link": link,
                "similarity": sim_pct,
                "snippet": snippet
            })

            if any(y in snippet.lower() for y in ["2025", "2024", "2023"]):
                recent_count += 1

        avg_similarity = sum(similarities) / len(similarities)
        top_recent_ratio = recent_count / competitor_count if competitor_count else 0

    # 🧮 STEP 2 — Market Feasibility
    market_feasibility = compute_market_feasibility(
        competitor_count, avg_similarity, top_recent_ratio
    )

    # 🌍 STEP 3 — Gemini Trend Analysis
    try:
        context = "\n".join([f"- {r['title']} ({r['similarity']}%)" for r in serp_results[:5]]) or "No competitors."
        prompt = f"""
Analyze the startup idea "{user_idea}" with the following competitor data:
{context}

Return JSON only:
{{
  "market_trend": "short summary",
  "top_competitors": [{{"name": "string", "type": "High/Moderate/Emerging"}}],
  "buzz_index": 0-100,
  "market_chance": 0-100,
  "market_recommendations": ["3 concise points"]
}}
"""
        gem_resp = genai.GenerativeModel(MODEL_NAME).generate_content(prompt)
        gem_text = re.sub(r"```(json)?", "", gem_resp.text).strip()
        gem_json = json.loads(gem_text)

        market_trend = gem_json.get("market_trend", "Data unavailable.")
        buzz_index = int(clamp(gem_json.get("buzz_index", 60)))
        market_chance = int(clamp(gem_json.get("market_chance", 60)))
        market_recommendations = gem_json.get("market_recommendations", [])
        top_competitors_ai = gem_json.get("top_competitors", [])
    except Exception as e:
        logger.warning(f"Gemini Market Analysis Error: {e}")
        market_trend = "Unable to fetch market trend."
        buzz_index = 55
        market_chance = market_feasibility
        market_recommendations = [
            "Validate idea with early adopters",
            "Focus on differentiation",
            "Target niche audience first"
        ]
        top_competitors_ai = [{"name": c["name"], "type": "Moderate"} for c in competitors[:3]]

    # 💡 STEP 4 — Innovation Metrics
    try:
        quick_prompt = f"""
For startup idea "{user_idea}", return JSON:
{{
  "ai_potential": 0-100,
  "innovation": 0-100,
  "uniqueness": 0-100,
  "risk": 0-100,
  "tech_complexity": 0-100,
  "success_probability": 0-100
}}
"""
        quick_resp = genai.GenerativeModel(MODEL_NAME).generate_content(quick_prompt)
        quick_json = json.loads(re.sub(r"```(json)?", "", quick_resp.text).strip())

        ai_potential = int(clamp(quick_json.get("ai_potential", 60)))
        innovation = int(clamp(quick_json.get("innovation", 65)))
        uniqueness = int(clamp(quick_json.get("uniqueness", 60)))
        risk = int(clamp(quick_json.get("risk", 50)))
        tech_complexity = int(clamp(quick_json.get("tech_complexity", 50)))
        success_probability = int(clamp(quick_json.get("success_probability", market_feasibility)))
    except Exception as e:
        logger.warning(f"Gemini Metrics Error: {e}")
        ai_potential, innovation, uniqueness, risk, tech_complexity, success_probability = (
            60, 65, 60, 50, 50, market_feasibility
        )

    # 💰 STEP 5 — Investor Readiness
    investor_readiness = int(clamp(round(
        0.25 * market_feasibility +
        0.25 * innovation +
        0.20 * ai_potential +
        0.15 * (100 - risk) +
        0.15 * buzz_index
    )))

    try:
        summary_prompt = f"""
Explain in one sentence why idea "{user_idea}" has investor readiness {investor_readiness}%.
Return JSON only: {{"investor_summary": "short summary"}}
"""
        resp = genai.GenerativeModel(MODEL_NAME).generate_content(summary_prompt)
        exp_json = json.loads(re.sub(r"```(json)?", "", resp.text).strip())
        investor_summary = exp_json.get("investor_summary", "Promising early-stage opportunity.")
    except Exception as e:
        logger.warning(f"Gemini Summary Error: {e}")
        investor_summary = "Promising early-stage opportunity."

    # ✅ Combine competitors
    final_competitors = []
    seen = set()
    for c in competitors + top_competitors_ai:
        name = c.get("name")
        if name and name not in seen:
            seen.add(name)
            final_competitors.append(c)

    # 💾 Save to DB
    try:
        new_entry = StartupIdea(
            idea_text=user_idea,
            ai_potential=ai_potential,
            innovation=innovation,
            uniqueness=uniqueness,
            risk=risk,
            tech_complexity=tech_complexity,
            market_feasibility=market_feasibility,
            investor_readiness=investor_readiness,
            market_chance=market_chance
        )
        db.session.add(new_entry)
        db.session.commit()
    except Exception as e:
        logger.error(f"Database insert failed: {e}")
        db.session.rollback()

    # ✅ Final Response
    return jsonify({
        "market_trend": market_trend,
        "market_chance": market_chance,
        "buzz_index": buzz_index,
        "market_recommendations": market_recommendations,
        "competitors": final_competitors[:5],
        "ai_potential": ai_potential,
        "innovation": innovation,
        "uniqueness": uniqueness,
        "risk": risk,
        "tech_complexity": tech_complexity,
        "success_probability": success_probability,
        "market_feasibility": market_feasibility,
        "investor_readiness": investor_readiness,
        "investor_summary": investor_summary,
        "search_signals": {
            "competitor_count": competitor_count,
            "avg_similarity": round(avg_similarity, 2),
            "top_recent_ratio": round(top_recent_ratio, 2)
        },
        "results": serp_results
    })

# -------------------------------------------------------------------
# History Route
# -------------------------------------------------------------------
@main.route("/history", methods=["GET"])
def history():
    try:
        ideas = StartupIdea.query.order_by(StartupIdea.timestamp.desc()).limit(10).all()
        return jsonify([
            {
                "id": i.id,
                "idea": i.idea_text,
                "ai_potential": i.ai_potential,
                "market_feasibility": i.market_feasibility,
                "investor_readiness": i.investor_readiness,
                "timestamp": i.timestamp.strftime("%Y-%m-%d %H:%M")
            } for i in ideas
        ])
    except Exception as e:
        logger.error(f"History fetch failed: {e}")
        return jsonify({"error": "Unable to fetch history"}), 500
