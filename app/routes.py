from flask import Blueprint, render_template, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import requests
import json
import re
from . import db
from .models import StartupIdea

main = Blueprint("main", __name__)

# 🔑 Load API Keys
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.0-flash"
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🧮 Clamp utility
def clamp(n, smallest=0, largest=100):
    return max(smallest, min(n, largest))

@main.route("/")
def index():
    return render_template("index.html")

# 📊 Market Feasibility
def compute_market_feasibility(competitor_count, avg_similarity, top_recent_ratio=0.0):
    base = 70
    comp_penalty = clamp(competitor_count * 5, 0, 50)
    sim_penalty = (avg_similarity / 100.0) * 25
    recency_boost = (top_recent_ratio * 30.0)
    score = base - comp_penalty - sim_penalty + recency_boost
    return int(clamp(round(score), 0, 100))

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
    except Exception:
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
                embeddings = model.encode([user_idea, title + " " + snippet], convert_to_tensor=True)
                sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
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

            if re.search(r"\b[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*\b", title):
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
    market_feasibility_heuristic = compute_market_feasibility(
        competitor_count, avg_similarity, top_recent_ratio
    )

    # 🌍 STEP 3 — Gemini Market & Competitor Intelligence
    try:
        context = "\n".join([f"- {r['title']} ({r['similarity']}%)" for r in serp_results[:5]]) or "No similar startups."
        prompt = f"""
You are a startup analyst. Based on:
Idea: "{user_idea}"
Competitor Data:
{context}

Return ONLY JSON:
{{
  "market_trend": "short summary of 2025 trend",
  "top_competitors": [{{"name": "string", "type": "High/Moderate/Emerging"}}],
  "buzz_index": 0-100,
  "market_chance": 0-100,
  "market_recommendations": ["3 short points"]
}}
"""
        gem_resp = genai.GenerativeModel(MODEL_NAME).generate_content(prompt)
        gem_text = re.sub(r"```(json)?", "", gem_resp.text).strip()
        gem_json = json.loads(gem_text)

        market_trend = gem_json.get("market_trend", "Trend data unavailable.")
        buzz_index = int(clamp(gem_json.get("buzz_index", 60)))
        market_chance = int(clamp(gem_json.get("market_chance", 60)))
        market_recommendations = gem_json.get("market_recommendations", [])
        top_competitors_ai = gem_json.get("top_competitors", [])
    except Exception:
        market_trend = "Unable to fetch trend data."
        buzz_index = 55
        market_chance = market_feasibility_heuristic
        market_recommendations = [
            "Validate idea with early adopters",
            "Focus on differentiation",
            "Build strong community presence"
        ]
        top_competitors_ai = [{"name": c["name"], "type": "Moderate"} for c in competitors[:3]]

    # 💡 STEP 4 — AI / Innovation Metrics
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
        quick_text = re.sub(r"```(json)?", "", quick_resp.text).strip()
        quick_json = json.loads(quick_text)
        ai_potential = int(clamp(quick_json.get("ai_potential", 60)))
        innovation = int(clamp(quick_json.get("innovation", 65)))
        uniqueness = int(clamp(quick_json.get("uniqueness", 60)))
        risk = int(clamp(quick_json.get("risk", 50)))
        tech_complexity = int(clamp(quick_json.get("tech_complexity", 50)))
        success_probability = int(clamp(quick_json.get("success_probability", market_feasibility_heuristic)))
    except Exception:
        ai_potential, innovation, uniqueness, risk, tech_complexity, success_probability = (
            60, 65, 60, 50, 50, market_feasibility_heuristic
        )

    # 💰 STEP 5 — Investor Readiness Score
    investor_readiness = int(clamp(round(
        0.25 * market_feasibility_heuristic +
        0.25 * innovation +
        0.20 * ai_potential +
        0.15 * (100 - risk) +
        0.15 * buzz_index
    )))

    try:
        summary_prompt = f"""
Explain briefly why idea "{user_idea}" has investor readiness {investor_readiness}%.
Return ONLY JSON: {{"investor_summary": "1-line summary for investors"}}
"""
        resp = genai.GenerativeModel(MODEL_NAME).generate_content(summary_prompt)
        text = re.sub(r"```(json)?", "", resp.text).strip()
        exp_json = json.loads(text)
        investor_summary = exp_json.get("investor_summary", "Strong early-stage potential with balanced risk.")
    except Exception:
        investor_summary = "Strong early-stage potential with balanced risk."

    # ✅ Combine competitor data (SERP + AI)
    final_competitors = []
    seen = set()
    for c in competitors + top_competitors_ai:
        name = c.get("name")
        if name and name not in seen:
            seen.add(name)
            final_competitors.append(c)

    # 💾 STEP 6 — Save to Database
    new_entry = StartupIdea(
        idea_text=user_idea,
        ai_potential=ai_potential,
        innovation=innovation,
        uniqueness=uniqueness,
        risk=risk,
        tech_complexity=tech_complexity,
        market_feasibility=market_feasibility_heuristic,
        investor_readiness=investor_readiness,
        market_chance=market_chance
    )
    db.session.add(new_entry)
    db.session.commit()

    # ✅ Final JSON Response
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
        "market_feasibility": market_feasibility_heuristic,
        "investor_readiness": investor_readiness,
        "investor_summary": investor_summary,
        "search_signals": {
            "competitor_count": competitor_count,
            "avg_similarity": round(avg_similarity, 2),
            "top_recent_ratio": round(top_recent_ratio, 2)
        },
        "results": serp_results
    })


# 📜 STEP 7 — View Past Analyses
@main.route("/history", methods=["GET"])
def history():
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
