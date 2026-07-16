import os
import json
import random
import datetime
from flask import Flask, render_template_string, request, jsonify
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

app = Flask(__name__)

# ── IBM watsonx.ai Configuration ──────────────────────────────────────────────
WATSONX_API_KEY = os.environ.get("WATSONX_API_KEY", "your-watsonx-api-key")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "your-project-id")
WATSONX_URL = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

def get_granite_model(model_id="ibm/granite-4-h-small"):
    credentials = Credentials(api_key=WATSONX_API_KEY, url=WATSONX_URL)
    return ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=WATSONX_PROJECT_ID,
    )

def granite_generate(prompt: str, model_id="ibm/granite-4-h-small") -> str:
    try:
        model = get_granite_model(model_id)
        messages = [{"role": "user", "content": prompt}]
        response = model.chat(
            messages=messages,
            params={
                GenParams.MAX_NEW_TOKENS: 900,
                GenParams.TEMPERATURE: 0.75,
                GenParams.TOP_P: 0.9,
            }
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[IBM Granite Error]: {str(e)}"

# ── Simulated Social Data (used to enrich prompts) ────────────────────────────
PLATFORMS = ["X (Twitter)", "LinkedIn", "Instagram", "Facebook"]

def simulate_social_data(topic: str, platform: str = "all"):
    return {
        "topic": topic,
        "platform": platform,
        "trending_hashtags": [f"#{topic.replace(' ','')}", "#AITrends", "#Innovation", "#DigitalMarketing", "#Tech2025"],
        "engagement_rate": f"{random.uniform(2.5, 9.8):.2f}%",
        "total_mentions": random.randint(12000, 980000),
        "likes": random.randint(5000, 500000),
        "shares": random.randint(1000, 120000),
        "comments": random.randint(800, 90000),
        "top_keywords": [topic, "AI", "Growth", "Strategy", "Digital", "Future"],
        "sentiment": random.choice(["Positive (72%)", "Neutral (18%)", "Negative (10%)"]),
        "peak_time": random.choice(["9 AM – 11 AM", "12 PM – 2 PM", "6 PM – 9 PM"]),
        "audience_age": random.choice(["18–24 (35%)", "25–34 (40%)", "35–44 (18%)"]),
        "platforms_active": PLATFORMS if platform == "all" else [platform],
        "viral_posts": random.randint(3, 28),
        "influencer_mentions": random.randint(10, 200),
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT 1 – Social Media Insights Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.route("/api/agent/insights", methods=["POST"])
def agent_insights():
    data = request.json or {}
    topic = data.get("topic", "Artificial Intelligence")
    platform = data.get("platform", "all")
    social = simulate_social_data(topic, platform)
    prompt = f"""You are a Social Media Insights Agent powered by IBM Granite AI.
Analyze the following live social media data and generate a comprehensive insights report.

Topic: {topic}
Platforms: {', '.join(social['platforms_active'])}
Total Mentions: {social['total_mentions']:,}
Trending Hashtags: {', '.join(social['trending_hashtags'])}
Top Keywords: {', '.join(social['top_keywords'])}
Engagement Rate: {social['engagement_rate']}
Likes: {social['likes']:,} | Shares: {social['shares']:,} | Comments: {social['comments']:,}
Sentiment: {social['sentiment']}
Peak Engagement Time: {social['peak_time']}
Audience Age Group: {social['audience_age']}
Viral Posts: {social['viral_posts']} | Influencer Mentions: {social['influencer_mentions']}

Generate:
1. A concise executive summary of current social media trends for this topic
2. Top performing content categories and why they resonate
3. Cross-platform comparison and which platform dominates
4. Audience engagement insights and behavioral patterns
5. Top 5 actionable recommendations for brands or marketers

Be specific, data-driven, and professional. Format with clear numbered sections."""
    result = granite_generate(prompt)
    return jsonify({"insights": result, "raw_data": social})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT 2 – Content Generation Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.route("/api/agent/content", methods=["POST"])
def agent_content():
    data = request.json or {}
    brand = data.get("brand", "MyBrand")
    topic = data.get("topic", "Product Launch")
    tone = data.get("tone", "professional")
    platforms = data.get("platforms", ["X (Twitter)", "LinkedIn", "Instagram"])
    goal = data.get("goal", "increase engagement")
    prompt = f"""You are a Creative Content Generation Agent powered by IBM Granite AI.
Your mission is to craft high-performing, platform-optimized social media content.

Brand: {brand}
Topic / Campaign: {topic}
Tone: {tone}
Goal: {goal}
Target Platforms: {', '.join(platforms)}

Generate the following content package:

1. X (Twitter) Post (max 280 chars, punchy, with 3 hashtags)
2. LinkedIn Post (professional, 150–200 words, thought-leadership tone)
3. Instagram Caption (engaging, emotive, with 10 relevant hashtags)
4. Facebook Post (conversational, community-focused, 100–150 words)
5. Short-form Video Script Hook (15 seconds, attention-grabbing opening line)
6. Email Subject Line (for a campaign newsletter, curiosity-driven)
7. Content Calendar Suggestion (best days/times to post each piece)

Ensure all content is brand-safe, engaging, and optimized for the stated goal."""
    result = granite_generate(prompt)
    return jsonify({"content": result})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT 3 – Competitor Monitoring Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.route("/api/agent/competitor", methods=["POST"])
def agent_competitor():
    data = request.json or {}
    your_brand = data.get("your_brand", "YourBrand")
    competitors = data.get("competitors", ["CompetitorA", "CompetitorB", "CompetitorC"])
    industry = data.get("industry", "Technology")

    comp_data = {}
    for c in competitors:
        comp_data[c] = {
            "followers": random.randint(10000, 2000000),
            "avg_engagement": f"{random.uniform(1.2, 8.5):.2f}%",
            "posts_per_week": random.randint(3, 21),
            "top_content_type": random.choice(["Video", "Infographics", "Threads", "Carousels", "Stories"]),
            "sentiment_score": f"{random.uniform(55, 95):.1f}/100",
            "viral_campaigns": random.randint(1, 12),
        }

    prompt = f"""You are a Competitor Intelligence Agent powered by IBM Granite AI.
Perform a deep competitive analysis of social media strategies in the {industry} industry.

Your Brand: {your_brand}
Industry: {industry}

Competitor Intelligence Data:
{json.dumps(comp_data, indent=2)}

Provide:
1. Competitive Landscape Overview — who leads and why
2. Individual competitor strengths and weaknesses for each
3. Content strategy comparison (what formats/styles win)
4. Engagement gap analysis — where your brand can outperform
5. Identified whitespace opportunities competitors are missing
6. Recommended counter-strategies for {your_brand} to gain competitive advantage
7. 30-day action plan to close the competitive gap

Be analytical, strategic, and evidence-based."""
    result = granite_generate(prompt)
    return jsonify({"analysis": result, "competitor_data": comp_data})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT 4 – Trend Prediction Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.route("/api/agent/trends", methods=["POST"])
def agent_trends():
    data = request.json or {}
    industry = data.get("industry", "Technology")
    horizon = data.get("horizon", "30 days")
    region = data.get("region", "Global")

    rising_signals = [
        {"signal": "AI-generated video content", "velocity": f"+{random.randint(45,200)}%", "confidence": f"{random.randint(72,96)}%"},
        {"signal": "Short-form educational reels", "velocity": f"+{random.randint(30,180)}%", "confidence": f"{random.randint(68,94)}%"},
        {"signal": "Community-led brand campaigns", "velocity": f"+{random.randint(20,120)}%", "confidence": f"{random.randint(65,90)}%"},
        {"signal": "Social commerce integrations", "velocity": f"+{random.randint(35,150)}%", "confidence": f"{random.randint(70,92)}%"},
        {"signal": "Employee advocacy content", "velocity": f"+{random.randint(15,90)}%", "confidence": f"{random.randint(60,88)}%"},
    ]

    prompt = f"""You are a Trend Prediction Agent powered by IBM Granite AI.
Use predictive analytics to forecast social media trends for the {industry} industry.

Forecast Horizon: {horizon}
Region: {region}
Industry: {industry}

Rising Trend Signals Detected:
{json.dumps(rising_signals, indent=2)}

Generate a comprehensive trend forecast report:
1. Executive Trend Summary — macro shifts happening right now
2. Top 5 Predicted Trends with confidence scores and timeline
3. Emerging content formats that will dominate in {horizon}
4. Audience behavioral shifts — how users will engage differently
5. Platform-specific predictions (X, LinkedIn, Instagram, Facebook)
6. Early mover advantage — what brands should act on immediately
7. Risk trends — what's declining and what to avoid investing in
8. Recommended trend-riding strategy with specific tactics

Ground your predictions in current digital signals. Be bold, specific, and forward-looking."""
    result = granite_generate(prompt)
    return jsonify({"predictions": result, "signals": rising_signals})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT 5 – Campaign Optimization Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.route("/api/agent/campaign", methods=["POST"])
def agent_campaign():
    data = request.json or {}
    campaign_name = data.get("campaign_name", "Summer Launch 2025")
    brand = data.get("brand", "MyBrand")
    budget = data.get("budget", "$10,000")
    duration = data.get("duration", "30 days")
    objective = data.get("objective", "brand awareness")
    target_audience = data.get("target_audience", "millennials aged 25–34")

    current_metrics = {
        "ctr": f"{random.uniform(0.8, 4.2):.2f}%",
        "cpc": f"${random.uniform(0.30, 3.50):.2f}",
        "roas": f"{random.uniform(1.5, 6.8):.1f}x",
        "conversion_rate": f"{random.uniform(1.2, 8.5):.2f}%",
        "reach": random.randint(50000, 2000000),
        "impressions": random.randint(200000, 8000000),
        "engagement_rate": f"{random.uniform(2.0, 9.5):.2f}%",
        "cost_per_acquisition": f"${random.uniform(5, 75):.2f}",
    }

    prompt = f"""You are a Campaign Optimization Agent powered by IBM Granite AI.
Analyze and optimize the following social media campaign for maximum ROI and performance.

Campaign: {campaign_name}
Brand: {brand}
Budget: {budget}
Duration: {duration}
Primary Objective: {objective}
Target Audience: {target_audience}

Current Campaign Metrics:
{json.dumps(current_metrics, indent=2)}

Deliver a full campaign optimization report:
1. Campaign Performance Diagnosis — what the metrics reveal
2. Critical issues hurting performance and root causes
3. Budget reallocation recommendations across platforms
4. Audience targeting refinements — narrow or expand and why
5. Creative optimization — what ad formats/hooks to test next
6. A/B testing strategy with 3 specific test hypotheses
7. Bid strategy and scheduling optimizations
8. Projected improved KPIs if recommendations are implemented
9. 7-day quick-win action plan to boost results immediately

Be precise with numbers, percentages, and platform-specific tactics."""
    result = granite_generate(prompt)
    return jsonify({"optimization": result, "current_metrics": current_metrics})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ORCHESTRATOR – Multi-Agent Pipeline (IBM Langflow / Orchestrate style)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@app.route("/api/orchestrate", methods=["POST"])
def orchestrate():
    data = request.json or {}
    brand = data.get("brand", "MyBrand")
    topic = data.get("topic", "AI Innovation")
    industry = data.get("industry", "Technology")

    social = simulate_social_data(topic)
    prompt = f"""You are the IBM Langflow Orchestration Agent — the master coordinator of a multi-agent AI system for social media intelligence.

You have just received outputs from 5 specialized AI agents:
- Agent 1 (Insights): Analyzed {social['total_mentions']:,} mentions of "{topic}" with {social['engagement_rate']} engagement
- Agent 2 (Content): Generated platform-optimized content for {brand}
- Agent 3 (Competitors): Benchmarked 3 competitors in the {industry} space
- Agent 4 (Trends): Predicted top 5 trends for the next 30 days
- Agent 5 (Campaign): Optimized ongoing campaign with ROAS improvement opportunities

Brand: {brand} | Industry: {industry} | Topic: {topic}
Sentiment: {social['sentiment']} | Peak Time: {social['peak_time']}
Trending Hashtags: {', '.join(social['trending_hashtags'])}

As the Orchestrator, synthesize all agent outputs into a unified strategic brief:

1. EXECUTIVE SUMMARY — The most critical finding across all agents in 3 sentences
2. CROSS-AGENT SYNERGIES — Where agent outputs reinforce or contradict each other
3. UNIFIED STRATEGY — One cohesive 30-day social media strategy for {brand}
4. PRIORITY ACTION MATRIX — Top 10 actions ranked by impact and urgency (use High/Medium/Low)
5. SUCCESS METRICS — KPIs and benchmarks to measure strategy effectiveness
6. ORCHESTRATION NOTES — How the 5 agents should collaborate on an ongoing basis
7. FINAL RECOMMENDATION — The single most important thing {brand} should do this week

Format this as a polished C-suite ready strategic brief."""
    result = granite_generate(prompt)
    return jsonify({"orchestration": result, "agents_used": 5, "brand": brand, "topic": topic})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Route – Full Single-Page Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SocialPulse AI – Intelligent Social Media Agent</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet"/>
<style>
  :root{
    --ibm-blue:#0f62fe;--ibm-dark:#001d6c;--ibm-teal:#009d9a;
    --ibm-purple:#8a3ffc;--ibm-red:#da1e28;--ibm-green:#198038;
    --ibm-orange:#ff832b;--surface:#f4f4f4;--card-bg:#ffffff;
    --border:#e0e0e0;--text:#161616;--muted:#6f6f6f;
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--surface);color:var(--text);font-family:'IBM Plex Sans','Segoe UI',system-ui,sans-serif;font-size:14px;line-height:1.6;}

  /* ── Navbar ── */
  .navbar-brand-custom{display:flex;align-items:center;gap:10px;}
  .navbar-brand-custom .logo-pill{
    background:linear-gradient(135deg,var(--ibm-blue),var(--ibm-purple));
    color:#fff;font-weight:700;font-size:13px;padding:4px 10px;border-radius:20px;letter-spacing:.5px;
  }
  .brand-name{font-weight:700;font-size:18px;color:#fff;letter-spacing:-.3px;}
  .brand-sub{font-size:11px;color:rgba(255,255,255,.65);letter-spacing:.5px;text-transform:uppercase;}
  .navbar-custom{background:linear-gradient(135deg,#001141,#0f3460) !important;border-bottom:2px solid var(--ibm-blue);}

  /* ── Sidebar ── */
  .sidebar{
    width:240px;min-height:calc(100vh - 64px);background:#0a0a23;
    border-right:1px solid #1a1a3e;padding:20px 0;position:sticky;top:64px;
    flex-shrink:0;
  }
  .sidebar-section{padding:0 16px;margin-bottom:8px;font-size:10px;font-weight:700;
    color:rgba(255,255,255,.35);text-transform:uppercase;letter-spacing:1px;}
  .sidebar-item{
    display:flex;align-items:center;gap:10px;padding:10px 20px;
    color:rgba(255,255,255,.75);cursor:pointer;transition:all .2s;
    border-left:3px solid transparent;font-size:13px;font-weight:500;
  }
  .sidebar-item:hover{background:rgba(15,98,254,.15);color:#fff;border-left-color:var(--ibm-blue);}
  .sidebar-item.active{background:rgba(15,98,254,.25);color:#fff;border-left-color:var(--ibm-blue);}
  .sidebar-item i{font-size:16px;width:20px;text-align:center;}
  .agent-badge{
    margin-left:auto;font-size:9px;background:var(--ibm-blue);
    color:#fff;padding:1px 7px;border-radius:10px;font-weight:700;
  }

  /* ── Main layout ── */
  .app-body{display:flex;min-height:calc(100vh - 64px);}
  .main-content{flex:1;padding:28px;overflow-x:hidden;}

  /* ── Cards ── */
  .sp-card{background:var(--card-bg);border:1px solid var(--border);border-radius:8px;padding:24px;margin-bottom:22px;}
  .sp-card-header{display:flex;align-items:center;gap:12px;margin-bottom:20px;padding-bottom:14px;border-bottom:1px solid var(--border);}
  .agent-icon{
    width:42px;height:42px;border-radius:8px;display:flex;align-items:center;
    justify-content:center;font-size:20px;flex-shrink:0;
  }
  .ai1{background:linear-gradient(135deg,#0f62fe22,#0f62fe44);color:var(--ibm-blue);}
  .ai2{background:linear-gradient(135deg,#8a3ffc22,#8a3ffc44);color:var(--ibm-purple);}
  .ai3{background:linear-gradient(135deg,#009d9a22,#009d9a44);color:var(--ibm-teal);}
  .ai4{background:linear-gradient(135deg,#ff832b22,#ff832b44);color:var(--ibm-orange);}
  .ai5{background:linear-gradient(135deg,#19803822,#19803844);color:var(--ibm-green);}
  .ai-orch{background:linear-gradient(135deg,#da1e2822,#da1e2844);color:var(--ibm-red);}

  .agent-title{font-weight:700;font-size:16px;}
  .agent-subtitle{font-size:12px;color:var(--muted);}

  /* ── Hero ── */
  .hero-section{
    background:linear-gradient(135deg,#001141 0%,#0f3460 50%,#1a0533 100%);
    border-radius:12px;padding:40px;margin-bottom:28px;color:#fff;position:relative;overflow:hidden;
  }
  .hero-section::before{
    content:'';position:absolute;top:-40px;right:-40px;width:280px;height:280px;
    background:radial-gradient(circle,rgba(15,98,254,.25) 0%,transparent 70%);
    border-radius:50%;
  }
  .hero-section::after{
    content:'';position:absolute;bottom:-60px;left:30%;width:200px;height:200px;
    background:radial-gradient(circle,rgba(138,63,252,.15) 0%,transparent 70%);
    border-radius:50%;
  }
  .hero-badge{
    display:inline-flex;align-items:center;gap:6px;background:rgba(15,98,254,.3);
    border:1px solid rgba(15,98,254,.5);color:#a8c4ff;padding:4px 14px;
    border-radius:20px;font-size:11px;font-weight:600;margin-bottom:14px;letter-spacing:.5px;
  }
  .hero-title{font-size:32px;font-weight:800;line-height:1.2;margin-bottom:10px;letter-spacing:-.5px;}
  .hero-title span{background:linear-gradient(135deg,#74b3ff,#b87dff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
  .hero-desc{color:rgba(255,255,255,.75);font-size:14px;max-width:580px;margin-bottom:22px;}
  .hero-stats{display:flex;gap:28px;flex-wrap:wrap;}
  .hero-stat{text-align:center;}
  .hero-stat-val{font-size:22px;font-weight:800;color:#74b3ff;}
  .hero-stat-lbl{font-size:11px;color:rgba(255,255,255,.55);text-transform:uppercase;letter-spacing:.5px;}

  /* ── Metric tiles ── */
  .metric-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:14px;margin-bottom:22px;}
  .metric-tile{
    background:var(--card-bg);border:1px solid var(--border);border-radius:8px;
    padding:16px;text-align:center;transition:box-shadow .2s;
  }
  .metric-tile:hover{box-shadow:0 4px 16px rgba(15,98,254,.1);}
  .metric-val{font-size:22px;font-weight:800;margin-bottom:2px;}
  .metric-lbl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px;}
  .metric-trend{font-size:11px;font-weight:600;margin-top:4px;}
  .up{color:var(--ibm-green);} .down{color:var(--ibm-red);}

  /* ── Forms ── */
  .form-label{font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.4px;color:var(--muted);margin-bottom:5px;}
  .form-control,.form-select{border:1px solid var(--border);border-radius:6px;font-size:13px;padding:9px 12px;}
  .form-control:focus,.form-select:focus{border-color:var(--ibm-blue);box-shadow:0 0 0 3px rgba(15,98,254,.1);}

  /* ── Buttons ── */
  .btn-ibm{background:var(--ibm-blue);color:#fff;border:none;padding:10px 22px;border-radius:6px;font-weight:600;font-size:13px;cursor:pointer;transition:all .2s;display:inline-flex;align-items:center;gap:7px;}
  .btn-ibm:hover{background:#0353e9;color:#fff;transform:translateY(-1px);}
  .btn-ibm:disabled{background:#8d8d8d;cursor:not-allowed;transform:none;}
  .btn-ibm-outline{background:transparent;color:var(--ibm-blue);border:1px solid var(--ibm-blue);padding:9px 20px;border-radius:6px;font-weight:600;font-size:13px;cursor:pointer;transition:all .2s;display:inline-flex;align-items:center;gap:7px;}
  .btn-ibm-outline:hover{background:rgba(15,98,254,.08);}
  .btn-purple{background:var(--ibm-purple);} .btn-purple:hover{background:#6f32c8;}
  .btn-teal{background:var(--ibm-teal);} .btn-teal:hover{background:#007d7a;}
  .btn-orange{background:var(--ibm-orange);} .btn-orange:hover{background:#e5711e;}
  .btn-green{background:var(--ibm-green);} .btn-green:hover{background:#115229;}
  .btn-red{background:var(--ibm-red);} .btn-red:hover{background:#b01922;}

  /* ── Loader ── */
  .ai-loader{display:none;align-items:center;gap:10px;padding:18px;background:rgba(15,98,254,.04);border:1px solid rgba(15,98,254,.15);border-radius:8px;margin-top:14px;}
  .ai-loader.active{display:flex;}
  .pulse-dots{display:flex;gap:5px;}
  .pulse-dots span{width:8px;height:8px;border-radius:50%;background:var(--ibm-blue);animation:pulse 1.2s infinite;}
  .pulse-dots span:nth-child(2){animation-delay:.2s;}
  .pulse-dots span:nth-child(3){animation-delay:.4s;}
  @keyframes pulse{0%,80%,100%{transform:scale(.6);opacity:.4;}40%{transform:scale(1);opacity:1;}}
  .loader-text{font-size:13px;color:var(--ibm-blue);font-weight:600;}

  /* ── Output ── */
  .ai-output{display:none;margin-top:16px;}
  .ai-output.visible{display:block;}
  .output-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
  .output-badge{font-size:11px;background:rgba(25,128,56,.15);color:var(--ibm-green);padding:3px 10px;border-radius:12px;font-weight:700;display:flex;align-items:center;gap:4px;}
  .output-body{
    background:#0a0a1e;color:#e8f0fe;padding:20px 22px;border-radius:8px;
    font-family:'IBM Plex Mono','Courier New',monospace;font-size:12.5px;line-height:1.8;
    white-space:pre-wrap;word-break:break-word;max-height:520px;overflow-y:auto;
    border:1px solid #1a1a4e;
  }
  .output-body::-webkit-scrollbar{width:5px;}
  .output-body::-webkit-scrollbar-track{background:#0a0a1e;}
  .output-body::-webkit-scrollbar-thumb{background:#0f62fe;border-radius:3px;}

  /* ── Raw data tiles ── */
  .data-chip{display:inline-flex;align-items:center;gap:5px;background:rgba(15,98,254,.08);border:1px solid rgba(15,98,254,.2);color:var(--ibm-blue);padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600;margin:2px;}

  /* ── Pipeline ── */
  .pipeline{display:flex;align-items:center;gap:0;flex-wrap:wrap;margin-bottom:24px;}
  .pipeline-step{
    display:flex;align-items:center;gap:8px;background:var(--card-bg);
    border:1px solid var(--border);border-radius:6px;padding:10px 16px;
    font-size:12px;font-weight:600;cursor:pointer;transition:all .2s;
  }
  .pipeline-step:hover,.pipeline-step.done{background:#0f62fe;color:#fff;border-color:#0f62fe;}
  .pipeline-step.active{background:#0353e9;color:#fff;border-color:#0353e9;box-shadow:0 0 0 3px rgba(15,98,254,.25);}
  .pipeline-arrow{color:#8d8d8d;font-size:18px;margin:0 4px;}

  /* ── Tabs ── */
  .sp-tabs{display:flex;gap:0;border-bottom:2px solid var(--border);margin-bottom:22px;overflow-x:auto;}
  .sp-tab{padding:11px 20px;font-size:13px;font-weight:600;cursor:pointer;color:var(--muted);border-bottom:3px solid transparent;white-space:nowrap;transition:all .2s;margin-bottom:-2px;}
  .sp-tab:hover{color:var(--ibm-blue);}
  .sp-tab.active{color:var(--ibm-blue);border-bottom-color:var(--ibm-blue);}
  .tab-content-panel{display:none;} .tab-content-panel.active{display:block;}

  /* ── Platform badges ── */
  .plat-x{background:#000;color:#fff;} .plat-li{background:#0a66c2;color:#fff;}
  .plat-ig{background:linear-gradient(135deg,#e1306c,#833ab4);color:#fff;} .plat-fb{background:#1877f2;color:#fff;}
  .plat-badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:700;margin:2px;}

  /* ── Sections ── */
  .section-page{display:none;} .section-page.active{display:block;}

  /* ── Orchestrator visual ── */
  .orch-flow{
    display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:22px;
  }
  .orch-node{
    background:var(--card-bg);border:1px solid var(--border);border-radius:8px;
    padding:14px 10px;text-align:center;transition:all .3s;position:relative;
  }
  .orch-node.fired{border-color:var(--ibm-green);background:rgba(25,128,56,.06);}
  .orch-node-icon{font-size:22px;margin-bottom:6px;}
  .orch-node-name{font-size:11px;font-weight:700;}
  .orch-node-status{font-size:10px;color:var(--muted);margin-top:3px;}
  .orch-center{
    display:flex;align-items:center;justify-content:center;
    margin-bottom:16px;
  }
  .orch-hub{
    width:80px;height:80px;border-radius:50%;
    background:linear-gradient(135deg,var(--ibm-blue),var(--ibm-purple));
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    color:#fff;font-size:10px;font-weight:700;text-align:center;
    box-shadow:0 0 0 8px rgba(15,98,254,.12),0 0 0 16px rgba(15,98,254,.06);
  }

  /* ── Footer ── */
  footer{text-align:center;padding:20px;color:var(--muted);font-size:11px;border-top:1px solid var(--border);background:var(--card-bg);}

  /* ── Responsive ── */
  @media(max-width:768px){
    .sidebar{display:none;} .hero-title{font-size:22px;}
    .orch-flow{grid-template-columns:repeat(2,1fr);}
    .hero-stats{gap:14px;}
  }
</style>
</head>
<body>

<!-- ════════════════════════════════════════════
     NAVBAR
════════════════════════════════════════════ -->
<nav class="navbar navbar-custom navbar-expand-lg sticky-top px-3 py-2">
  <div class="navbar-brand-custom">
    <div class="logo-pill"><i class="bi bi-broadcast-pin"></i> IBM</div>
    <div>
      <div class="brand-name">SocialPulse AI</div>
      <div class="brand-sub">Powered by IBM watsonx.ai Granite</div>
    </div>
  </div>
  <div class="ms-auto d-flex align-items-center gap-3">
    <span class="badge" style="background:rgba(25,128,56,.25);color:#42be65;font-size:10px;padding:5px 10px;border-radius:12px;">
      <i class="bi bi-circle-fill" style="font-size:7px;"></i> Granite Models Active
    </span>
    <span class="badge" style="background:rgba(15,98,254,.25);color:#74b3ff;font-size:10px;padding:5px 10px;border-radius:12px;">
      <i class="bi bi-diagram-3"></i> 5 Agents Online
    </span>
  </div>
</nav>

<!-- ════════════════════════════════════════════
     LAYOUT
════════════════════════════════════════════ -->
<div class="app-body">

  <!-- ── Sidebar ── -->
  <aside class="sidebar">
    <div class="sidebar-section" style="margin-top:8px;">Navigation</div>
    <div class="sidebar-item active" onclick="showSection('dashboard')"><i class="bi bi-grid-1x2-fill"></i> Dashboard <span class="agent-badge">HOME</span></div>
    <div style="margin:12px 0 4px;"><div class="sidebar-section">AI Agents</div></div>
    <div class="sidebar-item" onclick="showSection('insights')"><i class="bi bi-graph-up-arrow" style="color:#74b3ff"></i> Insights Agent <span class="agent-badge">A1</span></div>
    <div class="sidebar-item" onclick="showSection('content')"><i class="bi bi-pencil-square" style="color:#b87dff"></i> Content Agent <span class="agent-badge">A2</span></div>
    <div class="sidebar-item" onclick="showSection('competitor')"><i class="bi bi-trophy-fill" style="color:#3ddbd9"></i> Competitor Agent <span class="agent-badge">A3</span></div>
    <div class="sidebar-item" onclick="showSection('trends')"><i class="bi bi-lightning-charge-fill" style="color:#ff832b"></i> Trend Agent <span class="agent-badge">A4</span></div>
    <div class="sidebar-item" onclick="showSection('campaign')"><i class="bi bi-rocket-takeoff-fill" style="color:#42be65"></i> Campaign Agent <span class="agent-badge">A5</span></div>
    <div style="margin:12px 0 4px;"><div class="sidebar-section">Orchestration</div></div>
    <div class="sidebar-item" onclick="showSection('orchestrate')"><i class="bi bi-cpu-fill" style="color:#ff7eb6"></i> Orchestrator <span class="agent-badge">IBM</span></div>
  </aside>

  <!-- ── Main ── -->
  <main class="main-content">

    <!-- ══════════ DASHBOARD ══════════ -->
    <div id="sec-dashboard" class="section-page active">
      <div class="hero-section">
        <div class="hero-badge"><i class="bi bi-stars"></i> IBM watsonx.ai · Granite Models · Langflow Orchestration</div>
        <h1 class="hero-title">SocialPulse AI<br/><span>Intelligent Social Media Agent</span></h1>
        <p class="hero-desc">A multi-agent AI platform powered by IBM Granite Models that analyzes social media, generates high-performing content, monitors competitors, predicts emerging trends, and optimizes campaigns — all in real time.</p>
        <div class="hero-stats">
          <div class="hero-stat"><div class="hero-stat-val">5</div><div class="hero-stat-lbl">AI Agents</div></div>
          <div class="hero-stat"><div class="hero-stat-val">4</div><div class="hero-stat-lbl">Platforms</div></div>
          <div class="hero-stat"><div class="hero-stat-val">∞</div><div class="hero-stat-lbl">Insights</div></div>
          <div class="hero-stat"><div class="hero-stat-val">IBM</div><div class="hero-stat-lbl">Granite AI</div></div>
        </div>
      </div>

      <!-- Metric tiles -->
      <div class="metric-grid">
        <div class="metric-tile"><div class="metric-val" style="color:var(--ibm-blue)">2.4M</div><div class="metric-lbl">Total Reach</div><div class="metric-trend up"><i class="bi bi-arrow-up-short"></i>+18.4%</div></div>
        <div class="metric-tile"><div class="metric-val" style="color:var(--ibm-purple)">6.7%</div><div class="metric-lbl">Avg Engagement</div><div class="metric-trend up"><i class="bi bi-arrow-up-short"></i>+2.1%</div></div>
        <div class="metric-tile"><div class="metric-val" style="color:var(--ibm-teal)">847K</div><div class="metric-lbl">Mentions</div><div class="metric-trend up"><i class="bi bi-arrow-up-short"></i>+31%</div></div>
        <div class="metric-tile"><div class="metric-val" style="color:var(--ibm-orange)">3.2x</div><div class="metric-lbl">ROAS</div><div class="metric-trend down"><i class="bi bi-arrow-down-short"></i>-0.4x</div></div>
        <div class="metric-tile"><div class="metric-val" style="color:var(--ibm-green)">72%</div><div class="metric-lbl">Positive Sentiment</div><div class="metric-trend up"><i class="bi bi-arrow-up-short"></i>+5%</div></div>
        <div class="metric-tile"><div class="metric-val" style="color:var(--ibm-red)">18</div><div class="metric-lbl">Viral Posts</div><div class="metric-trend up"><i class="bi bi-arrow-up-short"></i>+6</div></div>
      </div>

      <!-- Agent cards overview -->
      <div class="row g-3">
        <div class="col-md-6 col-lg-4">
          <div class="sp-card h-100" style="border-top:3px solid var(--ibm-blue);cursor:pointer;" onclick="showSection('insights')">
            <div class="sp-card-header">
              <div class="agent-icon ai1"><i class="bi bi-graph-up-arrow"></i></div>
              <div><div class="agent-title">Insights Agent</div><div class="agent-subtitle">Social Media Intelligence</div></div>
            </div>
            <p style="font-size:13px;color:var(--muted);">Analyzes trending topics, viral hashtags, engagement metrics, and cross-platform audience behavior using IBM Granite AI.</p>
            <div style="margin-top:12px;display:flex;gap:6px;flex-wrap:wrap;">
              <span class="plat-badge plat-x"><i class="bi bi-twitter-x"></i> X</span>
              <span class="plat-badge plat-li"><i class="bi bi-linkedin"></i> LinkedIn</span>
              <span class="plat-badge plat-ig"><i class="bi bi-instagram"></i> Instagram</span>
              <span class="plat-badge plat-fb"><i class="bi bi-facebook"></i> Facebook</span>
            </div>
          </div>
        </div>
        <div class="col-md-6 col-lg-4">
          <div class="sp-card h-100" style="border-top:3px solid var(--ibm-purple);cursor:pointer;" onclick="showSection('content')">
            <div class="sp-card-header">
              <div class="agent-icon ai2"><i class="bi bi-pencil-square"></i></div>
              <div><div class="agent-title">Content Agent</div><div class="agent-subtitle">AI Content Generation</div></div>
            </div>
            <p style="font-size:13px;color:var(--muted);">Generates platform-optimized posts, captions, scripts, and content calendars for any brand or campaign objective.</p>
            <div style="margin-top:12px;font-size:12px;color:var(--muted);">Outputs: Posts · Captions · Scripts · Email Lines · Calendars</div>
          </div>
        </div>
        <div class="col-md-6 col-lg-4">
          <div class="sp-card h-100" style="border-top:3px solid var(--ibm-teal);cursor:pointer;" onclick="showSection('competitor')">
            <div class="sp-card-header">
              <div class="agent-icon ai3"><i class="bi bi-trophy-fill"></i></div>
              <div><div class="agent-title">Competitor Agent</div><div class="agent-subtitle">Competitive Intelligence</div></div>
            </div>
            <p style="font-size:13px;color:var(--muted);">Benchmarks competitors, identifies content gaps, and surfaces strategic opportunities to outperform rivals.</p>
            <div style="margin-top:12px;font-size:12px;color:var(--muted);">Covers: Engagement · Content · Sentiment · Whitespace</div>
          </div>
        </div>
        <div class="col-md-6 col-lg-4">
          <div class="sp-card h-100" style="border-top:3px solid var(--ibm-orange);cursor:pointer;" onclick="showSection('trends')">
            <div class="sp-card-header">
              <div class="agent-icon ai4"><i class="bi bi-lightning-charge-fill"></i></div>
              <div><div class="agent-title">Trend Agent</div><div class="agent-subtitle">Predictive Trend Forecasting</div></div>
            </div>
            <p style="font-size:13px;color:var(--muted);">Predicts emerging trends, content formats, and audience behavioral shifts before they peak — giving brands first-mover advantage.</p>
            <div style="margin-top:12px;font-size:12px;color:var(--muted);">Horizon: 7 · 30 · 90 days · Global & Regional</div>
          </div>
        </div>
        <div class="col-md-6 col-lg-4">
          <div class="sp-card h-100" style="border-top:3px solid var(--ibm-green);cursor:pointer;" onclick="showSection('campaign')">
            <div class="sp-card-header">
              <div class="agent-icon ai5"><i class="bi bi-rocket-takeoff-fill"></i></div>
              <div><div class="agent-title">Campaign Agent</div><div class="agent-subtitle">Campaign Optimization</div></div>
            </div>
            <p style="font-size:13px;color:var(--muted);">Diagnoses campaign performance, reallocates budget, refines targeting, and prescribes A/B tests to maximize ROI.</p>
            <div style="margin-top:12px;font-size:12px;color:var(--muted);">KPIs: CTR · CPC · ROAS · CVR · CPA</div>
          </div>
        </div>
        <div class="col-md-6 col-lg-4">
          <div class="sp-card h-100" style="border-top:3px solid var(--ibm-red);cursor:pointer;" onclick="showSection('orchestrate')">
            <div class="sp-card-header">
              <div class="agent-icon ai-orch"><i class="bi bi-cpu-fill"></i></div>
              <div><div class="agent-title">IBM Orchestrator</div><div class="agent-subtitle">Langflow Multi-Agent Pipeline</div></div>
            </div>
            <p style="font-size:13px;color:var(--muted);">Coordinates all 5 agents through the IBM Langflow Orchestration pipeline to produce a unified strategic brief.</p>
            <div style="margin-top:12px;font-size:12px;color:var(--muted);">A1 → A2 → A3 → A4 → A5 → Synthesize</div>
          </div>
        </div>
      </div>
    </div>

    <!-- ══════════ AGENT 1: INSIGHTS ══════════ -->
    <div id="sec-insights" class="section-page">
      <div class="sp-card">
        <div class="sp-card-header">
          <div class="agent-icon ai1"><i class="bi bi-graph-up-arrow"></i></div>
          <div>
            <div class="agent-title">Agent 1 — Social Media Insights Agent</div>
            <div class="agent-subtitle">IBM Granite AI · Cross-Platform Social Intelligence · Real-Time Analysis</div>
          </div>
        </div>
        <div class="row g-3">
          <div class="col-md-6">
            <label class="form-label">Topic / Keyword</label>
            <input type="text" id="in-topic" class="form-control" placeholder="e.g. Artificial Intelligence, Climate Tech, NFTs…" value="Artificial Intelligence"/>
          </div>
          <div class="col-md-6">
            <label class="form-label">Platform Focus</label>
            <select id="in-platform" class="form-select">
              <option value="all">All Platforms (Cross-Platform)</option>
              <option value="X (Twitter)">X (Twitter)</option>
              <option value="LinkedIn">LinkedIn</option>
              <option value="Instagram">Instagram</option>
              <option value="Facebook">Facebook</option>
            </select>
          </div>
        </div>
        <div style="margin-top:16px;">
          <button class="btn-ibm" onclick="runInsights()"><i class="bi bi-cpu"></i> Analyze with IBM Granite</button>
        </div>
        <div class="ai-loader" id="load-insights"><div class="pulse-dots"><span></span><span></span><span></span></div><div class="loader-text">IBM Granite AI is analyzing social data across platforms…</div></div>
        <div class="ai-output" id="out-insights">
          <div id="insights-chips" style="margin-bottom:12px;"></div>
          <div class="output-header"><span style="font-weight:700;font-size:13px;"><i class="bi bi-graph-up-arrow" style="color:var(--ibm-blue)"></i> Insights Report</span><span class="output-badge"><i class="bi bi-check-circle-fill"></i> Granite Response</span></div>
          <div class="output-body" id="insights-text"></div>
        </div>
      </div>
    </div>

    <!-- ══════════ AGENT 2: CONTENT ══════════ -->
    <div id="sec-content" class="section-page">
      <div class="sp-card">
        <div class="sp-card-header">
          <div class="agent-icon ai2"><i class="bi bi-pencil-square"></i></div>
          <div>
            <div class="agent-title">Agent 2 — Content Generation Agent</div>
            <div class="agent-subtitle">IBM Granite AI · Platform-Optimized Content · Campaign Copy</div>
          </div>
        </div>
        <div class="row g-3">
          <div class="col-md-4">
            <label class="form-label">Brand Name</label>
            <input type="text" id="ct-brand" class="form-control" placeholder="Your Brand" value="TechVision"/>
          </div>
          <div class="col-md-4">
            <label class="form-label">Topic / Campaign</label>
            <input type="text" id="ct-topic" class="form-control" placeholder="e.g. Product Launch" value="AI Product Launch 2025"/>
          </div>
          <div class="col-md-4">
            <label class="form-label">Tone</label>
            <select id="ct-tone" class="form-select">
              <option value="professional">Professional</option>
              <option value="casual and fun">Casual & Fun</option>
              <option value="inspirational">Inspirational</option>
              <option value="bold and provocative">Bold & Provocative</option>
              <option value="educational">Educational</option>
            </select>
          </div>
          <div class="col-md-6">
            <label class="form-label">Campaign Goal</label>
            <select id="ct-goal" class="form-select">
              <option value="increase engagement">Increase Engagement</option>
              <option value="drive website traffic">Drive Website Traffic</option>
              <option value="generate leads">Generate Leads</option>
              <option value="boost brand awareness">Boost Brand Awareness</option>
              <option value="drive conversions">Drive Conversions</option>
            </select>
          </div>
          <div class="col-md-6">
            <label class="form-label">Target Platforms</label>
            <select id="ct-platforms" class="form-select" multiple>
              <option selected>X (Twitter)</option>
              <option selected>LinkedIn</option>
              <option selected>Instagram</option>
              <option selected>Facebook</option>
            </select>
          </div>
        </div>
        <div style="margin-top:16px;">
          <button class="btn-ibm btn-purple" onclick="runContent()"><i class="bi bi-magic"></i> Generate Content with IBM Granite</button>
        </div>
        <div class="ai-loader" id="load-content"><div class="pulse-dots"><span></span><span></span><span></span></div><div class="loader-text">IBM Granite is crafting your multi-platform content package…</div></div>
        <div class="ai-output" id="out-content">
          <div class="output-header"><span style="font-weight:700;font-size:13px;"><i class="bi bi-pencil-square" style="color:var(--ibm-purple)"></i> Generated Content Package</span><span class="output-badge"><i class="bi bi-check-circle-fill"></i> Granite Response</span></div>
          <div class="output-body" id="content-text"></div>
        </div>
      </div>
    </div>

    <!-- ══════════ AGENT 3: COMPETITOR ══════════ -->
    <div id="sec-competitor" class="section-page">
      <div class="sp-card">
        <div class="sp-card-header">
          <div class="agent-icon ai3"><i class="bi bi-trophy-fill"></i></div>
          <div>
            <div class="agent-title">Agent 3 — Competitor Monitoring Agent</div>
            <div class="agent-subtitle">IBM Granite AI · Competitive Intelligence · Market Positioning</div>
          </div>
        </div>
        <div class="row g-3">
          <div class="col-md-4">
            <label class="form-label">Your Brand</label>
            <input type="text" id="cp-brand" class="form-control" value="YourBrand"/>
          </div>
          <div class="col-md-4">
            <label class="form-label">Industry</label>
            <select id="cp-industry" class="form-select">
              <option value="Technology">Technology</option>
              <option value="E-Commerce">E-Commerce</option>
              <option value="Healthcare">Healthcare</option>
              <option value="Finance">Finance</option>
              <option value="Fashion & Lifestyle">Fashion & Lifestyle</option>
              <option value="Food & Beverage">Food & Beverage</option>
              <option value="Education">Education</option>
            </select>
          </div>
          <div class="col-md-4">
            <label class="form-label">Competitor 1</label>
            <input type="text" id="cp-c1" class="form-control" value="CompetitorAlpha"/>
          </div>
          <div class="col-md-4">
            <label class="form-label">Competitor 2</label>
            <input type="text" id="cp-c2" class="form-control" value="CompetitorBeta"/>
          </div>
          <div class="col-md-4">
            <label class="form-label">Competitor 3</label>
            <input type="text" id="cp-c3" class="form-control" value="CompetitorGamma"/>
          </div>
        </div>
        <div style="margin-top:16px;">
          <button class="btn-ibm btn-teal" onclick="runCompetitor()"><i class="bi bi-binoculars"></i> Analyze Competitors with IBM Granite</button>
        </div>
        <div class="ai-loader" id="load-competitor"><div class="pulse-dots"><span></span><span></span><span></span></div><div class="loader-text">IBM Granite is performing competitive intelligence analysis…</div></div>
        <div class="ai-output" id="out-competitor">
          <div id="comp-chips" style="margin-bottom:12px;"></div>
          <div class="output-header"><span style="font-weight:700;font-size:13px;"><i class="bi bi-trophy-fill" style="color:var(--ibm-teal)"></i> Competitor Intelligence Report</span><span class="output-badge"><i class="bi bi-check-circle-fill"></i> Granite Response</span></div>
          <div class="output-body" id="competitor-text"></div>
        </div>
      </div>
    </div>

    <!-- ══════════ AGENT 4: TRENDS ══════════ -->
    <div id="sec-trends" class="section-page">
      <div class="sp-card">
        <div class="sp-card-header">
          <div class="agent-icon ai4"><i class="bi bi-lightning-charge-fill"></i></div>
          <div>
            <div class="agent-title">Agent 4 — Trend Prediction Agent</div>
            <div class="agent-subtitle">IBM Granite AI · Predictive Analytics · Emerging Signal Detection</div>
          </div>
        </div>
        <div class="row g-3">
          <div class="col-md-4">
            <label class="form-label">Industry</label>
            <select id="tr-industry" class="form-select">
              <option value="Technology">Technology</option>
              <option value="Marketing">Marketing</option>
              <option value="Healthcare">Healthcare</option>
              <option value="Finance">Finance</option>
              <option value="Education">Education</option>
              <option value="Retail">Retail</option>
            </select>
          </div>
          <div class="col-md-4">
            <label class="form-label">Forecast Horizon</label>
            <select id="tr-horizon" class="form-select">
              <option value="7 days">7 Days (Short-term)</option>
              <option value="30 days" selected>30 Days (Monthly)</option>
              <option value="90 days">90 Days (Quarterly)</option>
            </select>
          </div>
          <div class="col-md-4">
            <label class="form-label">Region</label>
            <select id="tr-region" class="form-select">
              <option value="Global">Global</option>
              <option value="North America">North America</option>
              <option value="Europe">Europe</option>
              <option value="Asia Pacific">Asia Pacific</option>
              <option value="Middle East & Africa">Middle East & Africa</option>
            </select>
          </div>
        </div>
        <div style="margin-top:16px;">
          <button class="btn-ibm btn-orange" onclick="runTrends()"><i class="bi bi-lightning-charge"></i> Predict Trends with IBM Granite</button>
        </div>
        <div class="ai-loader" id="load-trends"><div class="pulse-dots"><span></span><span></span><span></span></div><div class="loader-text">IBM Granite is forecasting emerging social media trends…</div></div>
        <div class="ai-output" id="out-trends">
          <div id="trend-chips" style="margin-bottom:12px;"></div>
          <div class="output-header"><span style="font-weight:700;font-size:13px;"><i class="bi bi-lightning-charge-fill" style="color:var(--ibm-orange)"></i> Trend Forecast Report</span><span class="output-badge"><i class="bi bi-check-circle-fill"></i> Granite Response</span></div>
          <div class="output-body" id="trends-text"></div>
        </div>
      </div>
    </div>

    <!-- ══════════ AGENT 5: CAMPAIGN ══════════ -->
    <div id="sec-campaign" class="section-page">
      <div class="sp-card">
        <div class="sp-card-header">
          <div class="agent-icon ai5"><i class="bi bi-rocket-takeoff-fill"></i></div>
          <div>
            <div class="agent-title">Agent 5 — Campaign Optimization Agent</div>
            <div class="agent-subtitle">IBM Granite AI · ROI Maximization · Performance Diagnostics</div>
          </div>
        </div>
        <div class="row g-3">
          <div class="col-md-4">
            <label class="form-label">Campaign Name</label>
            <input type="text" id="ca-name" class="form-control" value="Summer AI Launch 2025"/>
          </div>
          <div class="col-md-4">
            <label class="form-label">Brand</label>
            <input type="text" id="ca-brand" class="form-control" value="TechVision"/>
          </div>
          <div class="col-md-4">
            <label class="form-label">Budget</label>
            <input type="text" id="ca-budget" class="form-control" value="$25,000"/>
          </div>
          <div class="col-md-4">
            <label class="form-label">Duration</label>
            <input type="text" id="ca-duration" class="form-control" value="30 days"/>
          </div>
          <div class="col-md-4">
            <label class="form-label">Primary Objective</label>
            <select id="ca-objective" class="form-select">
              <option value="brand awareness">Brand Awareness</option>
              <option value="lead generation">Lead Generation</option>
              <option value="conversions">Conversions</option>
              <option value="app installs">App Installs</option>
              <option value="video views">Video Views</option>
            </select>
          </div>
          <div class="col-md-4">
            <label class="form-label">Target Audience</label>
            <input type="text" id="ca-audience" class="form-control" value="Tech professionals aged 25–40"/>
          </div>
        </div>
        <div style="margin-top:16px;">
          <button class="btn-ibm btn-green" onclick="runCampaign()"><i class="bi bi-rocket-takeoff"></i> Optimize Campaign with IBM Granite</button>
        </div>
        <div class="ai-loader" id="load-campaign"><div class="pulse-dots"><span></span><span></span><span></span></div><div class="loader-text">IBM Granite is diagnosing and optimizing your campaign…</div></div>
        <div class="ai-output" id="out-campaign">
          <div id="camp-chips" style="margin-bottom:12px;"></div>
          <div class="output-header"><span style="font-weight:700;font-size:13px;"><i class="bi bi-rocket-takeoff-fill" style="color:var(--ibm-green)"></i> Campaign Optimization Report</span><span class="output-badge"><i class="bi bi-check-circle-fill"></i> Granite Response</span></div>
          <div class="output-body" id="campaign-text"></div>
        </div>
      </div>
    </div>

    <!-- ══════════ ORCHESTRATOR ══════════ -->
    <div id="sec-orchestrate" class="section-page">
      <div class="sp-card">
        <div class="sp-card-header">
          <div class="agent-icon ai-orch"><i class="bi bi-cpu-fill"></i></div>
          <div>
            <div class="agent-title">IBM Langflow Orchestrator — Multi-Agent Pipeline</div>
            <div class="agent-subtitle">Coordinates all 5 Agents · Synthesizes a Unified Strategic Brief · Powered by IBM Granite</div>
          </div>
        </div>

        <!-- Pipeline visual -->
        <div class="pipeline" id="orch-pipeline">
          <div class="pipeline-step" id="ps1"><i class="bi bi-graph-up-arrow" style="color:#74b3ff"></i> Insights A1</div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step" id="ps2"><i class="bi bi-pencil-square" style="color:#b87dff"></i> Content A2</div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step" id="ps3"><i class="bi bi-trophy-fill" style="color:#3ddbd9"></i> Competitor A3</div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step" id="ps4"><i class="bi bi-lightning-charge-fill" style="color:#ff832b"></i> Trends A4</div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step" id="ps5"><i class="bi bi-rocket-takeoff-fill" style="color:#42be65"></i> Campaign A5</div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step" id="ps6" style="background:var(--ibm-red);color:#fff;border-color:var(--ibm-red);"><i class="bi bi-cpu-fill"></i> Synthesize</div>
        </div>

        <div class="orch-flow" id="orch-nodes">
          <div class="orch-node" id="node1"><div class="orch-node-icon">📊</div><div class="orch-node-name">Insights Agent</div><div class="orch-node-status" id="ns1">Waiting…</div></div>
          <div class="orch-node" id="node2"><div class="orch-node-icon">✍️</div><div class="orch-node-name">Content Agent</div><div class="orch-node-status" id="ns2">Waiting…</div></div>
          <div class="orch-node" id="node3"><div class="orch-node-icon">🏆</div><div class="orch-node-name">Competitor Agent</div><div class="orch-node-status" id="ns3">Waiting…</div></div>
          <div class="orch-node" id="node4"><div class="orch-node-icon">⚡</div><div class="orch-node-name">Trend Agent</div><div class="orch-node-status" id="ns4">Waiting…</div></div>
          <div class="orch-node" id="node5"><div class="orch-node-icon">🚀</div><div class="orch-node-name">Campaign Agent</div><div class="orch-node-status" id="ns5">Waiting…</div></div>
        </div>

        <div class="row g-3">
          <div class="col-md-4">
            <label class="form-label">Brand</label>
            <input type="text" id="or-brand" class="form-control" value="TechVision"/>
          </div>
          <div class="col-md-4">
            <label class="form-label">Topic</label>
            <input type="text" id="or-topic" class="form-control" value="AI Innovation 2025"/>
          </div>
          <div class="col-md-4">
            <label class="form-label">Industry</label>
            <select id="or-industry" class="form-select">
              <option value="Technology">Technology</option>
              <option value="Healthcare">Healthcare</option>
              <option value="Finance">Finance</option>
              <option value="E-Commerce">E-Commerce</option>
            </select>
          </div>
        </div>
        <div style="margin-top:16px;">
          <button class="btn-ibm btn-red" onclick="runOrchestrate()"><i class="bi bi-cpu"></i> Launch IBM Langflow Orchestration Pipeline</button>
        </div>
        <div class="ai-loader" id="load-orch"><div class="pulse-dots"><span></span><span></span><span></span></div><div class="loader-text">IBM Langflow Orchestrator is coordinating all 5 agents and synthesizing results…</div></div>
        <div class="ai-output" id="out-orch">
          <div class="output-header"><span style="font-weight:700;font-size:13px;"><i class="bi bi-cpu-fill" style="color:var(--ibm-red)"></i> Unified Strategic Brief — IBM Granite Orchestration</span><span class="output-badge"><i class="bi bi-check-circle-fill"></i> All Agents Synthesized</span></div>
          <div class="output-body" id="orch-text"></div>
        </div>
      </div>
    </div>

  </main>
</div>

<footer>
  <strong>SocialPulse AI</strong> · Powered by IBM watsonx.ai Granite Models · IBM Langflow / IBM Orchestrate · Flask &amp; Bootstrap 5<br/>
  <span style="color:#8d8d8d;">© 2025 IBM · All AI-generated insights are for demonstration purposes.</span>
</footer>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
// ── Navigation ──────────────────────────────────────────────
function showSection(name){
  document.querySelectorAll('.section-page').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('.sidebar-item').forEach(s=>s.classList.remove('active'));
  document.getElementById('sec-'+name).classList.add('active');
  document.querySelectorAll('.sidebar-item').forEach(s=>{
    if(s.getAttribute('onclick')===`showSection('${name}')`){s.classList.add('active');}
  });
  window.scrollTo({top:0,behavior:'smooth'});
}

// ── Helpers ─────────────────────────────────────────────────
function showLoader(id){document.getElementById(id).classList.add('active');}
function hideLoader(id){document.getElementById(id).classList.remove('active');}
function showOutput(id){document.getElementById(id).classList.add('visible');}

function makeChips(obj){
  return Object.entries(obj).map(([k,v])=>{
    const val=Array.isArray(v)?v.join(', '):v;
    return `<span class="data-chip"><b>${k}:</b> ${val}</span>`;
  }).join('');
}

function animateText(el, text, speed=6){
  el.textContent=''; let i=0;
  const t=setInterval(()=>{
    el.textContent+=text[i++];
    el.scrollTop=el.scrollHeight;
    if(i>=text.length)clearInterval(t);
  },speed);
}

async function postJSON(url, body){
  const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

// ── Agent 1: Insights ────────────────────────────────────────
async function runInsights(){
  const topic=document.getElementById('in-topic').value;
  const platform=document.getElementById('in-platform').value;
  showLoader('load-insights');
  try{
    const d=await postJSON('/api/agent/insights',{topic,platform});
    hideLoader('load-insights');
    document.getElementById('insights-chips').innerHTML=makeChips({
      'Mentions':d.raw_data.total_mentions.toLocaleString(),
      'Engagement':d.raw_data.engagement_rate,
      'Sentiment':d.raw_data.sentiment,
      'Peak Time':d.raw_data.peak_time,
      'Viral Posts':d.raw_data.viral_posts,
    });
    showOutput('out-insights');
    animateText(document.getElementById('insights-text'), d.insights);
  }catch(e){hideLoader('load-insights');alert('Error: '+e.message);}
}

// ── Agent 2: Content ─────────────────────────────────────────
async function runContent(){
  const brand=document.getElementById('ct-brand').value;
  const topic=document.getElementById('ct-topic').value;
  const tone=document.getElementById('ct-tone').value;
  const goal=document.getElementById('ct-goal').value;
  const sel=document.getElementById('ct-platforms');
  const platforms=[...sel.selectedOptions].map(o=>o.value);
  showLoader('load-content');
  try{
    const d=await postJSON('/api/agent/content',{brand,topic,tone,goal,platforms});
    hideLoader('load-content');
    showOutput('out-content');
    animateText(document.getElementById('content-text'), d.content);
  }catch(e){hideLoader('load-content');alert('Error: '+e.message);}
}

// ── Agent 3: Competitor ──────────────────────────────────────
async function runCompetitor(){
  const your_brand=document.getElementById('cp-brand').value;
  const industry=document.getElementById('cp-industry').value;
  const competitors=[
    document.getElementById('cp-c1').value,
    document.getElementById('cp-c2').value,
    document.getElementById('cp-c3').value,
  ];
  showLoader('load-competitor');
  try{
    const d=await postJSON('/api/agent/competitor',{your_brand,industry,competitors});
    hideLoader('load-competitor');
    const chips={};
    for(const [name,m] of Object.entries(d.competitor_data)){
      chips[name]=`Eng:${m.avg_engagement} | ${m.top_content_type}`;
    }
    document.getElementById('comp-chips').innerHTML=makeChips(chips);
    showOutput('out-competitor');
    animateText(document.getElementById('competitor-text'), d.analysis);
  }catch(e){hideLoader('load-competitor');alert('Error: '+e.message);}
}

// ── Agent 4: Trends ──────────────────────────────────────────
async function runTrends(){
  const industry=document.getElementById('tr-industry').value;
  const horizon=document.getElementById('tr-horizon').value;
  const region=document.getElementById('tr-region').value;
  showLoader('load-trends');
  try{
    const d=await postJSON('/api/agent/trends',{industry,horizon,region});
    hideLoader('load-trends');
    const chips={};
    d.signals.slice(0,4).forEach(s=>{ chips[s.signal]=s.velocity+' ('+s.confidence+')'; });
    document.getElementById('trend-chips').innerHTML=makeChips(chips);
    showOutput('out-trends');
    animateText(document.getElementById('trends-text'), d.predictions);
  }catch(e){hideLoader('load-trends');alert('Error: '+e.message);}
}

// ── Agent 5: Campaign ────────────────────────────────────────
async function runCampaign(){
  const campaign_name=document.getElementById('ca-name').value;
  const brand=document.getElementById('ca-brand').value;
  const budget=document.getElementById('ca-budget').value;
  const duration=document.getElementById('ca-duration').value;
  const objective=document.getElementById('ca-objective').value;
  const target_audience=document.getElementById('ca-audience').value;
  showLoader('load-campaign');
  try{
    const d=await postJSON('/api/agent/campaign',{campaign_name,brand,budget,duration,objective,target_audience});
    hideLoader('load-campaign');
    document.getElementById('camp-chips').innerHTML=makeChips({
      'CTR':d.current_metrics.ctr,'CPC':d.current_metrics.cpc,
      'ROAS':d.current_metrics.roas,'CVR':d.current_metrics.conversion_rate,
      'Reach':d.current_metrics.reach.toLocaleString(),
    });
    showOutput('out-campaign');
    animateText(document.getElementById('campaign-text'), d.optimization);
  }catch(e){hideLoader('load-campaign');alert('Error: '+e.message);}
}

// ── Orchestrator ─────────────────────────────────────────────
function fireNode(i, status){
  const node=document.getElementById('node'+i);
  const step=document.getElementById('ps'+i);
  if(node){ node.classList.add('fired'); document.getElementById('ns'+i).textContent=status; }
  if(step){ step.classList.add('done'); }
}

async function runOrchestrate(){
  const brand=document.getElementById('or-brand').value;
  const topic=document.getElementById('or-topic').value;
  const industry=document.getElementById('or-industry').value;

  // Reset nodes
  for(let i=1;i<=6;i++){
    const n=document.getElementById('node'+i);
    const s=document.getElementById('ps'+i);
    if(n){n.classList.remove('fired');document.getElementById('ns'+i).textContent='Waiting…';}
    if(s){s.classList.remove('done','active');}
  }
  document.getElementById('out-orch').classList.remove('visible');
  showLoader('load-orch');

  // Animate pipeline steps
  const delays=[0,600,1200,1800,2400,3200];
  const labels=['Analyzing social data…','Generating content…','Benchmarking competitors…','Forecasting trends…','Optimizing campaign…','Synthesizing brief…'];
  delays.forEach((d,i)=>{
    setTimeout(()=>fireNode(i+1, labels[i]),d);
  });

  try{
    const data=await postJSON('/api/orchestrate',{brand,topic,industry});
    hideLoader('load-orch');
    document.getElementById('ps6').classList.add('done');
    showOutput('out-orch');
    animateText(document.getElementById('orch-text'), data.orchestration, 5);
  }catch(e){hideLoader('load-orch');alert('Error: '+e.message);}
}
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
