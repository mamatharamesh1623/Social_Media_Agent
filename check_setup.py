"""
SocialPulse AI -- Setup Checker
Run: py -3.13 check_setup.py
"""
import sys, os, importlib

print("\n-- SocialPulse AI Setup Checker ----------------------")

# Python version
ver = sys.version_info
status = "OK" if ver >= (3, 9) else "FAIL - Need Python 3.9+"
print(f"\n[Python]  {sys.version.split()[0]}  {status}")

# Required packages
PACKAGES = {
    "flask":          "flask",
    "ibm_watsonx_ai": "ibm-watsonx-ai",
}
print("\n[Packages]")
all_ok = True
for mod, pip_name in PACKAGES.items():
    try:
        importlib.import_module(mod)
        print(f"  {pip_name:25s} OK  installed")
    except ImportError:
        print(f"  {pip_name:25s} MISSING  -->  pip install {pip_name}")
        all_ok = False

# Credentials check (env vars take priority; fallback reads .env file)
def load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v.strip()

load_env_file()

print("\n[Credentials]")
CRED_KEYS = [
    ("WATSONX_API_KEY",    "your-watsonx-api-key-here"),
    ("WATSONX_PROJECT_ID", "your-project-id-here"),
    ("WATSONX_URL",        ""),
]
for k, placeholder in CRED_KEYS:
    v = os.environ.get(k, "")
    if v and v not in (placeholder, "your-watsonx-api-key", "your-project-id", ""):
        masked = v[:6] + "..." + v[-4:] if len(v) > 10 else "****"
        print(f"  {k:25s} OK  ({masked})")
    elif k == "WATSONX_URL" and not v:
        print(f"  {k:25s} OK  (default: us-south.ml.cloud.ibm.com)")
    else:
        print(f"  {k:25s} NOT SET  <-- edit your .env file")
        all_ok = False

# Live connectivity test
print("\n[IBM Granite Connectivity]")
api_key    = os.environ.get("WATSONX_API_KEY", "")
project_id = os.environ.get("WATSONX_PROJECT_ID", "")
url        = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

if api_key in ("", "your-watsonx-api-key-here", "your-watsonx-api-key"):
    print("  Skipped -- credentials not configured yet.")
    print("  Once credentials are set, re-run this checker.")
else:
    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
        credentials = Credentials(api_key=api_key, url=url)
        model = ModelInference(
            model_id="ibm/granite-13b-instruct-v2",
            credentials=credentials,
            project_id=project_id,
            params={GenParams.MAX_NEW_TOKENS: 20, GenParams.TEMPERATURE: 0.5},
        )
        result = model.generate_text(prompt="Say hello in one word.")
        print(f'  IBM Granite responded: "{result.strip()}"  OK')
    except Exception as e:
        print(f"  Connection failed: {e}")
        all_ok = False

# Final verdict
print("\n------------------------------------------------------")
if all_ok:
    print("  All checks passed!")
    print("  Start the app with:  .\\run.ps1")
    print("  Then open:           http://localhost:5000")
else:
    print("  Fix the issues above, then re-run this checker.")
    print("  Credentials guide:   cloud.ibm.com --> watsonx.ai")
print("------------------------------------------------------\n")
