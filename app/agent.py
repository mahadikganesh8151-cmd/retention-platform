import os
import json
import smtplib
from email.mime.text import MIMEText
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_APP_PASSWORD = os.getenv("SENDER_APP_PASSWORD")


def send_email(customer_id: str, reason: str) -> dict:
    subject = f"Retention Platform Alert - Customer {customer_id}"
    body = f"""A retention action was triggered by the agent.

Customer ID: {customer_id}
Reason: {reason}

This is an automated alert from the Retention Platform's agent layer."""

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = SENDER_EMAIL

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
            server.send_message(msg)
        print(f"[TOOL: send_email] Real email sent for customer={customer_id}")
        return {"tool": "send_email", "status": "sent", "customer_id": customer_id, "reason": reason}
    except Exception as e:
        print(f"[TOOL: send_email] Failed to send: {e}")
        return {"tool": "send_email", "status": "failed", "customer_id": customer_id, "reason": reason, "error": str(e)}


def offer_discount(customer_id: str, reason: str) -> dict:
    print(f"[TOOL: offer_discount] customer={customer_id} reason='{reason}'")
    return {"tool": "offer_discount", "status": "offered", "customer_id": customer_id, "reason": reason}


def escalate_human(customer_id: str, reason: str) -> dict:
    print(f"[TOOL: escalate_human] customer={customer_id} reason='{reason}'")
    return {"tool": "escalate_human", "status": "escalated", "customer_id": customer_id, "reason": reason}


def no_action(customer_id: str, reason: str) -> dict:
    print(f"[TOOL: no_action] customer={customer_id} reason='{reason}'")
    return {"tool": "no_action", "status": "logged", "customer_id": customer_id, "reason": reason}


AVAILABLE_TOOLS = {
    "send_email": send_email,
    "offer_discount": offer_discount,
    "escalate_human": escalate_human,
    "no_action": no_action,
}


def decide_action(customer_id: str, churn_risk: dict, customer_message: str) -> dict:
    prompt = f"""You are a retention agent for a telecom company. Based on the
situation below, decide which ONE tool to use. Respond with ONLY a JSON object,
no other text, in this exact format:
{{"tool": "<tool_name>", "reason": "<brief reason>"}}

Available tools:
- send_email: send a retention-focused follow-up email
- offer_discount: offer a temporary discount to retain the customer
- escalate_human: flag for a human agent to personally follow up
- no_action: no special action needed, this is a routine interaction

Situation:
- Customer churn risk: {churn_risk['risk_level']} ({churn_risk['churn_probability']*100:.1f}% probability)
- Customer's message: "{customer_message}"

Decide the single most appropriate action."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.strip("`").replace("json", "", 1).strip()

    decision = json.loads(text)
    tool_name = decision["tool"]
    reason = decision["reason"]

    if tool_name not in AVAILABLE_TOOLS:
        tool_name = "no_action"
        reason = "Invalid tool requested, defaulting to no_action"

    result = AVAILABLE_TOOLS[tool_name](customer_id, reason)

    return {
        "decision": tool_name,
        "reason": reason,
        "execution_result": result
    }
if __name__ == "__main__":
    test_churn_risk = {"risk_level": "high", "churn_probability": 0.65}
    test_message = "Hi, just confirming my new billing address was updated correctly."

    result = decide_action("test-customer-002", test_churn_risk, test_message)
    print(json.dumps(result, indent=2))