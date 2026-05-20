from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import os
from supabase import create_client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ReignAI WhatsApp Agent")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Anthropic client
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Supabase client
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# WhatsApp config
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "reign_ai_verify")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
ESCALATION_THRESHOLD = 0.7

class Message(BaseModel):
    customer_number: str
    message: str
    org_id: str = "default"

@app.get("/")
async def health():
    return {"status": "healthy", "service": "ReignAI Agent"}

@app.get("/webhook")
async def verify_webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    
    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        logger.info("Webhook verified")
        return int(challenge)
    else:
        raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook")
async def handle_message(request: Request):
    try:
        body = await request.json()
        
        entry = body.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])
        
        if not messages:
            return {"status": "no_messages"}
        
        msg = messages[0]
        customer_number = msg.get("from")
        message_text = msg.get("text", {}).get("body", "")
        
        if not message_text:
            return {"status": "no_text"}
        
        response = await process_with_claude(customer_number, message_text, "default")
        await send_whatsapp_message(customer_number, response["reply"])
        
        return {"status": "success", "escalated": response["escalated"]}
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error", "detail": str(e)}

async def process_with_claude(customer_number: str, message: str, org_id: str):
    """Process message with Claude and return response"""
    
    system_prompt = """You are a professional customer support agent for an e-commerce company.

Rules:
- Be helpful, concise, professional
- Answer common queries: order status, returns, shipping
- For complex issues (refunds, technical problems), say you'll escalate to human agent
- Keep responses under 160 characters when possible (SMS length)
- Never make promises about refunds/exchanges without human approval

If you need to escalate, end response with: [ESCALATE]"""

    try:
        response = claude_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": f"Customer message: {message}\n\nYour response:"
            }]
        )
        
        reply = response.content[0].text
        escalated = "[ESCALATE]" in reply
        reply = reply.replace("[ESCALATE]", "").strip()
        
        log_conversation(org_id, customer_number, message, reply, escalated)
        
        return {
            "reply": reply,
            "escalated": escalated,
            "confidence": 1.0 - (0.8 if escalated else 0.2)
        }
        
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        fallback_reply = "Sorry, I'm having trouble processing your request. A human agent will contact you shortly."
        log_conversation(org_id, customer_number, message, fallback_reply, True)
        return {"reply": fallback_reply, "escalated": True}

def log_conversation(org_id: str, customer_number: str, customer_message: str, agent_reply: str, escalated: bool):
    """Log conversation to Supabase"""
    try:
        supabase.table("conversations").insert({
            "org_id": org_id,
            "customer_number": customer_number,
            "customer_message": customer_message,
            "agent_reply": agent_reply,
            "escalated": escalated
        }).execute()
    except Exception as e:
        logger.error(f"Supabase log error: {e}")

async def send_whatsapp_message(to: str, message: str):
    """Send message via WhatsApp Business API"""
    import httpx
    
    url = f"https://graph.facebook.com/v17.0/{os.getenv('WHATSAPP_PHONE_NUMBER_ID')}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "text": {"body": message}
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            logger.error(f"WhatsApp send failed: {response.text}")
            raise HTTPException(status_code=500, detail="Failed to send message")

@app.get("/test")
async def test_claude_simple():
    """Simple test endpoint"""
    try:
        response = claude_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say 'Hello from Claude!' in one sentence."}]
        )
        return {
            "status": "success",
            "model": "claude-sonnet-4-6",
            "response": response.content[0].text
        }
    except Exception as e:
        return {
            "status": "error",
            "model": "claude-sonnet-4-6",
            "error": str(e)
        }

@app.post("/test")
async def test_message(msg: Message):
    """Direct test endpoint (no WhatsApp)"""
    response = await process_with_claude(msg.customer_number, msg.message, msg.org_id)
    return response

@app.get("/debug")
async def debug():
    key = os.getenv("ANTHROPIC_API_KEY", "")
    return {
        "key_exists": bool(key),
        "key_length": len(key),
        "first_4_chars": key[:4] if key else "NONE"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)