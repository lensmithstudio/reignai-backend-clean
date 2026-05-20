from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import anthropic
import os
from supabase import create_client
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ReignAI WhatsApp Agent")

# CORS - Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://reign-dashboard.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001",
        "*"  # Remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Anthropic client
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Supabase client - use SERVICE_KEY for backend operations
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_KEY", ""))
)

# WhatsApp config
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "reign_ai_verify")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
ESCALATION_THRESHOLD = 0.7

class Message(BaseModel):
    customer_number: str
    message: str
    org_id: Optional[str] = "default"
    tool_id: Optional[str] = "support_agent"

class AgentRunRequest(BaseModel):
    message: str
    org_id: str
    user_id: Optional[str] = None
    tool_id: str = "support_agent"

@app.get("/")
async def health():
    return {
        "status": "healthy",
        "service": "ReignAI Agent",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /",
            "webhook": "POST /webhook",
            "test": "GET|POST /test",
            "run_agent": "POST /api/run",
            "debug": "GET /debug"
        }
    }

@app.get("/webhook")
async def verify_webhook(request: Request):
    """WhatsApp webhook verification"""
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
    """WhatsApp webhook handler"""
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
        
        # Process with Claude
        response = await process_with_claude(
            customer_number=customer_number,
            message=message_text,
            org_id="default",
            tool_id="support_agent"
        )
        
        # Send WhatsApp reply
        await send_whatsapp_message(customer_number, response["reply"])
        
        return {
            "status": "success",
            "escalated": response["escalated"],
            "customer_number": customer_number
        }
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error", "detail": str(e)}

@app.post("/api/run")
async def run_agent(request: AgentRunRequest):
    """
    Run agent endpoint for dashboard
    This is what the frontend calls directly
    """
    start_time = time.time()
    
    try:
        response = await process_with_claude(
            customer_number=f"dashboard_{request.user_id or 'unknown'}",
            message=request.message,
            org_id=request.org_id,
            tool_id=request.tool_id
        )
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        return {
            "status": "success",
            "tool_id": request.tool_id,
            "reply": response["reply"],
            "escalated": response["escalated"],
            "confidence": response["confidence"],
            "duration_ms": duration_ms,
            "model": "claude-sonnet-4-6"
        }
        
    except Exception as e:
        logger.error(f"Agent run error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_with_claude(
    customer_number: str,
    message: str,
    org_id: str,
    tool_id: str = "support_agent"
):
    """Process message with Claude and return response"""
    
    # Tool-specific system prompts
    tool_prompts = {
        "support_agent": """You are a professional customer support agent for an e-commerce company.

Rules:
- Be helpful, concise, professional
- Answer common queries: order status, returns, shipping
- For complex issues (refunds, technical problems), say you'll escalate to human agent
- Keep responses under 160 characters when possible
- Never make promises about refunds/exchanges without human approval

If you need to escalate, end response with: [ESCALATE]""",
        
        "resume_screener": """You are an expert HR recruiter analyzing resumes.

Rules:
- Evaluate candidates objectively
- Look for: relevant experience, skills match, education
- Rate candidates 1-10 based on job description fit
- Be concise and specific
- Highlight key strengths and gaps

If resume doesn't match requirements, score low and explain why.""",
        
        "review_response": """You are a brand reputation manager responding to customer reviews.

Rules:
- Thank customers for positive reviews
- Address negative reviews with empathy
- Offer solutions for complaints
- Keep tone professional and helpful
- Don't make specific promises without approval

If serious issue, end with: [ESCALATE]""",
    }
    
    system_prompt = tool_prompts.get(tool_id, tool_prompts["support_agent"])
    
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
        
        # Log to Supabase
        log_conversation(org_id, customer_number, message, reply, escalated, tool_id)
        
        return {
            "reply": reply,
            "escalated": escalated,
            "confidence": 1.0 - (0.8 if escalated else 0.2)
        }
        
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        fallback_reply = "Sorry, I'm having trouble processing your request. A human agent will contact you shortly."
        log_conversation(org_id, customer_number, message, fallback_reply, True, tool_id)
        return {
            "reply": fallback_reply,
            "escalated": True,
            "confidence": 0.0
        }

def log_conversation(
    org_id: str,
    customer_number: str,
    customer_message: str,
    agent_reply: str,
    escalated: bool,
    tool_id: str = "support_agent"
):
    """Log conversation to Supabase"""
    try:
        supabase.table("conversations").insert({
            "org_id": org_id,
            "customer_number": customer_number,
            "customer_message": customer_message,
            "agent_reply": agent_reply,
            "escalated": escalated,
            "tool_id": tool_id
        }).execute()
        logger.info(f"Logged conversation for org {org_id}, tool {tool_id}")
    except Exception as e:
        logger.error(f"Supabase log error: {e}")

async def send_whatsapp_message(to: str, message: str):
    """Send message via WhatsApp Business API"""
    import httpx
    
    phone_number_id = os.getenv('WHATSAPP_PHONE_NUMBER_ID')
    
    if not phone_number_id or not WHATSAPP_ACCESS_TOKEN:
        logger.warning("WhatsApp not configured, skipping send")
        return
    
    url = f"https://graph.facebook.com/v17.0/{phone_number_id}/messages"
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
            messages=[{
                "role": "user",
                "content": "Say 'Hello from Claude!' in one sentence."
            }]
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
    response = await process_with_claude(
        customer_number=msg.customer_number,
        message=msg.message,
        org_id=msg.org_id or "default",
        tool_id=msg.tool_id or "support_agent"
    )
    return response

@app.get("/debug")
async def debug():
    """Debug endpoint - check environment"""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_KEY", ""))
    
    return {
        "anthropic_key_exists": bool(anthropic_key),
        "anthropic_key_length": len(anthropic_key),
        "anthropic_key_prefix": anthropic_key[:7] if anthropic_key else "NONE",
        "supabase_url_exists": bool(supabase_url),
        "supabase_url": supabase_url if supabase_url else "NOT_SET",
        "supabase_key_exists": bool(supabase_key),
        "whatsapp_configured": bool(WHATSAPP_ACCESS_TOKEN),
        "cors_origins": [
            "https://reign-dashboard.vercel.app",
            "http://localhost:3000",
            "http://localhost:3001"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)