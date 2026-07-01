import os
import logging
from pathlib import Path
from fastapi import FastAPI, Depends, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi_clerk_auth import ClerkConfig, ClerkHTTPBearer, HTTPAuthorizationCredentials
from fastapi.encoders import jsonable_encoder
import jwt
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("saas-api")

app = FastAPI()

# Add CORS middleware (allows frontend to call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Clerk Bearer to log JWT decode errors
class LoggingClerkHTTPBearer(ClerkHTTPBearer):
    def _decode_token(self, token: str) -> dict | None:
        try:
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)
            decoded_token = jwt.decode(
                token,
                key=signing_key.key,
                audience=self.audience,
                issuer=self.issuer,
                algorithms=["RS256"],
                options={
                    "verify_exp": self.config.verify_exp,
                    "verify_aud": self.config.verify_aud,
                    "verify_iss": self.config.verify_iss,
                    "verify_iat": self.config.verify_iat,
                },
                leeway=self.config.leeway,
            )
            return dict(jsonable_encoder(decoded_token))
        except Exception as e:
            try:
                import time
                unverified = jwt.decode(token, options={"verify_signature": False, "verify_exp": False})
                logger.error(f"Failed JWT claims - exp: {unverified.get('exp')} ({time.ctime(unverified.get('exp')) if unverified.get('exp') else 'None'}), iat: {unverified.get('iat')} ({time.ctime(unverified.get('iat')) if unverified.get('iat') else 'None'})")
                logger.error(f"Current server time: {time.time()} ({time.ctime()})")
            except Exception as decode_err:
                logger.error(f"Could not parse unverified token claims: {decode_err}")
            logger.error(f"Clerk JWT authentication failed: {e}", exc_info=True)
            if self.debug_mode:
                raise e
            return None

# Clerk authentication setup
clerk_config = ClerkConfig(
    jwks_url=os.getenv("CLERK_JWKS_URL"),
    leeway=60  # Added 60s leeway to mitigate clock drift issues in Docker/Lambda
)
clerk_guard = LoggingClerkHTTPBearer(clerk_config)

class Visit(BaseModel):
    patient_name: str
    date_of_visit: str
    notes: str

system_prompt = """
You are provided with notes written by a doctor from a patient's visit.
Your job is to summarize the visit for the doctor and provide an email.
Reply with exactly three sections with the headings:
### Summary of visit for the doctor's records
### Next steps for the doctor
### Draft of email to patient in patient-friendly language
"""

def user_prompt_for(visit: Visit) -> str:
    return f"""Create the summary, next steps and draft email for:
Patient Name: {visit.patient_name}
Date of Visit: {visit.date_of_visit}
Notes:
{visit.notes}"""

@app.post("/api/consultation")
def consultation_summary(
        visit: Visit,
        creds: HTTPAuthorizationCredentials = Depends(clerk_guard),
):
    user_id = creds.decoded["sub"]
    client = OpenAI()

    user_prompt = user_prompt_for(visit)
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        logger.info(f"Requesting OpenAI chat completion for user: {user_id}")
        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=prompt,
            stream=True,
        )
    except Exception as e:
        logger.error(f"OpenAI completion creation failed: {e}", exc_info=True)
        raise e

    def event_stream():
        try:
            for chunk in stream:
                text = chunk.choices[0].delta.content
                if text:
                    lines = text.split("\n")
                    for line in lines[:-1]:
                        yield f"data: {line}\n\n"
                        yield "data:  \n"
                    yield f"data: {lines[-1]}\n\n"
        except Exception as e:
            logger.error(f"OpenAI stream iteration failed: {e}", exc_info=True)
            raise e

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/health")
def health_check():
    """Health check endpoint (used for local Docker; Lambda does not invoke it)"""
    return {"status": "healthy"}

# Serve static files (our Next.js export) - MUST BE LAST!
static_path = Path("static")
if static_path.exists():
    @app.get("/")
    async def serve_root():
        return FileResponse(static_path / "index.html")

    @app.get("/product")
    async def serve_product():
        return FileResponse(static_path / "product.html")

    app.mount("/", StaticFiles(directory="static", html=True), name="static")

