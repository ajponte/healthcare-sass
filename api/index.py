import os
import logging
from fastapi import FastAPI, Depends  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
from pydantic import BaseModel  # type: ignore
from fastapi_clerk_auth import ClerkConfig, ClerkHTTPBearer, HTTPAuthorizationCredentials  # type: ignore
from fastapi.encoders import jsonable_encoder  # type: ignore
import jwt
from openai import OpenAI  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("saas-api-vercel")

app = FastAPI()

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
            logger.error(f"Clerk JWT authentication failed: {e}", exc_info=True)
            if self.debug_mode:
                raise e
            return None

clerk_config = ClerkConfig(
    jwks_url=os.getenv("CLERK_JWKS_URL"),
    leeway=60  # Added 60s leeway to mitigate clock drift issues
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


@app.post("/api")
def consultation_summary(
        visit: Visit,
        creds: HTTPAuthorizationCredentials = Depends(clerk_guard),
):
    user_id = creds.decoded["sub"]  # Available for tracking/auditing
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

