
import os, io, base64, time, json
from typing import Dict, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROXY_API_KEY  = os.getenv("PROXY_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

client = OpenAI(api_key=openai_api_key)
app = FastAPI(title="Desk Butler Cloud Proxy")

LATEST: Dict[str, Dict] = {}

ARM_LIMITS = {
    "base":    [-150, 150],
    "shoulder":[ -10, 120],
    "elbow":   [   0, 130],
    "wrist":   [ -90,  90],
}

SYSTEM_PROMPT = f"""
You control a 5‑DOF desk robot arm via simple commands. Joints: base, shoulder, elbow, wrist, gripper.
Joint limits (deg): {ARM_LIMITS}. Keep moves slow and safe.

From the latest desk photo, decide if tidying is needed.
Return ONLY a JSON object: {{\"cmds\":[...]}} where each item is one of:
  [\"HOME\"]
  [\"MOVE\", base, shoulder, elbow, wrist, gripper(0/1), duration_ms]
  [\"GRIP\", 0|1]
  [\"WAIT\", ms]
  [\"STOP\"]
  [\"STATUS\"]
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "cmds": {
            "type": "array",
            "items": {
                "type":"array",
                "items": {"anyOf":[
                    {"type":"string","enum":["HOME","STOP","STATUS","WAIT","GRIP","MOVE"]},
                    {"type":"number"},{"type":"integer"}
                ]}
            }
        }
    },
    "required": ["cmds"],
    "additionalProperties": False
}

class BrainRequest(BaseModel):
    status: Optional[str] = "idle"

def _require_key(x_api_key: Optional[str]):
    if PROXY_API_KEY and (x_api_key != PROXY_API_KEY):
        raise HTTPException(status_code=401, detail="Bad API key")

@app.get("/health")
def health():
    return {"ok": True, "stored": list(LATEST.keys())}

@app.post("/upload/{robot_id}")
async def upload(robot_id: str, image: UploadFile = File(...), x_api_key: Optional[str] = Header(default=None)):
    _require_key(x_api_key)
    if image.content_type not in ("image/jpeg","image/jpg","image/png"):
        raise HTTPException(415, "Send JPEG/PNG")
    raw = await image.read()
    if len(raw) > 1_500_000:
        raise HTTPException(413, "Image too large")
    b64 = base64.b64encode(raw).decode("ascii")
    LATEST[robot_id] = {"b64": b64, "ts": time.time(), "mime": image.content_type}
    return {"ok": True, "size": len(raw)}

@app.post("/brain/{robot_id}")
def brain(robot_id: str, req: BrainRequest, x_api_key: Optional[str] = Header(default=None)):
    _require_key(x_api_key)
    entry = LATEST.get(robot_id)
    if not entry or (time.time() - entry["ts"] > 60):
        return {"cmds": []}

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            instructions=SYSTEM_PROMPT,
            input=[
                {"role":"user","content":[
                    {"type":"input_text","text":"Analyze the desk photo and produce tidy commands if needed."},
                    {"type":"input_image","image_data": entry["b64"], "mime_type": entry["mime"]}
                ]}
            ],
            text={
                "format": {
                    "type":"json_schema",
                    "name":"RobotPlan",
                    "schema": JSON_SCHEMA,
                    "strict": True
                }
            },
            max_output_tokens=400
        )
        plan = json.loads(resp.output_text)
        cmds = plan.get("cmds", [])
    except Exception as e:
        return {"cmds": [], "error": str(e)}

    def clip(v, lo, hi): 
        return max(lo, min(hi, v))
    safe = []
    for c in cmds:
        if not c: continue
        op = c[0]
        if op == "MOVE" and len(c) == 7:
            _, a0,a1,a2,a3,g,T = c
            a0 = clip(float(a0), *ARM_LIMITS["base"])
            a1 = clip(float(a1), *ARM_LIMITS["shoulder"])
            a2 = clip(float(a2), *ARM_LIMITS["elbow"])
            a3 = clip(float(a3), *ARM_LIMITS["wrist"])
            g  = 1 if int(g) else 0
            T  = int(max(1000, min(20000, int(T))))
            safe.append(["MOVE", a0,a1,a2,a3,g,T])
        elif op in ("HOME","STOP","STATUS"):
            safe.append([op])
        elif op == "GRIP" and len(c) == 2:
            safe.append(["GRIP", 1 if int(c[1]) else 0])
        elif op == "WAIT" and len(c) == 2:
            safe.append(["WAIT", int(max(0,min(5000,int(c[1]))))])
    return {"cmds": safe}
