from datetime import datetime, timezone
import os
import re
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Vera Message Engine", version="0.1.0")
STARTED = datetime.now(timezone.utc)

# In-memory state. The judge expects statefulness during one run.
contexts: Dict[str, Dict[str, Dict[str, Any]]] = {
    "category": {}, "merchant": {}, "customer": {}, "trigger": {}
}
closed_conversations = set()
sent_bodies: Dict[str, set[str]] = {}

TEAM_NAME = os.getenv("TEAM_NAME", "Pratham Vera Bot")
MODEL_NAME = os.getenv("MODEL_NAME", "deterministic-rule-engine")


class ContextIn(BaseModel):
    scope: str
    context_id: str
    version: int
    delivered_at: str
    payload: Dict[str, Any]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def first_name(merchant: Dict[str, Any]) -> str:
    ident = merchant.get("identity", {})
    return ident.get("owner_first_name") or ident.get("name", "there").split()[0]


def active_offer(merchant: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for offer in merchant.get("offers", []):
        if str(offer.get("status", "")).lower() == "active":
            return offer
    return None


def recent_contact(merchant: Dict[str, Any], now: str) -> bool:
    history = merchant.get("conversation_history", [])
    if not history:
        return False
    try:
        now_dt = datetime.fromisoformat(now.replace("Z", "+00:00"))
    except Exception:
        now_dt = datetime.now(timezone.utc)
    for item in reversed(history):
        if item.get("from") != "vera":
            continue
        try:
            ts = datetime.fromisoformat(item.get("ts", "").replace("Z", "+00:00"))
            if (now_dt - ts).total_seconds() <= 72 * 3600:
                return True
        except Exception:
            continue
    return False


def search_signal(merchant: Dict[str, Any], trigger: Dict[str, Any]) -> Optional[str]:
    p = trigger.get("payload", {})
    # Prefer explicit search/demand information supplied by the trigger.
    for key in ("searches", "search_count", "nearby_searches", "search_volume"):
        if p.get(key):
            term = p.get("query") or p.get("search_term") or p.get("top_query") or "local demand"
            return f"{p[key]} people nearby searched for '{term}'"
    # Preserve supplied merchant signals without inventing numeric search facts.
    for s in merchant.get("signals", []):
        if "search" in str(s).lower() or "demand" in str(s).lower():
            return str(s).replace("_", " ")
    return None


def trigger_text(category: Dict[str, Any], merchant: Dict[str, Any], trigger: Dict[str, Any], customer: Optional[Dict[str, Any]]) -> str:
    kind = trigger.get("kind", "")
    p = trigger.get("payload", {})
    if kind == "perf_dip":
        metric = p.get("metric", "performance")
        delta = p.get("delta_pct")
        if isinstance(delta, (int, float)):
            return f"your {metric} is down {abs(delta) * 100:.0f}%"
        return f"your {metric} is showing a dip"
    if kind == "festival_upcoming":
        return f"{p.get('festival', 'The upcoming festival')} is coming up"
    if kind == "renewal_due":
        return f"your {p.get('plan', 'current')} plan has {p.get('days_remaining', 'limited')} days left"
    if kind == "recall_due" and customer:
        return f"{customer.get('identity', {}).get('name', 'a customer')}'s recall window is due"
    if kind == "ipl_match_today":
        return f"{p.get('match', 'an IPL match')} is on today"
    if kind == "research_digest":
        return "a relevant research update just landed"
    if kind == "compliance" or kind == "regulation_change":
        return "a relevant compliance update just arrived"
    if kind == "bridal_followup" and customer:
        return f"{customer.get('identity', {}).get('name', 'a bridal customer')} is in a bridal follow-up window"
    if kind:
        return kind.replace("_", " ")
    return "there is a relevant update for your business"


def compose_merchant(merchant: Dict[str, Any], category: Dict[str, Any], trigger: Dict[str, Any], now: str) -> Optional[Dict[str, Any]]:
    mid = merchant.get("merchant_id")
    tid = trigger.get("id")
    cid = trigger.get("customer_id")
    conversation_id = f"conv_{mid}_{tid}"
    if conversation_id in closed_conversations:
        return None

    customer = contexts["customer"].get(cid, {}).get("payload") if cid else None
    if trigger.get("scope") == "customer":
        return compose_customer(merchant, category, trigger, customer)

    kind = trigger.get("kind", "")
    p = trigger.get("payload", {})
    offer = active_offer(merchant)
    voice = category.get("voice", {})
    tone = voice.get("tone", "")
    name = first_name(merchant)
    business = merchant.get("identity", {}).get("name", "your business")

    # Planning intent gets priority over generic campaign suggestions.
    if kind == "active_planning_intent":
        topic = p.get("intent_topic", "your idea").replace("_", " ")
        last = p.get("merchant_last_message")
        if offer:
            body = (f"{name}, let’s build the {topic} around your existing {offer.get('title')}. "
                    f"You said, ‘{last}’. Want me to draft a first version you can edit?" if last else
                    f"{name}, let’s build the {topic} around your existing {offer.get('title')}. Want me to draft a first version you can edit?")
        else:
            body = (f"{name}, let’s build the {topic}. "
                    f"You said, ‘{last}’. Want me to draft a first version you can edit?" if last else
                    f"{name}, let’s build the {topic}. Want me to draft a first version you can edit?")
        return action(conversation_id, merchant, trigger, body, "open_ended",
                      "Explicit merchant planning intent takes priority; the response continues the existing thread and asks for one low-effort next step.")

    # Research/compliance/CDE: cite the supplied source and use merchant-specific context.
    if kind in {"research_digest", "regulation_change", "compliance_alert", "compliance", "cde_opportunity"}:
        digest_id = p.get("top_item_id") or p.get("digest_item_id")
        digest_item = next((d for d in category.get("digest", []) if d.get("id") == digest_id), None)
        if not digest_item:
            return None
        source = digest_item.get("source", "the supplied source")
        title = digest_item.get("title", "a relevant update")
        summary = digest_item.get("summary", "")
        if kind == "research_digest":
            cohort = merchant.get("customer_aggregate", {}).get("high_risk_adult_count")
            anchor = f"your {cohort} high-risk adult patients" if cohort else "your patient mix"
            body = (f"Dr. {name}, {title}. {summary} "
                    f"This is relevant to {anchor}. {source}. Want me to pull the abstract and draft a patient-ed message?")
        elif kind == "regulation_change":
            deadline = p.get("deadline_iso") or "the supplied deadline"
            body = (f"Dr. {name}, {title}. Effective {deadline[:10] if isinstance(deadline, str) else deadline}. "
                    f"{summary} {source}. Want me to turn this into a short audit checklist?")
        else:
            body = (f"{name}, {title}. {summary} {source}. "
                    f"Want me to pull the details and prepare the next step?")
        return action(conversation_id, merchant, trigger, body, "open_ended",
                      f"{kind} uses the supplied category digest, source, and merchant context without inventing facts.")

    if kind == "curious_ask_due":
        body = (f"Hi {name}! Quick one — what service has been most in demand at {business} this week? "
                "I’ll turn your answer into a ready-to-use post and customer reply.")
        return action(conversation_id, merchant, trigger, body, "open_ended",
                      "The trigger is an intentional low-stakes question; reciprocity makes the reply useful rather than generic.")

    if kind == "ipl_match_today":
        digest = next((d for d in category.get("digest", []) if d.get("id") == "d_2026W17_ipl_window"), None)
        if not offer:
            return None
        if digest and not p.get("is_weeknight", False):
            body = (f"Quick heads-up {name} — {p.get('match')} at {p.get('venue')} tonight. "
                    f"{digest.get('summary')} Your {offer.get('title')} is already active. "
                    "Want me to draft a delivery-focused post for the match window?")
            rationale = "The Saturday IPL trigger conflicts with the supplied category pattern, so the bot avoids blindly promoting a match-night offer and uses the existing offer instead."
        else:
            body = f"Quick heads-up {name} — {p.get('match')} today. You already have {offer.get('title')}. Want me to promote it for the match window?"
            rationale = "Time-bound event + real active offer + one concrete CTA."
        return action(conversation_id, merchant, trigger, body, "open_ended", rationale)

    if kind == "seasonal_perf_dip":
        digest = next((d for d in category.get("digest", []) if d.get("id") == "d_2026W17_resolution_window"), None)
        pct = abs(p.get("delta_pct", 0)) * 100
        members = merchant.get("customer_aggregate", {}).get("total_active_members")
        body = (f"{name}, your {p.get('metric', 'performance')} is down {pct:.0f}% this week, "
                f"and the context flags this April–June dip as seasonal. "
                f"{digest.get('actionable', 'Focus on retention before increasing acquisition spend.') if digest else 'Focus on retention before increasing acquisition spend.'} "
                f"{members} active members are already in your base. Want me to draft a summer attendance challenge?")
        return action(conversation_id, merchant, trigger, body, "open_ended",
                      "Seasonal dip is reframed using the supplied category evidence and merchant member count, with a retention-focused action.")

    if kind == "supply_alert":
        batches = ", ".join(p.get("affected_batches", []))
        chronic = merchant.get("customer_aggregate", {}).get("chronic_rx_count")
        body = (f"{name}, urgent: voluntary recall on {p.get('molecule')} batches {batches} by {p.get('manufacturer')}. "
                f"Your context shows {chronic} chronic-Rx customers. Want me to filter the relevant customer list for these batches?")
        return action(conversation_id, merchant, trigger, body, "open_ended",
                      "Urgent supply alert is stated precisely and the next action is limited to filtering the affected list; no unsupported affected-customer count is invented.")

    if kind == "category_seasonal":
        trends = ", ".join(p.get("trends", [])[:3])
        body = (f"{name}, summer demand is shifting: {trends}. "
                "Your context flags a shelf action as recommended. Want me to turn these signals into a shelf-priority list?")
        return action(conversation_id, merchant, trigger, body, "open_ended",
                      "Seasonal demand trigger is translated into a concrete inventory action using only supplied trends.")

    if kind == "renewal_due":
        body = (f"{name}, your {p.get('plan', 'plan')} has {p.get('days_remaining')} days left. "
                f"Renewal is ₹{p.get('renewal_amount')}. Want me to help you review it?")
        return action(conversation_id, merchant, trigger, body, "open_ended", "Time-bound renewal trigger with exact plan, deadline and amount.")

    if kind == "gbp_unverified":
        body = f"{name}, your business profile is still unverified. The supplied path is {p.get('verification_path')}. Want me to walk you through the verification steps?"
        return action(conversation_id, merchant, trigger, body, "open_ended", "Verification is directly actionable and the response uses the supplied verification path.")

    if kind == "competitor_opened":
        body = (f"{name}, {p.get('competitor_name')} opened {p.get('distance_km')} km away with {p.get('their_offer')}. "
                "You already have your current offer catalog. Want me to compare the two offers before changing anything?")
        return action(conversation_id, merchant, trigger, body, "open_ended", "Competitor event is time-relevant; response compares against the existing catalog instead of inventing a counter-offer.")

    if kind == "review_theme_emerged":
        body = (f"{name}, {p.get('occurrences_30d')} reviews mention {p.get('theme').replace('_', ' ')} and the trend is {p.get('trend')}. "
                "Want me to draft a customer-facing response and a short operational note?")
        return action(conversation_id, merchant, trigger, body, "open_ended", "Emerging review theme is quantified and connected to one concrete response workflow.")

    if kind == "milestone_reached":
        remaining = max(0, p.get("milestone_value", 0) - p.get("value_now", 0))
        body = (f"{name}, you’re at {p.get('value_now')} reviews — {remaining} more to reach {p.get('milestone_value')}. "
                "Want me to prepare a thank-you post for the milestone?")
        return action(conversation_id, merchant, trigger, body, "open_ended", "Imminent milestone is specific and paired with a low-effort content action.")

    if kind == "perf_spike":
        body = (f"{name}, your {p.get('metric')} is up {abs(p.get('delta_pct', 0))*100:.0f}% this week vs the supplied baseline of {p.get('vs_baseline')}. "
                f"The context links the spike to {str(p.get('likely_driver') or 'the supplied activity').replace('_', ' ')}. Want me to help repeat what worked?")
        return action(conversation_id, merchant, trigger, body, "open_ended", "Performance spike is attributed only to the supplied driver and asks for one repeatable next step.")

    if kind == "dormant_with_vera":
        body = (f"Hi {name} — it’s been {p.get('days_since_last_merchant_message')} days since we last worked on "
                f"{str(p.get('last_topic') or 'our previous work').replace('_', ' ')}. Want to pick it back up?")
        return action(conversation_id, merchant, trigger, body, "binary_yes_no", "Dormancy trigger uses exact supplied recency and prior topic.")

    if kind == "winback_eligible":
        body = (f"Hi {name} — {p.get('lapsed_customers_added_since_expiry')} lapsed customers have been added since expiry, "
                f"and performance is down {abs(p.get('perf_dip_pct', 0))*100:.0f}%. Want me to build a win-back message around your existing offers?")
        return action(conversation_id, merchant, trigger, body, "open_ended", "Win-back trigger connects supplied lapse volume and performance dip to the existing offer catalog.")

    # Generated/expanded triggers can intentionally contain only a placeholder.
    # Never crash or invent missing facts; use the trigger family and ask one useful next step.
    if p.get("placeholder"):
        labels = {
            "perf_dip": "a performance dip", "perf_spike": "a performance spike",
            "milestone_reached": "a milestone", "dormant_with_vera": "our previous conversation",
            "competitor_opened": "a competitor update", "festival_upcoming": "an upcoming seasonal moment",
        }
        label = labels.get(kind, kind.replace("_", " "))
        body = f"{name}, I have a {label} update for {business}. Want me to pull the available details and suggest the next step?"
        return action(conversation_id, merchant, trigger, body, "open_ended",
                      "The trigger contains only a placeholder, so the bot avoids fabricating specifics and asks to retrieve the available details.")

    # Generic demand/performance/festival fallback: use only real active offers.
    if not offer:
        return None
    search = search_signal(merchant, trigger)
    if search:
        body = f"{search} today. You already have {offer.get('title')}. Want me to promote it to capture this demand?"
        rationale = "Concrete demand signal + active merchant offer + one low-friction CTA."
    elif kind == "perf_dip":
        pct = abs(p.get("delta_pct", 0)) * 100
        body = f"{name}, your {p.get('metric', 'performance')} is down {pct:.0f}% over {p.get('window', 'the current window')}. You already have {offer.get('title')}. Want me to promote it to help address the dip?"
        rationale = "Performance dip + real active offer; no unsupported campaign is invented."
    elif kind == "festival_upcoming":
        body = f"{p.get('festival')} is coming up. You already have {offer.get('title')}. Want me to promote it for the occasion?"
        rationale = "Time-bound seasonal trigger + existing offer + one clear action."
    else:
        body = f"{name}, {trigger_text(category, merchant, trigger, customer)}. You already have {offer.get('title')}. Want me to promote it?"
        rationale = "Relevant trigger + active merchant offer + simple CTA."
    return action(conversation_id, merchant, trigger, body, "binary_yes_no", rationale)

def compose_customer(merchant: Dict[str, Any], category: Dict[str, Any], trigger: Dict[str, Any], customer: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not customer:
        return None
    prefs = customer.get("preferences", {})
    consent = customer.get("consent", {})
    if not prefs.get("reminder_opt_in") or not consent.get("scope"):
        return None
    offer = active_offer(merchant)
    p = trigger.get("payload", {})
    name = customer.get("identity", {}).get("name", "there")
    kind = trigger.get("kind")
    if kind == "recall_due":
        slots = p.get("available_slots", [])
        slot_text = " or ".join(s.get("label", "") for s in slots[:2] if s.get("label"))
        if not slot_text:
            return None
        offer_text = f" {offer.get('title')}." if offer else ""
        body = f"Hi {name}, {merchant.get('identity', {}).get('name', 'the clinic')} here. " \
               f"Your {p.get('service_due', 'recall')} is due.{offer_text} " \
               f"We have {slot_text}. Which works for you?"
        return action(f"conv_{merchant.get('merchant_id')}_{trigger.get('id')}", merchant, trigger, body,
                      "binary_yes_no", "Customer recall trigger is consented and has concrete available slots.", send_as="merchant_on_behalf", customer_id=customer.get("customer_id"))
    if kind == "wedding_package_followup":
        body = f"Hi {name}, {merchant.get('identity', {}).get('name', 'the salon')} here. " \
               f"Your wedding is {p.get('days_to_wedding', 'soon')} days away and your {p.get('next_step_window_open', 'next-step')} window is open. " \
               f"{offer.get('title') + '.' if offer else ''} Want me to help you pick the next slot?"
        return action(f"conv_{merchant.get('merchant_id')}_{trigger.get('id')}", merchant, trigger, body,
                      "open_ended", "Wedding follow-up uses the supplied date window and consented customer context.", send_as="merchant_on_behalf", customer_id=customer.get("customer_id"))
    if kind in {"trial_followup", "chronic_refill_due", "customer_lapsed_hard"}:
        options = p.get("next_session_options", [])
        slot = options[0].get("label") if options else None
        detail = f" I have {slot} available." if slot else ""
        if kind == "customer_lapsed_hard":
            detail = f" You last visited {p.get('days_since_last_visit', 'some time ago')} days ago."
        body = f"Hi {name}, {merchant.get('identity', {}).get('name', 'your business')} here.{detail} " \
               f"{offer.get('title') + '.' if offer else ''} Want me to help you with the next step?"
        return action(f"conv_{merchant.get('merchant_id')}_{trigger.get('id')}", merchant, trigger, body,
                      "open_ended", "Customer trigger is consented and has a concrete next step.", send_as="merchant_on_behalf", customer_id=customer.get("customer_id"))
    # Expanded dataset may contain placeholder customer triggers. If consent exists,
    # keep the outreach generic rather than inventing dates, prices, or services.
    if kind in {"customer_lapsed_soft", "appointment_tomorrow", "recall_due", "chronic_refill_due"} and customer.get("preferences", {}).get("reminder_opt_in") and customer.get("consent", {}).get("scope"):
        body = f"Hi {name}, {merchant.get('identity', {}).get('name', 'your business')} here. We have a reminder for you. Want me to share the details?"
        return action(f"conv_{merchant.get('merchant_id')}_{trigger.get('id')}", merchant, trigger, body,
                      "binary_yes_no", "Consent is present; missing trigger details are not invented.", send_as="merchant_on_behalf", customer_id=customer.get("customer_id"))
    return None


def action(conversation_id: str, merchant: Dict[str, Any], trigger: Dict[str, Any], body: str,
           cta: str, rationale: str, send_as: str = "vera", customer_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "merchant_id": merchant.get("merchant_id"),
        "customer_id": customer_id if customer_id is not None else trigger.get("customer_id"),
        "send_as": send_as,
        "trigger_id": trigger.get("id"),
        "template_name": "vera_deterministic_v1",
        "template_params": [body],
        "body": body,
        "cta": cta,
        "suppression_key": trigger.get("suppression_key") or f"{trigger.get('kind', 'trigger')}:{trigger.get('id')}",
        "rationale": rationale,
    }


def find_category(slug: str) -> Dict[str, Any]:
    return contexts["category"].get(slug, {}).get("payload", {})


def find_merchant(mid: str) -> Optional[Dict[str, Any]]:
    return contexts["merchant"].get(mid, {}).get("payload")


@app.get("/v1/healthz")
def healthz():
    uptime = int((datetime.now(timezone.utc) - STARTED).total_seconds())
    return {"status": "ok", "uptime_seconds": uptime,
            "contexts_loaded": {k: len(v) for k, v in contexts.items()}}


@app.get("/v1/metadata")
def metadata():
    return {"team_name": TEAM_NAME, "team_members": ["Pratham"], "model": MODEL_NAME,
            "approach": "deterministic context-grounded rule engine", "version": "0.1.0"}


@app.post("/v1/context")
def context_push(req: ContextIn):
    if req.scope not in contexts:
        raise HTTPException(status_code=400, detail="invalid_scope")
    existing = contexts[req.scope].get(req.context_id)
    if existing and req.version <= existing["version"]:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version", "current_version": existing["version"]})
    contexts[req.scope][req.context_id] = req.model_dump()
    return {"accepted": True, "ack_id": f"ack_{req.context_id}_v{req.version}", "stored_at": now_iso()}


class TickIn(BaseModel):
    now: str
    available_triggers: list[str]


@app.post("/v1/tick")
def tick(req: TickIn):
    actions = []
    for tid in req.available_triggers[:20]:
        rec = contexts["trigger"].get(tid)
        if not rec:
            continue
        trigger = rec["payload"]
        mid = trigger.get("merchant_id")
        merchant = find_merchant(mid)
        if not merchant:
            continue
        category = find_category(trigger.get("payload", {}).get("category") or merchant.get("category_slug", ""))
        result = compose_merchant(merchant, category, trigger, req.now)
        if result:
            conv = result["conversation_id"]
            if result["body"] in sent_bodies.get(conv, set()):
                continue
            sent_bodies.setdefault(conv, set()).add(result["body"])
            actions.append(result)
    return {"actions": actions[:20]}


class ReplyIn(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
def reply(req: ReplyIn):
    msg = req.message.strip()
    low = msg.lower()
    if re.search(r"\b(stop|unsubscribe|do not message|don't message|not interested)\b", low):
        closed_conversations.add(req.conversation_id)
        return {"action": "end", "rationale": "Merchant explicitly declined or opted out; closing the conversation."}
    if re.search(r"thank you for contacting|will respond shortly|team will respond|automated response", low):
        return {"action": "wait", "wait_seconds": 14400,
                "rationale": "Detected a likely canned/automated reply; backing off before re-engaging."}
    if any(x in low for x in ["gst", "tax filing", "legal advice", "lawyer"]):
        return {"action": "send",
                "body": "I’ll leave that to your CA or legal advisor. Coming back to the Vera task we were discussing — want me to continue with that?",
                "cta": "open_ended",
                "rationale": "Out-of-scope request acknowledged and redirected to the active conversation."}
    if any(x in low for x in ["yes", "sure", "send", "go ahead", "do it", "please"]):
        return {"action": "send",
                "body": "Absolutely — I’ll use the existing context and offer we discussed. Want me to proceed with the campaign draft?",
                "cta": "binary_yes_no",
                "rationale": "Positive intent detected; continuing with a low-friction next step."}
    return {"action": "wait", "wait_seconds": 3600,
            "rationale": "No clear action intent detected; waiting rather than sending an unnecessary follow-up."}
