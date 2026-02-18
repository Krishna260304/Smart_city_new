from datetime import datetime, timedelta
import logging
import os
from fastapi import APIRouter, Depends, HTTPException
from app.database import incidents, tickets, users
from app.auth import get_official_user, is_head_supervisor_account
from app.models import TicketUpdateStatus, TicketAssign
from app.utils import serialize_doc, serialize_list, to_object_id
from app.services.notification_service import send_sms, send_whatsapp
from app.services.email_service import send_ticket_update_email
from app.services.image_service import save_image

router = APIRouter(prefix="/api/tickets")
LOGGER = logging.getLogger(__name__)

def _now_iso():
    return datetime.utcnow().isoformat()

def _get_ticket_doc(ticket_id: str):
    try:
        obj_id = to_object_id(ticket_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ticket id")
    doc = tickets.find_one({"_id": obj_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return doc

def _resolve_ticket_reporter_email(doc: dict) -> str | None:
    direct_email = (doc.get("reporterEmail") or "").strip()
    if direct_email and "@" in direct_email:
        return direct_email

    incident_doc = None
    incident_id = (doc.get("incidentId") or "").strip()
    if incident_id:
        try:
            incident_doc = incidents.find_one({"_id": to_object_id(incident_id)}, {"reporterEmail": 1, "reporterId": 1, "reporterPhone": 1})
        except Exception:
            incident_doc = None

    incident_email = ((incident_doc or {}).get("reporterEmail") or "").strip()
    if incident_email and "@" in incident_email:
        return incident_email

    reporter_id = (doc.get("reporterId") or (incident_doc or {}).get("reporterId") or "").strip()
    if reporter_id:
        user_doc = None
        try:
            user_doc = users.find_one({"_id": to_object_id(reporter_id)}, {"email": 1})
        except Exception:
            user_doc = users.find_one({"_id": reporter_id}, {"email": 1})
        user_email = ((user_doc or {}).get("email") or "").strip()
        if user_email and "@" in user_email:
            return user_email

    reporter_phone = (doc.get("reporterPhone") or (incident_doc or {}).get("reporterPhone") or "").strip()
    if reporter_phone:
        user_doc = users.find_one({"phone": reporter_phone}, {"email": 1})
        user_email = ((user_doc or {}).get("email") or "").strip()
        if user_email and "@" in user_email:
            return user_email

    return None

def _notify_ticket_update(doc: dict):
    message = f"SafeLive ticket update: {doc.get('title', 'Ticket')} is now {doc.get('status', 'updated')}."
    if doc.get("reporterPhone"):
        sms_ok, sms_error = send_sms(doc.get("reporterPhone"), message)
        if not sms_ok:
            LOGGER.warning("SMS notification failed for ticket %s: %s", doc.get("_id"), sms_error)
        wa_ok, wa_error = send_whatsapp(doc.get("reporterPhone"), message)
        if not wa_ok:
            LOGGER.warning("WhatsApp notification failed for ticket %s: %s", doc.get("_id"), wa_error)
    status_value = (doc.get("status") or "").strip().lower()
    reporter_email = _resolve_ticket_reporter_email(doc)
    if reporter_email and not doc.get("reporterEmail") and doc.get("_id"):
        try:
            tickets.update_one({"_id": doc.get("_id")}, {"$set": {"reporterEmail": reporter_email}})
        except Exception:
            pass
    if reporter_email and status_value == "resolved":
        try:
            send_ticket_update_email(
                reporter_email,
                doc.get("title", "Ticket"),
                doc.get("status", "updated"),
            )
        except Exception as exc:
            LOGGER.warning("Email notification failed for ticket %s: %s", doc.get("_id"), exc)
    elif status_value == "resolved":
        LOGGER.warning("Resolved email skipped: reporter email unavailable for ticket %s", doc.get("_id"))

def _save_assignee_photo(photo: str | None) -> str | None:
    if not photo:
        return None
    try:
        path = save_image(photo)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid assignee photo")
    filename = os.path.basename(path)
    return f"/images/{filename}"

def _normalize_ticket_status(value: str) -> str:
    status = (value or "").strip().lower()
    if status == "verified":
        return "in_progress"
    return status

def _incident_selector_from_ticket(doc: dict) -> dict | None:
    incident_id = (doc.get("incidentId") or "").strip()
    if not incident_id:
        return None
    try:
        return {"_id": to_object_id(incident_id)}
    except Exception:
        return {"_id": incident_id}

def _sync_incident_from_ticket(doc: dict, updates: dict):
    selector = _incident_selector_from_ticket(doc)
    if not selector or not updates:
        return
    incidents.update_one(selector, {"$set": updates})


def _normalize_phone_digits(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch for ch in value if ch.isdigit())


def _find_official_by_phone(phone_value: str | None):
    digits = _normalize_phone_digits(phone_value)
    if not digits:
        return None
    suffix = digits[-10:] if len(digits) >= 10 else digits
    candidates = {digits, suffix, f"+{digits}", f"+{suffix}", f"+91{suffix}"}
    query = {"phone": {"$in": sorted({c for c in candidates if c})}}
    try:
        return users.find_one(query)
    except Exception:
        return None


def _attach_assignee_metadata(update: dict, assignee_phone: str | None):
    official_doc = _find_official_by_phone(assignee_phone)
    if not official_doc:
        update.pop("assigneeUserId", None)
        update.pop("assigneeEmail", None)
        return
    update["assigneeUserId"] = str(official_doc.get("_id"))
    email_value = (official_doc.get("email") or "").strip()
    if email_value:
        update["assigneeEmail"] = email_value


def _notify_ticket_reopened(doc: dict, reopened_by: dict):
    supervisor_name = reopened_by.get("name") or reopened_by.get("email") or "Head Supervisor"
    ticket_title = doc.get("title", "Ticket")
    message = (
        f"SafeLive notice: Ticket '{ticket_title}' has been reopened by {supervisor_name}. "
        "Please review and take action."
    )

    contact_phone = doc.get("assigneePhone") or doc.get("assignedToPhone")
    if contact_phone:
        sms_ok, sms_err = send_sms(contact_phone, message)
        if not sms_ok and sms_err:
            LOGGER.warning("Ticket %s reopen SMS failed: %s", doc.get("_id"), sms_err)
        wa_ok, wa_err = send_whatsapp(contact_phone, message)
        if not wa_ok and wa_err:
            LOGGER.warning("Ticket %s reopen WhatsApp failed: %s", doc.get("_id"), wa_err)

    assignee_email = (doc.get("assigneeEmail") or "").strip()
    if not assignee_email:
        assignee_user_id = (doc.get("assigneeUserId") or "").strip()
        if assignee_user_id:
            try:
                assignee_doc = users.find_one({"_id": to_object_id(assignee_user_id)})
            except Exception:
                assignee_doc = users.find_one({"_id": assignee_user_id})
            assignee_email = ((assignee_doc or {}).get("email") or "").strip()

    if assignee_email:
        try:
            send_ticket_update_email(assignee_email, ticket_title, "Reopened by Head Supervisor")
        except Exception as exc:
            LOGGER.warning("Ticket %s reopen email failed: %s", doc.get("_id"), exc)

    warning_payload = {
        "message": message,
        "issuedAt": _now_iso(),
        "supervisorName": supervisor_name,
    }
    try:
        tickets.update_one(
            {"_id": doc.get("_id")},
            {"$set": {
                "reopenWarning": warning_payload,
            }},
        )
        doc["reopenWarning"] = warning_payload
    except Exception as exc:
        LOGGER.warning("Ticket %s warning persistence failed: %s", doc.get("_id"), exc)

@router.get("/stats")
def get_stats(current_user: dict = Depends(get_official_user)):
    total = tickets.count_documents({})
    open_t = tickets.count_documents({"status": "open"})
    in_prog = tickets.count_documents({"status": "in_progress"})
    resolved = tickets.count_documents({"status": "resolved"})
    since = (datetime.utcnow() - timedelta(days=1)).isoformat()
    resolved_today = tickets.count_documents({"status": "resolved", "updatedAt": {"$gte": since}})
    resolution_rate = round((resolved / total) * 100, 2) if total > 0 else 0
    avg_response = "N/A"
    return {
        "success": True,
        "data": {
            "totalTickets": total,
            "openTickets": open_t,
            "inProgress": in_prog,
            "resolvedToday": resolved_today,
            "avgResponseTime": avg_response,
            "resolutionRate": resolution_rate
        }
    }

@router.get("")
def get_tickets(status: str | None = None, priority: str | None = None, category: str | None = None, current_user: dict = Depends(get_official_user)):
    query = {}
    if status:
        query["status"] = status
    if priority:
        query["priority"] = priority
    if category:
        query["category"] = category
    data = list(tickets.find(query).sort("createdAt", -1))
    return {"success": True, "data": serialize_list(data)}

@router.get("/{ticket_id}")
def get_ticket(ticket_id: str, current_user: dict = Depends(get_official_user)):
    doc = _get_ticket_doc(ticket_id)
    return {"success": True, "data": serialize_doc(doc)}

@router.patch("/{ticket_id}/status")
def update_status(ticket_id: str, payload: TicketUpdateStatus, current_user: dict = Depends(get_official_user)):
    existing = _get_ticket_doc(ticket_id)
    normalized_status = _normalize_ticket_status(payload.status)
    if normalized_status not in {"open", "in_progress", "resolved"}:
        raise HTTPException(status_code=400, detail="Invalid status")
    was_resolved = (existing.get("status") or "").strip().lower() == "resolved"
    reopening = normalized_status == "open" and was_resolved
    if reopening and not is_head_supervisor_account(current_user):
        raise HTTPException(status_code=403, detail="Only head supervisors can reopen resolved tickets")

    update = {"status": normalized_status, "updatedAt": _now_iso()}
    reopened_by_supervisor = reopening and is_head_supervisor_account(current_user)
    if reopened_by_supervisor:
        update["reopenedBy"] = {
            "id": current_user.get("id"),
            "name": current_user.get("name") or current_user.get("email"),
            "timestamp": _now_iso(),
        }
    clear_warning = not reopened_by_supervisor and bool(existing.get("reopenWarning"))
    op = {"$set": update}
    if payload.notes:
        op["$push"] = {"notes": {"note": payload.notes, "createdAt": _now_iso(), "by": current_user.get("id")}}
    if clear_warning:
        op.setdefault("$unset", {})["reopenWarning"] = ""
    obj_id = to_object_id(ticket_id)
    tickets.update_one({"_id": obj_id}, op)
    doc = tickets.find_one({"_id": obj_id})
    if doc:
        _sync_incident_from_ticket(
            doc,
            {
                "status": doc.get("status"),
                "updatedAt": doc.get("updatedAt"),
            },
        )
        _notify_ticket_update(doc)
        if reopened_by_supervisor:
            _notify_ticket_reopened(doc, current_user)
    return {"success": True, "data": serialize_doc(doc)}

@router.post("/{ticket_id}/assign")
def assign_ticket(ticket_id: str, payload: TicketAssign, current_user: dict = Depends(get_official_user)):
    existing = _get_ticket_doc(ticket_id)

    assignee_name = (payload.assigneeName or payload.assignedTo or "").strip()
    assignee_phone = (payload.assigneePhone or "").strip()
    phone_digits = "".join(ch for ch in assignee_phone if ch.isdigit())

    if not assignee_name:
        raise HTTPException(status_code=400, detail="Assignee name is required")
    if not phone_digits or len(phone_digits) < 10 or len(phone_digits) > 15:
        raise HTTPException(status_code=400, detail="A valid assignee phone number is required")

    assignee_photo_url = existing.get("assigneePhotoUrl")
    photo_data = (payload.assigneePhoto or "").strip()
    if photo_data:
        assignee_photo_url = _save_assignee_photo(photo_data)
    if not assignee_photo_url:
        raise HTTPException(status_code=400, detail="Assignee photo is required")

    update = {
        "assignedTo": assignee_name,
        "assigneeName": assignee_name,
        "assigneePhone": phone_digits,
        "assigneePhotoUrl": assignee_photo_url,
        "updatedAt": _now_iso(),
    }
    _attach_assignee_metadata(update, phone_digits)
    op = {"$set": update}
    if payload.notes:
        op["$push"] = {"notes": {"note": payload.notes, "createdAt": _now_iso(), "by": current_user.get("id")}}
    obj_id = to_object_id(ticket_id)
    tickets.update_one({"_id": obj_id}, op)
    doc = tickets.find_one({"_id": obj_id})
    if doc:
        _sync_incident_from_ticket(
            doc,
            {
                "assignedTo": doc.get("assignedTo"),
                "assigneeName": doc.get("assigneeName"),
                "assigneePhone": doc.get("assigneePhone"),
                "assigneePhotoUrl": doc.get("assigneePhotoUrl"),
                "updatedAt": doc.get("updatedAt"),
            },
        )
        _notify_ticket_update(doc)
    return {"success": True, "data": serialize_doc(doc)}
