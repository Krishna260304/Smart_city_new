from datetime import datetime, timedelta, timezone
import logging
import re

from fastapi import APIRouter, Depends, HTTPException

from app.auth import get_official_user, require_official_roles
from app.database import incidents, tickets, users
from app.models import TicketAssign, TicketProgressUpdate, TicketUpdateStatus
from app.roles import normalize_official_role
from app.services.audit_log import append_incident_log, get_ticket_logbook
from app.services.email_service import send_ticket_update_email
from app.services.notification_service import send_sms, send_whatsapp
from app.services.progress_ai import predict_ticket_progress
from app.utils import serialize_doc, serialize_list, to_object_id

router = APIRouter(prefix="/api/tickets")
LOGGER = logging.getLogger(__name__)
IST = timezone(timedelta(hours=5, minutes=30))


def _now_iso():
    return datetime.utcnow().isoformat()


def _today_ist_key() -> str:
    return datetime.now(timezone.utc).astimezone(IST).date().isoformat()


def _official_role(current_user: dict) -> str | None:
    return normalize_official_role(current_user.get("officialRole"))


def _is_department(current_user: dict) -> bool:
    return _official_role(current_user) == "department"


def _is_supervisor(current_user: dict) -> bool:
    return _official_role(current_user) == "supervisor"


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
            incident_doc = incidents.find_one(
                {"_id": to_object_id(incident_id)},
                {"reporterEmail": 1, "reporterId": 1, "reporterPhone": 1},
            )
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


def _resolve_worker_doc(worker_id: str) -> dict:
    worker_doc = None
    try:
        worker_doc = users.find_one({"_id": to_object_id(worker_id)})
    except Exception:
        worker_doc = users.find_one({"_id": worker_id})
    if not worker_doc:
        raise HTTPException(status_code=404, detail="Worker not found")
    if worker_doc.get("userType") != "official" or normalize_official_role(worker_doc.get("officialRole")) != "worker":
        raise HTTPException(status_code=400, detail="Selected user is not a worker account")
    return worker_doc


def _normalize_phone(value: str | None) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if len(digits) > 10:
        return digits[-10:]
    return digits


def _assignee_from_worker_doc(worker_doc: dict) -> dict:
    worker_id = str(worker_doc.get("_id") or "").strip()
    worker_name = worker_doc.get("name") or worker_doc.get("email") or worker_doc.get("phone") or "Worker"
    worker_phone = _normalize_phone(str(worker_doc.get("phone") or ""))
    worker_specialization = worker_doc.get("workerSpecialization") or "Other"
    return {
        "workerId": worker_id,
        "name": worker_name,
        "phone": worker_phone or None,
        "specialization": worker_specialization,
        "photoUrl": None,
    }


def _distinct_non_empty(values: list[str | None]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        current = str(value or "").strip()
        if not current or current in seen:
            continue
        seen.add(current)
        output.append(current)
    return output


def _build_worker_scope_query(current_user: dict) -> dict:
    user_id = str(current_user.get("id") or "").strip()
    user_phone = _normalize_phone(current_user.get("phone"))
    user_name = str(current_user.get("name") or "").strip()
    user_email = str(current_user.get("email") or "").strip()

    conditions = []
    if user_id:
        conditions.extend(
            [
                {"workerId": user_id},
                {"workerIds": user_id},
                {"assignees.workerId": user_id},
            ]
        )
    if user_phone:
        conditions.extend(
            [
                {"assigneePhone": user_phone},
                {"assigneePhones": user_phone},
                {"assignees.phone": user_phone},
            ]
        )

    for label in [user_name, user_email]:
        if not label:
            continue
        exact_regex = {"$regex": f"^{re.escape(label)}$", "$options": "i"}
        conditions.extend(
            [
                {"assigneeName": exact_regex},
                {"assignedTo": exact_regex},
                {"assigneeNames": exact_regex},
                {"assignees.name": exact_regex},
            ]
        )

    if not conditions:
        return {"_id": {"$exists": False}}
    return {"$or": conditions}


def _reopen_reset_fields() -> dict:
    return {
        "assignedTo": None,
        "assigneeName": None,
        "assigneePhone": None,
        "assigneePhotoUrl": None,
        "workerId": None,
        "workerIds": [],
        "workerSpecialization": None,
        "workerSpecializations": [],
        "assignees": [],
        "assigneeNames": [],
        "assigneePhones": [],
        "assignedBySupervisorId": None,
        "verifiedBySupervisorId": None,
        "verifiedAt": None,
        "fieldInspectorId": None,
        "fieldInspectorName": None,
        "progressPercent": 0,
        "progressSource": "awaiting_assignment",
        "progressConfidence": None,
        "progressSummary": None,
        "progressUpdatedAt": None,
        "lastInspectorUpdateAt": None,
        "inspectorReminderSentForDate": None,
    }


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
            "resolutionRate": resolution_rate,
        },
    }


@router.get("")
def get_tickets(
    status: str | None = None,
    priority: str | None = None,
    category: str | None = None,
    current_user: dict = Depends(get_official_user),
):
    query = {}
    if status:
        query["status"] = status
    if priority:
        query["priority"] = priority
    if category:
        query["category"] = category
    role = _official_role(current_user)
    if role == "worker":
        worker_scope = _build_worker_scope_query(current_user)
        query = {"$and": [query, worker_scope]} if query else worker_scope
    data = list(tickets.find(query).sort("createdAt", -1))
    return {"success": True, "data": serialize_list(data)}


@router.get("/{ticket_id}")
def get_ticket(ticket_id: str, current_user: dict = Depends(get_official_user)):
    doc = _get_ticket_doc(ticket_id)
    return {"success": True, "data": serialize_doc(doc)}


@router.get("/{ticket_id}/logbook")
def get_ticket_logs(ticket_id: str, current_user: dict = Depends(require_official_roles("department"))):
    _ = _get_ticket_doc(ticket_id)
    return {"success": True, "data": get_ticket_logbook(ticket_id)}


@router.patch("/{ticket_id}/status")
def update_status(ticket_id: str, payload: TicketUpdateStatus, current_user: dict = Depends(get_official_user)):
    existing = _get_ticket_doc(ticket_id)
    normalized_status = _normalize_ticket_status(payload.status)
    if normalized_status not in {"open", "in_progress", "resolved"}:
        raise HTTPException(status_code=400, detail="Invalid status")

    role = _official_role(current_user)
    if normalized_status == "open" and role != "department":
        raise HTTPException(status_code=403, detail="Only department role can reopen tickets")
    if normalized_status == "resolved":
        if role not in {"department", "supervisor"}:
            raise HTTPException(status_code=403, detail="Only department or supervisor role can resolve tickets")
        if role == "supervisor" and int(existing.get("reopenCount") or 0) > 0:
            raise HTTPException(status_code=403, detail="Reopened tickets can only be resolved by department")
    if normalized_status == "in_progress" and role not in {"supervisor", "department"}:
        raise HTTPException(status_code=403, detail="Only supervisor or department role can verify tickets")

    update = {"status": normalized_status, "updatedAt": _now_iso()}
    if normalized_status == "in_progress":
        update["verifiedBySupervisorId"] = current_user.get("id")
        update["verifiedAt"] = _now_iso()
    if normalized_status == "open":
        update.update(_reopen_reset_fields())

    op = {"$set": update}
    if normalized_status == "open" and _normalize_ticket_status(existing.get("status")) != "open":
        op["$inc"] = {"reopenCount": 1}
    if payload.notes:
        op["$push"] = {"notes": {"note": payload.notes, "createdAt": _now_iso(), "by": current_user.get("id")}}

    obj_id = to_object_id(ticket_id)
    tickets.update_one({"_id": obj_id}, op)
    doc = tickets.find_one({"_id": obj_id})
    if doc:
        _sync_incident_from_ticket(
            doc,
            {
                "status": doc.get("status"),
                "updatedAt": doc.get("updatedAt"),
                "assignedTo": doc.get("assignedTo"),
                "assignees": doc.get("assignees"),
                "workerIds": doc.get("workerIds"),
                "assigneeNames": doc.get("assigneeNames"),
                "workerSpecializations": doc.get("workerSpecializations"),
            },
        )
        _notify_ticket_update(doc)
        append_incident_log(
            ticket_id=ticket_id,
            incident_id=doc.get("incidentId"),
            action=f"status_{normalized_status}",
            actor=current_user,
            details={
                "previousStatus": existing.get("status"),
                "newStatus": normalized_status,
                "notes": payload.notes,
            },
        )
    return {"success": True, "data": serialize_doc(doc)}


@router.post("/{ticket_id}/assign")
def assign_ticket(
    ticket_id: str,
    payload: TicketAssign,
    current_user: dict = Depends(require_official_roles("supervisor", "department")),
):
    existing = _get_ticket_doc(ticket_id)
    if existing.get("status") == "resolved":
        raise HTTPException(status_code=400, detail="Resolved ticket must be reopened by department before assignment")

    raw_worker_ids = []
    if payload.workerId:
        raw_worker_ids.append(payload.workerId)
    if payload.workerIds:
        raw_worker_ids.extend(payload.workerIds)

    worker_ids = _distinct_non_empty(raw_worker_ids)
    if not worker_ids:
        raise HTTPException(status_code=400, detail="workerId or workerIds is required")

    assignees = []
    for worker_id in worker_ids:
        worker_doc = _resolve_worker_doc(worker_id)
        assignee = _assignee_from_worker_doc(worker_doc)
        assignees.append(assignee)

    if not assignees:
        raise HTTPException(status_code=400, detail="No valid workers selected")

    assignee_worker_ids = _distinct_non_empty([row.get("workerId") for row in assignees])
    assignee_names = _distinct_non_empty([row.get("name") for row in assignees])
    assignee_phones = _distinct_non_empty([row.get("phone") for row in assignees])
    assignee_specializations = _distinct_non_empty([row.get("specialization") for row in assignees])
    primary = assignees[0]

    update = {
        "assignedTo": ", ".join(assignee_names) if assignee_names else (primary.get("name") or "Assigned Team"),
        "assigneeName": primary.get("name"),
        "assigneePhone": primary.get("phone"),
        "assigneePhotoUrl": primary.get("photoUrl"),
        "workerId": primary.get("workerId"),
        "workerIds": assignee_worker_ids,
        "workerSpecialization": primary.get("specialization"),
        "workerSpecializations": assignee_specializations,
        "assignees": assignees,
        "assigneeNames": assignee_names,
        "assigneePhones": assignee_phones,
        "assignedBySupervisorId": current_user.get("id"),
        "updatedAt": _now_iso(),
    }
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
                "workerId": doc.get("workerId"),
                "workerIds": doc.get("workerIds"),
                "workerSpecialization": doc.get("workerSpecialization"),
                "workerSpecializations": doc.get("workerSpecializations"),
                "assignees": doc.get("assignees"),
                "assigneeNames": doc.get("assigneeNames"),
                "assigneePhones": doc.get("assigneePhones"),
                "updatedAt": doc.get("updatedAt"),
            },
        )
        append_incident_log(
            ticket_id=ticket_id,
            incident_id=doc.get("incidentId"),
            action="assign_worker",
            actor=current_user,
            details={
                "workerIds": doc.get("workerIds"),
                "workerNames": doc.get("assigneeNames"),
                "workerSpecializations": doc.get("workerSpecializations"),
                "notes": payload.notes,
            },
        )
    return {"success": True, "data": serialize_doc(doc)}


@router.post("/{ticket_id}/progress-update")
def update_progress(
    ticket_id: str,
    payload: TicketProgressUpdate,
    current_user: dict = Depends(require_official_roles("field_inspector")),
):
    existing = _get_ticket_doc(ticket_id)
    if existing.get("status") == "resolved":
        raise HTTPException(status_code=400, detail="Cannot update progress for resolved tickets")

    update_text = (payload.updateText or "").strip()
    if not update_text:
        raise HTTPException(status_code=400, detail="updateText is required")

    prediction = predict_ticket_progress(update_text)
    next_percent = int(max(0, min(100, prediction.percent)))
    current_percent = int(existing.get("progressPercent") or 0)
    # Prevent completion from regressing due to low-confidence model outputs.
    if current_percent > 0:
        next_percent = max(next_percent, current_percent)
    now = _now_iso()
    update = {
        "progressPercent": next_percent,
        "progressSource": prediction.source,
        "progressConfidence": prediction.confidence,
        "progressSummary": update_text,
        "progressUpdatedAt": now,
        "lastInspectorUpdateAt": now,
        "fieldInspectorId": current_user.get("id"),
        "fieldInspectorName": current_user.get("name") or current_user.get("email") or current_user.get("phone"),
        "inspectorReminderSentForDate": _today_ist_key(),
        "updatedAt": now,
    }
    progress_log = {
        "summary": update_text,
        "percent": next_percent,
        "source": prediction.source,
        "confidence": prediction.confidence,
        "by": current_user.get("id"),
        "byName": current_user.get("name") or current_user.get("email") or current_user.get("phone"),
        "createdAt": now,
    }

    obj_id = to_object_id(ticket_id)
    tickets.update_one({"_id": obj_id}, {"$set": update, "$push": {"progressHistory": progress_log}})
    doc = tickets.find_one({"_id": obj_id})
    if doc:
        _sync_incident_from_ticket(
            doc,
            {
                "updatedAt": doc.get("updatedAt"),
                "progressPercent": doc.get("progressPercent"),
                "progressSummary": doc.get("progressSummary"),
            },
        )
        append_incident_log(
            ticket_id=ticket_id,
            incident_id=doc.get("incidentId"),
            action="field_progress_update",
            actor=current_user,
            details={
                "progressPercent": next_percent,
                "source": prediction.source,
                "confidence": prediction.confidence,
                "summary": update_text,
            },
        )
    return {"success": True, "data": serialize_doc(doc)}
