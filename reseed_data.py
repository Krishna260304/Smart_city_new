"""
Script to clean up and reseed with proper user assignment
"""
import sys
from datetime import datetime, timedelta
from pymongo import MongoClient
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "Backend"))

from app.config.settings import settings

# Connect to MongoDB
client = MongoClient(settings.MONGO_URL)
db = client[settings.DB_NAME]

def reseed_incidents():
    """Reseed incidents without user filter - make them visible to all"""
    
    # First, find any existing user to use as reporter
    users_collection = db["users"]
    any_user = users_collection.find_one()
    
    if not any_user:
        print("❌ No users found in database. Please register a user first.")
        return
    
    user_id = any_user.get("id") or str(any_user.get("_id"))
    user_name = any_user.get("name", "Unknown")
    user_email = any_user.get("email", "unknown@example.com")
    user_phone = any_user.get("phone", "+91 0000000000")
    
    print(f"Using user: {user_name} ({user_email})")
    
    # Clear old incidents
    db.incidents.delete_many({})
    
    # Create sample incidents with current user as reporter
    base_time = datetime.utcnow()
    sample_incidents = [
        {
            "title": "Traffic Accident on Main Road",
            "description": "Multi-vehicle collision blocking traffic flow during peak hours",
            "category": "traffic",
            "priority": "high",
            "status": "in_progress",
            "location": "Main Road, Downtown",
            "latitude": 28.7041,
            "longitude": 77.1025,
            "reporterId": user_id,
            "reportedBy": user_name,
            "reporterEmail": user_email,
            "reporterPhone": user_phone,
            "assignedTo": None,
            "imageUrls": [],
            "createdAt": (base_time - timedelta(hours=2)).isoformat(),
            "updatedAt": (base_time - timedelta(hours=1)).isoformat(),
        },
        {
            "title": "Pothole on Park Avenue",
            "description": "Large pothole creating hazard for vehicles and pedestrians",
            "category": "infrastructure",
            "priority": "medium",
            "status": "open",
            "location": "Park Avenue, North Zone",
            "latitude": 28.6139,
            "longitude": 77.2090,
            "reporterId": user_id,
            "reportedBy": user_name,
            "reporterEmail": user_email,
            "reporterPhone": user_phone,
            "assignedTo": None,
            "imageUrls": [],
            "createdAt": (base_time - timedelta(hours=5)).isoformat(),
            "updatedAt": (base_time - timedelta(hours=5)).isoformat(),
        },
        {
            "title": "Street Light Malfunction",
            "description": "Broken streetlight on residential area affecting safety at night",
            "category": "infrastructure",
            "priority": "medium",
            "status": "open",
            "location": "Residential Colony, West Zone",
            "latitude": 28.5721,
            "longitude": 77.1245,
            "reporterId": user_id,
            "reportedBy": user_name,
            "reporterEmail": user_email,
            "reporterPhone": user_phone,
            "assignedTo": None,
            "imageUrls": [],
            "createdAt": (base_time - timedelta(hours=8)).isoformat(),
            "updatedAt": (base_time - timedelta(hours=8)).isoformat(),
        },
        {
            "title": "Water Pipeline Leakage",
            "description": "Significant water wastage from broken pipeline",
            "category": "utilities",
            "priority": "high",
            "status": "resolved",
            "location": "Industrial Area, East Zone",
            "latitude": 28.5921,
            "longitude": 77.2771,
            "reporterId": user_id,
            "reportedBy": user_name,
            "reporterEmail": user_email,
            "reporterPhone": user_phone,
            "assignedTo": None,
            "imageUrls": [],
            "createdAt": (base_time - timedelta(days=2)).isoformat(),
            "updatedAt": (base_time - timedelta(hours=3)).isoformat(),
        },
        {
            "title": "Park Bench Damaged",
            "description": "Broken park bench needs replacement",
            "category": "public_amenity",
            "priority": "low",
            "status": "resolved",
            "location": "Central Park",
            "latitude": 28.6329,
            "longitude": 77.3197,
            "reporterId": user_id,
            "reportedBy": user_name,
            "reporterEmail": user_email,
            "reporterPhone": user_phone,
            "assignedTo": None,
            "imageUrls": [],
            "createdAt": (base_time - timedelta(days=3)).isoformat(),
            "updatedAt": (base_time - timedelta(days=2)).isoformat(),
        },
        {
            "title": "Garbage Overflow",
            "description": "Overflowing trash bins in market area attracting pests",
            "category": "sanitation",
            "priority": "medium",
            "status": "in_progress",
            "location": "Market Square",
            "latitude": 28.6315,
            "longitude": 77.2205,
            "reporterId": user_id,
            "reportedBy": user_name,
            "reporterEmail": user_email,
            "reporterPhone": user_phone,
            "assignedTo": None,
            "imageUrls": [],
            "createdAt": (base_time - timedelta(hours=12)).isoformat(),
            "updatedAt": (base_time - timedelta(hours=6)).isoformat(),
        },
        {
            "title": "Unsafe Construction Site",
            "description": "Construction site missing safety measures and barriers",
            "category": "safety",
            "priority": "critical",
            "status": "open",
            "location": "Downtown Development Zone",
            "latitude": 28.6125,
            "longitude": 77.1540,
            "reporterId": user_id,
            "reportedBy": user_name,
            "reporterEmail": user_email,
            "reporterPhone": user_phone,
            "assignedTo": None,
            "imageUrls": [],
            "createdAt": (base_time - timedelta(hours=1)).isoformat(),
            "updatedAt": (base_time - timedelta(hours=1)).isoformat(),
        },
        {
            "title": "Illegal Parking",
            "description": "Vehicles parked in no-parking zone blocking traffic",
            "category": "traffic",
            "priority": "low",
            "status": "resolved",
            "location": "Mall Entrance",
            "latitude": 28.6200,
            "longitude": 77.2345,
            "reporterId": user_id,
            "reportedBy": user_name,
            "reporterEmail": user_email,
            "reporterPhone": user_phone,
            "assignedTo": None,
            "imageUrls": [],
            "createdAt": (base_time - timedelta(hours=24)).isoformat(),
            "updatedAt": (base_time - timedelta(hours=20)).isoformat(),
        },
    ]
    
    # Insert incidents
    result = db.incidents.insert_many(sample_incidents)
    print(f"\n✅ Inserted {len(result.inserted_ids)} incidents")
    
    # Print summary
    print("\n📊 New Database Summary:")
    print(f"   Total Incidents: {db.incidents.count_documents({})}")
    print(f"   Open: {db.incidents.count_documents({'status': 'open'})}")
    print(f"   In Progress: {db.incidents.count_documents({'status': 'in_progress'})}")
    print(f"   Resolved: {db.incidents.count_documents({'status': 'resolved'})}")
    print(f"\n✅ Database updated successfully!")

if __name__ == "__main__":
    reseed_incidents()
