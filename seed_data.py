"""
Script to seed sample incident data into MongoDB
Run this from the project root: python seed_data.py
"""
import sys
from datetime import datetime, timedelta
from pymongo import MongoClient
import os
from pathlib import Path

# Add Backend to path
sys.path.insert(0, str(Path(__file__).parent / "Backend"))

from app.config.settings import settings
from app.auth import hash_password

# Connect to MongoDB
client = MongoClient(settings.MONGO_URL)
db = client[settings.DB_NAME]

# Clear existing data (optional)
# db.incidents.delete_many({})
# db.users.delete_many({})

# Create test users
test_users = [
    {
        "_id": "official_1",
        "id": "official_1",
        "name": "Officer John",
        "email": "officer@safelive.in",
        "phone": "+91 9876543210",
        "userType": "official",
        "password": hash_password("password123"),
        "createdAt": datetime.utcnow().isoformat(),
    },
    {
        "_id": "citizen_1",
        "id": "citizen_1",
        "name": "Raj Kumar",
        "email": "raj@example.com",
        "phone": "+91 9123456789",
        "userType": "citizen",
        "password": hash_password("password123"),
        "createdAt": datetime.utcnow().isoformat(),
    },
]

# Create sample incidents
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
        "reporterId": "citizen_1",
        "reportedBy": "Raj Kumar",
        "reporterEmail": "raj@example.com",
        "reporterPhone": "+91 9123456789",
        "assignedTo": "official_1",
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
        "reporterId": "citizen_1",
        "reportedBy": "Raj Kumar",
        "reporterEmail": "raj@example.com",
        "reporterPhone": "+91 9123456789",
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
        "reporterId": "citizen_1",
        "reportedBy": "Raj Kumar",
        "reporterEmail": "raj@example.com",
        "reporterPhone": "+91 9123456789",
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
        "reporterId": "citizen_1",
        "reportedBy": "Raj Kumar",
        "reporterEmail": "raj@example.com",
        "reporterPhone": "+91 9123456789",
        "assignedTo": "official_1",
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
        "reporterId": "citizen_1",
        "reportedBy": "Raj Kumar",
        "reporterEmail": "raj@example.com",
        "reporterPhone": "+91 9123456789",
        "assignedTo": "official_1",
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
        "reporterId": "citizen_1",
        "reportedBy": "Raj Kumar",
        "reporterEmail": "raj@example.com",
        "reporterPhone": "+91 9123456789",
        "assignedTo": "official_1",
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
        "reporterId": "citizen_1",
        "reportedBy": "Raj Kumar",
        "reporterEmail": "raj@example.com",
        "reporterPhone": "+91 9123456789",
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
        "reporterId": "citizen_1",
        "reportedBy": "Raj Kumar",
        "reporterEmail": "raj@example.com",
        "reporterPhone": "+91 9123456789",
        "assignedTo": "official_1",
        "imageUrls": [],
        "createdAt": (base_time - timedelta(hours=24)).isoformat(),
        "updatedAt": (base_time - timedelta(hours=20)).isoformat(),
    },
]

def seed_database():
    """Seed the database with sample data"""
    try:
        # Insert users
        users_collection = db["users"]
        for user in test_users:
            users_collection.update_one(
                {"email": user["email"]},
                {"$set": user},
                upsert=True
            )
        print(f"✅ Inserted/Updated {len(test_users)} users")

        # Insert incidents
        incidents_collection = db["incidents"]
        result = incidents_collection.insert_many(sample_incidents, ordered=False)
        print(f"✅ Inserted {len(result.inserted_ids)} sample incidents")

        # Print summary
        print("\n📊 Database Summary:")
        print(f"   Total Incidents: {incidents_collection.count_documents({})}")
        print(f"   Open: {incidents_collection.count_documents({'status': 'open'})}")
        print(f"   In Progress: {incidents_collection.count_documents({'status': 'in_progress'})}")
        print(f"   Resolved: {incidents_collection.count_documents({'status': 'resolved'})}")
        print(f"   Total Users: {users_collection.count_documents({})}")
        print("\n✅ Database seeding completed successfully!")

    except Exception as e:
        print(f"❌ Error seeding database: {e}")
        raise

if __name__ == "__main__":
    seed_database()
