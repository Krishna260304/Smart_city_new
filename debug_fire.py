#!/usr/bin/env python3
"""
Debug ML Model - Detailed Fire Classification Analysis
"""

import os
import sys
from pathlib import Path

backend_path = Path(__file__).parent / "Backend"
sys.path.insert(0, str(backend_path))
os.chdir(backend_path)

from app.services.priority_ai import PriorityClassifier

print("\n" + "="*80)
print("PRIORITY AI DEBUG - FIRE CLASSIFICATION ANALYSIS")
print("="*80 + "\n")

classifier = PriorityClassifier()

# Fire test case
fire_test = {
    "title": "Building Fire",
    "description": "Large building fire with people trapped inside, explosion potential",
    "category": "fire",
    "severity": "CRITICAL",
    "scope": "citywide",
    "source": "public_report",
}

print("Test Case: Critical Fire Emergency")
print(f"  Title: {fire_test['title']}")
print(f"  Description: {fire_test['description']}")
print(f"  Category: {fire_test['category']}")
print(f"  Severity: {fire_test['severity']}")
print(f"  Scope: {fire_test['scope']}\n")

# Get heuristic scores
print("HEURISTIC SCORES:")
heuristic = classifier._heuristic_scores(
    title=fire_test['title'],
    description=fire_test['description'],
    category=fire_test['category'],
    severity=fire_test['severity'],
    scope=fire_test['scope'],
    source=fire_test['source'],
    location="Unknown"
)
print(f"  Critical: {heuristic.get('critical', 0):.4f}")
print(f"  High: {heuristic.get('high', 0):.4f}")
print(f"  Medium: {heuristic.get('medium', 0):.4f}")
print(f"  Low: {heuristic.get('low', 0):.4f}")

# Get zero-shot scores
print("\nZERO-SHOT MODEL SCORES:")
text = " ".join(filter(None, [
    fire_test.get('title'),
    fire_test.get('description'),
    f"Category {fire_test.get('category')}",
    f"Severity {fire_test.get('severity')}"
]))
zero_shot = classifier._zero_shot_model.predict(text)
if zero_shot:
    print(f"  Critical: {zero_shot.get('critical', 0):.4f}")
    print(f"  High: {zero_shot.get('high', 0):.4f}")
    print(f"  Medium: {zero_shot.get('medium', 0):.4f}")
    print(f"  Low: {zero_shot.get('low', 0):.4f}")
else:
    print("  (No AI model scores available)")

# Get combined prediction
result = classifier.predict(
    title=fire_test['title'],
    description=fire_test['description'],
    category=fire_test['category'],
    severity=fire_test['severity'],
    scope=fire_test['scope'],
    source=fire_test['source'],
    location="Unknown"
)

print("\nFINAL PREDICTION:")
print(f"  Priority: {result.priority}")
print(f"  Confidence: {result.confidence:.4f}")
print(f"  Source: {result.source}")

print("\n" + "="*80)
