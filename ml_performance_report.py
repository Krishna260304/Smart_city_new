#!/usr/bin/env python3
"""
ML Performance Report for Smart City Application
Tests YOLO detector and Priority AI classifier models
"""

import os
import sys
import time
import psutil
import statistics
from pathlib import Path

# Add project paths
backend_path = Path(__file__).parent / "Backend"
sys.path.insert(0, str(backend_path))

os.chdir(backend_path)

print("\n" + "="*80)
print("SMART CITY ML PERFORMANCE EVALUATION REPORT")
print("="*80 + "\n")

# =============================================================================
# 1. PRIORITY AI MODEL PERFORMANCE
# =============================================================================
print("1. PRIORITY AI MODEL PERFORMANCE")
print("-" * 80)

try:
    from app.services.priority_ai import predict_incident_priority
    
    # Test cases with different priorities
    test_cases = [
        {
            "name": "Critical Fire Emergency",
            "title": "Building Fire",
            "description": "Large building fire with people trapped inside, explosion potential",
            "category": "fire",
            "severity": "CRITICAL",
            "scope": "citywide",
            "source": "public_report",
            "expected": "critical"
        },
        {
            "name": "High Priority Accident",
            "title": "Major Road Accident",
            "description": "Multi-vehicle crash with injuries on main highway",
            "category": "traffic",
            "severity": "HIGH",
            "scope": "district",
            "source": "traffic_cam",
            "expected": "high"
        },
        {
            "name": "Medium Priority Pothole",
            "title": "Damaged Road",
            "description": "Large pothole in residential area causing traffic issues",
            "category": "road",
            "severity": "MEDIUM",
            "scope": "local",
            "source": "citizen",
            "expected": "medium"
        },
        {
            "name": "Low Priority Graffiti",
            "title": "Wall Graffiti",
            "description": "Minor graffiti on public wall, cosmetic issue",
            "category": "maintenance",
            "severity": "LOW",
            "scope": "local",
            "source": "citizen",
            "expected": "low"
        },
        {
            "name": "Medical Emergency",
            "title": "Person Unconscious",
            "description": "Person collapsed on street, unconscious and not breathing properly",
            "category": "medical",
            "severity": "CRITICAL",
            "scope": "local",
            "source": "call_center",
            "expected": "critical"
        }
    ]
    
    inference_times = []
    accuracy_count = 0
    
    print(f"Testing {len(test_cases)} priority prediction scenarios...\n")
    
    for idx, test in enumerate(test_cases, 1):
        start_time = time.time()
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = predict_incident_priority(
            title=test["title"],
            description=test["description"],
            category=test["category"],
            severity=test["severity"],
            scope=test["scope"],
            source=test["source"],
            location="Unknown"
        )
        
        inference_time = time.time() - start_time
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before
        
        inference_times.append(inference_time)
        
        is_correct = result.priority == test["expected"]
        if is_correct:
            accuracy_count += 1
        
        status = "✓ PASS" if is_correct else "✗ FAIL"
        print(f"{idx}. {test['name']}")
        print(f"   Expected: {test['expected']:<10} | Got: {result.priority:<10} | {status}")
        print(f"   Confidence: {result.confidence:.2%} | Model: {result.source}")
        print(f"   Inference Time: {inference_time*1000:.2f}ms | Memory: {mem_used:+.2f}MB")
        print()
    
    avg_time = statistics.mean(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    accuracy = (accuracy_count / len(test_cases)) * 100
    
    print("PRIORITY AI SUMMARY")
    print(f"  Accuracy: {accuracy:.1f}% ({accuracy_count}/{len(test_cases)} correct)")
    print(f"  Avg Inference Time: {avg_time*1000:.2f}ms")
    print(f"  Min Inference Time: {min_time*1000:.2f}ms")
    print(f"  Max Inference Time: {max_time*1000:.2f}ms")
    
except Exception as e:
    print(f"ERROR: Failed to test Priority AI - {str(e)}")
    import traceback
    traceback.print_exc()

print("\n")

# =============================================================================
# 2. YOLO DETECTOR PERFORMANCE
# =============================================================================
print("2. YOLO OBJECT DETECTION MODEL PERFORMANCE")
print("-" * 80)

try:
    import cv2
    import numpy as np
    
    # Check if YOLO model exists
    yolo_model_path = "models/best.pt"
    
    if not os.path.exists(yolo_model_path):
        print(f"WARNING: YOLO model not found at {yolo_model_path}")
        print("Model path should be: <project_root>/Backend/models/best.pt")
        print("The model appears to be configured but not available for testing.")
    else:
        try:
            from ultralytics import YOLO
            
            print(f"Loading YOLO model from: {yolo_model_path}")
            model = YOLO(yolo_model_path)
            
            # Create a test image (100x100 random image)
            test_images = [
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
                np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8),
                np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
            ]
            
            image_sizes = ["100x100", "320x320", "640x640"]
            inference_times = []
            
            print(f"\nRunning inference on {len(test_images)} test images...\n")
            
            for img, size in zip(test_images, image_sizes):
                start_time = time.time()
                results = model(img, verbose=False)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                detections = len(results[0].boxes) if results else 0
                print(f"  Image Size {size}:")
                print(f"    Detections: {detections}")
                print(f"    Inference Time: {inference_time*1000:.2f}ms")
            
            print("\nYOLO MODEL SUMMARY")
            print(f"  Model Path: {yolo_model_path}")
            print(f"  Average Inference Time: {statistics.mean(inference_times)*1000:.2f}ms")
            print(f"  Model Architecture: YOLOv8 (Ultralytics)")
            
        except ImportError:
            print("WARNING: ultralytics package not installed")
            print("Install with: pip install ultralytics")
        except Exception as e:
            print(f"ERROR: Failed to load YOLO model - {str(e)}")
            
except Exception as e:
    print(f"ERROR: YOLO testing failed - {str(e)}")
    import traceback
    traceback.print_exc()

print("\n")

# =============================================================================
# 3. SYSTEM RESOURCE ANALYSIS
# =============================================================================
print("3. SYSTEM RESOURCE ANALYSIS")
print("-" * 80)

try:
    process = psutil.Process()
    cpu_percent = process.cpu_percent(interval=1)
    memory = process.memory_info()
    memory_mb = memory.rss / 1024 / 1024
    
    print(f"Current Process Resource Usage:")
    print(f"  CPU Usage: {cpu_percent:.2f}%")
    print(f"  Memory Usage: {memory_mb:.2f}MB")
    print(f"  Memory Type: RSS (Resident Set Size)")
    
    # System-wide resources
    cpu_count = psutil.cpu_count()
    total_memory = psutil.virtual_memory().total / 1024 / 1024 / 1024
    available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
    memory_percent = psutil.virtual_memory().percent
    
    print(f"\nSystem-Wide Resources:")
    print(f"  CPU Cores: {cpu_count}")
    print(f"  Total Memory: {total_memory:.2f}GB")
    print(f"  Available Memory: {available_memory:.2f}GB")
    print(f"  Memory Usage: {memory_percent:.1f}%")
    
except Exception as e:
    print(f"ERROR: Could not get system resources - {str(e)}")

print("\n")

# =============================================================================
# 4. MODEL CONFIGURATION
# =============================================================================
print("4. MODEL CONFIGURATION")
print("-" * 80)

try:
    from app.config.settings import settings
    
    print("Priority AI Configuration:")
    print(f"  Enabled: {settings.PRIORITY_AI_ENABLED}")
    print(f"  Model: {settings.PRIORITY_AI_MODEL}")
    print(f"  Model Weight: {settings.PRIORITY_AI_MODEL_WEIGHT}")
    print(f"  Request Timeout: {settings.PRIORITY_AI_REQUEST_TIMEOUT_SECONDS}s")
    print(f"  Offline Mode: {settings.PRIORITY_AI_OFFLINE_MODE}")
    
except Exception as e:
    print(f"ERROR: Could not load settings - {str(e)}")

print("\n")

# =============================================================================
# 5. RECOMMENDATIONS
# =============================================================================
print("5. RECOMMENDATIONS")
print("-" * 80)

recommendations = [
    "✓ Priority AI is using a hybrid approach (Zero-Shot + Heuristic)",
    "✓ YOLO detector is configured for garbage overflow detection",
    "→ Monitor inference times in production (target <100ms for real-time)",
    "→ Consider caching frequently-used predictions to reduce latency",
    "→ Implement batch processing for multiple images when possible",
    "→ Add telemetry/logging to track model performance in production",
    "→ Regularly validate model predictions against ground truth",
    "→ Consider model quantization for faster inference on edge devices",
    "→ Set up monitoring for memory usage on resource-constrained devices",
]

for rec in recommendations:
    print(f"  {rec}")

print("\n" + "="*80)
print("END OF REPORT")
print("="*80 + "\n")
