#!/usr/bin/env python3
"""
Model Performance Testing Script
Tests the optimized deepfake detection model for speed, accuracy, and efficiency
"""

import os
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_speed():
    """Test model inference speed"""
    logger.info("Testing model inference speed...")
    
    # Import the model
    from server import model, transform
    
    if model is None:
        logger.error("Model not available for testing")
        return None
    
    model.eval()
    
    # Create test tensors
    test_sizes = [(1, 3, 112, 112), (1, 8, 3, 112, 112)]  # Image and video
    results = {}
    
    for size in test_sizes:
        logger.info(f"Testing tensor size: {size}")
        
        # Create random test data
        if len(size) == 4:  # Image
            test_tensor = torch.randn(size)
        else:  # Video
            test_tensor = torch.randn(size)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(test_tensor)
        
        # Test inference speed
        times = []
        for _ in range(20):
            start_time = time.time()
            with torch.no_grad():
                _ = model(test_tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results[f"size_{size}"] = {
            "avg_time_ms": avg_time * 1000,
            "std_time_ms": std_time * 1000,
            "fps": 1.0 / avg_time
        }
        
        logger.info(f"  Average time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        logger.info(f"  FPS: {1.0/avg_time:.1f}")
    
    return results

def test_image_processing_speed():
    """Test image processing pipeline speed"""
    logger.info("Testing image processing pipeline speed...")
    
    from server import predict_image
    
    # Create test image
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_image_path = "test_performance_image.jpg"
    cv2.imwrite(test_image_path, test_image)
    
    try:
        times = []
        for _ in range(10):
            start_time = time.time()
            prediction, confidence = predict_image(test_image_path)
            times.append(time.time() - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results = {
            "avg_time_ms": avg_time * 1000,
            "std_time_ms": std_time * 1000,
            "fps": 1.0 / avg_time
        }
        
        logger.info(f"  Average time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        logger.info(f"  FPS: {1.0/avg_time:.1f}")
        
        return results
        
    finally:
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

def test_video_processing_speed():
    """Test video processing pipeline speed"""
    logger.info("Testing video processing pipeline speed...")
    
    from server import detectFakeVideo
    
    # Create test video
    test_video_path = "test_performance_video.mp4"
    
    # Create a simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video_path, fourcc, 30.0, (224, 224))
    
    for _ in range(30):  # 1 second at 30fps
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    
    try:
        times = []
        for _ in range(5):  # Fewer iterations for video (slower)
            start_time = time.time()
            prediction, processing_time = detectFakeVideo(test_video_path)
            times.append(processing_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results = {
            "avg_time_ms": avg_time * 1000,
            "std_time_ms": std_time * 1000,
            "fps": 1.0 / avg_time
        }
        
        logger.info(f"  Average time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        logger.info(f"  FPS: {1.0/avg_time:.1f}")
        
        return results
        
    finally:
        if os.path.exists(test_video_path):
            os.remove(test_video_path)

def test_model_accuracy():
    """Test model accuracy on synthetic data"""
    logger.info("Testing model accuracy on synthetic data...")
    
    from server import model, transform
    
    if model is None:
        logger.error("Model not available for testing")
        return None
    
    model.eval()
    
    # Create synthetic test data
    num_samples = 100
    correct_predictions = 0
    
    for i in range(num_samples):
        # Create synthetic image
        if i < num_samples // 2:
            # Create "fake" looking image (more noise, artifacts)
            image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            # Add noise to simulate fake
            noise = np.random.randint(-50, 50, (112, 112, 3), dtype=np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            expected_label = 0  # Fake
        else:
            # Create "real" looking image (smoother, more natural)
            image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            # Apply smoothing to simulate real
            image = cv2.GaussianBlur(image, (5, 5), 0)
            expected_label = 1  # Real
        
        # Convert to tensor
        pil_image = Image.fromarray(image)
        tensor = transform(pil_image).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            features, output = model(tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()
        
        if predicted_label == expected_label:
            correct_predictions += 1
    
    accuracy = correct_predictions / num_samples * 100
    logger.info(f"  Synthetic data accuracy: {accuracy:.2f}%")
    
    return {"accuracy": accuracy, "correct": correct_predictions, "total": num_samples}

def test_memory_usage():
    """Test model memory usage"""
    logger.info("Testing model memory usage...")
    
    from server import model
    
    if model is None:
        logger.error("Model not available for testing")
        return None
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # Estimate memory usage
    param_size = 4  # 4 bytes per float32 parameter
    total_memory_mb = (total_params * param_size) / (1024 * 1024)
    
    logger.info(f"  Estimated memory usage: {total_memory_mb:.2f} MB")
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "memory_mb": total_memory_mb
    }

def run_comprehensive_test():
    """Run comprehensive performance test"""
    logger.info("🚀 Starting Comprehensive Model Performance Test")
    logger.info("=" * 60)
    
    results = {}
    
    # Test model speed
    logger.info("\n📊 Testing Model Inference Speed...")
    speed_results = test_model_speed()
    if speed_results:
        results["model_speed"] = speed_results
    
    # Test image processing speed
    logger.info("\n🖼️ Testing Image Processing Speed...")
    image_speed = test_image_processing_speed()
    if image_speed:
        results["image_processing"] = image_speed
    
    # Test video processing speed
    logger.info("\n🎥 Testing Video Processing Speed...")
    video_speed = test_video_processing_speed()
    if video_speed:
        results["video_processing"] = video_speed
    
    # Test model accuracy
    logger.info("\n🎯 Testing Model Accuracy...")
    accuracy_results = test_model_accuracy()
    if accuracy_results:
        results["accuracy"] = accuracy_results
    
    # Test memory usage
    logger.info("\n💾 Testing Memory Usage...")
    memory_results = test_memory_usage()
    if memory_results:
        results["memory"] = memory_results
    
    # Performance summary
    logger.info("\n" + "=" * 60)
    logger.info("📈 PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    
    if "image_processing" in results:
        img_time = results["image_processing"]["avg_time_ms"]
        logger.info(f"🖼️ Image Processing: {img_time:.2f}ms {'✅' if img_time < 5000 else '⚠️'}")
    
    if "video_processing" in results:
        vid_time = results["video_processing"]["avg_time_ms"]
        logger.info(f"🎥 Video Processing: {vid_time:.2f}ms {'✅' if vid_time < 5000 else '⚠️'}")
    
    if "accuracy" in results:
        acc = results["accuracy"]["accuracy"]
        logger.info(f"🎯 Model Accuracy: {acc:.2f}% {'✅' if acc > 80 else '⚠️'}")
    
    if "memory" in results:
        mem = results["memory"]["memory_mb"]
        logger.info(f"💾 Memory Usage: {mem:.2f} MB {'✅' if mem < 100 else '⚠️'}")
    
    # Save results
    results_path = "model/performance_test_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {results_path}")
    logger.info("🎉 Performance test completed!")
    
    return results

if __name__ == "__main__":
    run_comprehensive_test()
