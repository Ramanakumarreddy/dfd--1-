#!/usr/bin/env python3
"""
Performance Monitoring Script
Monitors real-time performance metrics for the deepfake detection system
"""

import os
import time
import psutil
import threading
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Real-time performance monitoring for the deepfake detection system"""
    
    def __init__(self):
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'network_io': [],
            'processing_times': [],
            'accuracy_metrics': []
        }
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start performance monitoring in background thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                network_io = psutil.net_io_counters()
                
                # Store metrics with timestamp
                timestamp = datetime.now().isoformat()
                
                self.metrics['cpu_usage'].append({
                    'timestamp': timestamp,
                    'value': cpu_percent
                })
                
                self.metrics['memory_usage'].append({
                    'timestamp': timestamp,
                    'value': memory.percent
                })
                
                self.metrics['disk_io'].append({
                    'timestamp': timestamp,
                    'read_bytes': disk_io.read_bytes if disk_io else 0,
                    'write_bytes': disk_io.write_bytes if disk_io else 0
                })
                
                self.metrics['network_io'].append({
                    'timestamp': timestamp,
                    'bytes_sent': network_io.bytes_sent if network_io else 0,
                    'bytes_recv': network_io.bytes_recv if network_io else 0
                })
                
                # Keep only last 1000 entries to prevent memory bloat
                for key in self.metrics:
                    if len(self.metrics[key]) > 1000:
                        self.metrics[key] = self.metrics[key][-1000:]
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)
    
    def record_processing_time(self, task_type, processing_time, accuracy=None):
        """Record processing time for specific tasks"""
        timestamp = datetime.now().isoformat()
        
        record = {
            'timestamp': timestamp,
            'task_type': task_type,
            'processing_time': processing_time
        }
        
        if accuracy is not None:
            record['accuracy'] = accuracy
        
        self.metrics['processing_times'].append(record)
        
        # Log performance metrics
        if processing_time > 5.0:
            logger.warning(f"⚠️ {task_type} took {processing_time:.3f}s (above 5s target)")
        else:
            logger.info(f"✅ {task_type} completed in {processing_time:.3f}s")
    
    def record_accuracy(self, task_type, prediction, confidence, actual_label=None):
        """Record accuracy metrics"""
        timestamp = datetime.now().isoformat()
        
        record = {
            'timestamp': timestamp,
            'task_type': task_type,
            'prediction': prediction,
            'confidence': confidence
        }
        
        if actual_label is not None:
            record['actual_label'] = actual_label
            record['correct'] = prediction == actual_label
        
        self.metrics['accuracy_metrics'].append(record)
    
    def get_performance_summary(self):
        """Get current performance summary"""
        if not self.metrics['processing_times']:
            return "No performance data available"
        
        # Calculate average processing times
        image_times = [m['processing_time'] for m in self.metrics['processing_times'] 
                      if m['task_type'] == 'image_detection']
        video_times = [m['processing_time'] for m in self.metrics['processing_times'] 
                      if m['task_type'] == 'video_detection']
        
        summary = {
            'current_time': datetime.now().isoformat(),
            'system_metrics': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            },
            'performance_metrics': {
                'image_processing': {
                    'avg_time': sum(image_times) / len(image_times) if image_times else 0,
                    'total_processed': len(image_times),
                    'within_target': len([t for t in image_times if t <= 5.0]) / len(image_times) * 100 if image_times else 0
                },
                'video_processing': {
                    'avg_time': sum(video_times) / len(video_times) if video_times else 0,
                    'total_processed': len(video_times),
                    'within_target': len([t for t in video_times if t <= 5.0]) / len(video_times) * 100 if video_times else 0
                }
            }
        }
        
        return summary
    
    def save_metrics(self, filepath='model/performance_metrics.json'):
        """Save performance metrics to file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Add summary to metrics
            metrics_with_summary = self.metrics.copy()
            metrics_with_summary['summary'] = self.get_performance_summary()
            
            with open(filepath, 'w') as f:
                json.dump(metrics_with_summary, f, indent=2, default=str)
            
            logger.info(f"Performance metrics saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            return False
    
    def generate_performance_report(self):
        """Generate a comprehensive performance report"""
        summary = self.get_performance_summary()
        
        report = f"""
🚀 DEEPFAKE DETECTION SYSTEM - PERFORMANCE REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 SYSTEM PERFORMANCE
{'='*30}
CPU Usage: {summary['system_metrics']['cpu_usage']:.1f}%
Memory Usage: {summary['system_metrics']['memory_usage']:.1f}%
Disk Usage: {summary['system_metrics']['disk_usage']:.1f}%

🎯 PROCESSING PERFORMANCE
{'='*30}
Image Detection:
  • Average Time: {summary['performance_metrics']['image_processing']['avg_time']:.3f}s
  • Total Processed: {summary['performance_metrics']['image_processing']['total_processed']}
  • Within 5s Target: {summary['performance_metrics']['image_processing']['within_target']:.1f}%

Video Detection:
  • Average Time: {summary['performance_metrics']['video_processing']['avg_time']:.3f}s
  • Total Processed: {summary['performance_metrics']['video_processing']['total_processed']}
  • Within 5s Target: {summary['performance_metrics']['video_processing']['within_target']:.1f}%

📈 PERFORMANCE TARGETS
{'='*30}
✅ Image Processing: < 5 seconds
✅ Video Processing: < 5 seconds
✅ System Resource Usage: < 80%
✅ Model Accuracy: > 85%

🎉 STATUS: {'OPTIMAL' if self._is_performance_optimal(summary) else 'NEEDS OPTIMIZATION'}
"""
        
        return report
    
    def _is_performance_optimal(self, summary):
        """Check if performance meets targets"""
        img_optimal = summary['performance_metrics']['image_processing']['within_target'] >= 90
        vid_optimal = summary['performance_metrics']['video_processing']['within_target'] >= 90
        sys_optimal = summary['system_metrics']['cpu_usage'] < 80 and summary['system_metrics']['memory_usage'] < 80
        
        return img_optimal and vid_optimal and sys_optimal

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def start_performance_monitoring():
    """Start performance monitoring"""
    performance_monitor.start_monitoring()

def stop_performance_monitoring():
    """Stop performance monitoring"""
    performance_monitor.stop_monitoring()

def record_task_performance(task_type, processing_time, accuracy=None):
    """Record task performance"""
    performance_monitor.record_processing_time(task_type, processing_time, accuracy)

def get_performance_status():
    """Get current performance status"""
    return performance_monitor.get_performance_summary()

if __name__ == "__main__":
    # Test performance monitoring
    print("Starting performance monitoring test...")
    start_performance_monitoring()
    
    # Simulate some tasks
    time.sleep(10)
    record_task_performance('image_detection', 2.5)
    record_task_performance('video_detection', 4.8)
    
    time.sleep(5)
    
    # Generate report
    report = performance_monitor.generate_performance_report()
    print(report)
    
    # Save metrics
    performance_monitor.save_metrics()
    
    # Stop monitoring
    stop_performance_monitoring()
    print("Performance monitoring test completed!")
