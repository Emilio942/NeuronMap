import unittest
import requests
import time
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath("."))

class TestWebIntegration(unittest.TestCase):
    BASE_URL = "http://localhost:8083"

    def setUp(self):
        # Check if server is running
        try:
            requests.get(f"{self.BASE_URL}/health", timeout=2)
        except requests.exceptions.ConnectionError:
            self.skipTest(f"Web server is not running on {self.BASE_URL}")

    def test_analysis_flow(self):
        print("\nTesting Analysis Flow...")
        
        # 1. Submit Analysis
        payload = {
            "model_name": "gpt2",
            "input_text": "What is the capital of France?",
            "layers": ["transformer.h.11"],
            "analysis_type": "basic",
            "config": {"advanced": False}
        }
        
        print(f"Submitting analysis to {self.BASE_URL}/analyze...")
        # Use json=payload to send JSON content-type
        response = requests.post(f"{self.BASE_URL}/analyze", json=payload)
        self.assertEqual(response.status_code, 200, f"Analysis submission failed: {response.text}")
        
        data = response.json()
        self.assertIn("job_id", data)
        job_id = data["job_id"]
        print(f"Job ID: {job_id}")
        
        # 2. Poll Status
        print("Polling status...")
        max_retries = 60
        status_data = {}
        for i in range(max_retries):
            status_response = requests.get(f"{self.BASE_URL}/jobs/{job_id}")
            self.assertEqual(status_response.status_code, 200)
            status_data = status_response.json()
            
            status = status_data.get('status')
            progress = status_data.get('progress')
            print(f"Status: {status} (Progress: {progress})")
            
            if status == "completed":
                print("Analysis completed!")
                break
            
            if status == "failed":
                self.fail(f"Analysis failed: {status_data.get('error')}")
                
            time.sleep(1)
        else:
            self.fail("Analysis timed out")
            
        # 3. Verify Results
        result = status_data.get("result", {})
        self.assertIn("experiment_id", result, "experiment_id missing from result")
        self.assertIn("project_id", result, "project_id missing from result")
        
        experiment_id = result["experiment_id"]
        project_id = result["project_id"]
        print(f"Experiment ID: {experiment_id}")
        print(f"Project ID: {project_id}")
        
        # Wait for experiment to be available
        print("Waiting for experiment to be available...")
        for i in range(max_retries):
            exp_response = requests.get(f"{self.BASE_URL}/results/{project_id}/{experiment_id}")
            if exp_response.status_code == 200:
                exp_data = exp_response.json()
                status = exp_data.get("status")
                print(f"Experiment status: {status}")
                if status == "completed":
                    print("Experiment completed!")
                    break
            time.sleep(1)
        else:
            self.fail("Experiment not found or not completed after waiting")
        
        # 4. Test Visualization Generation
        print("Requesting visualization...")
        viz_payload = {
            "analysis_id": f"{project_id}:{experiment_id}",
            "plot_type": "heatmap",
            "config": {"includeStatistics": True}
        }
        
        viz_response = requests.post(f"{self.BASE_URL}/visualize", json=viz_payload)
        self.assertEqual(viz_response.status_code, 200, f"Visualization failed: {viz_response.text}")
        
        viz_data = viz_response.json()
        print(f"Visualization Response: {viz_data}")
        
        viz_job_id = viz_data.get("job_id")
        self.assertIsNotNone(viz_job_id)
        
        # Poll for visualization
        print("Polling visualization status...")
        viz_result = {}
        for i in range(max_retries):
            status_response = requests.get(f"{self.BASE_URL}/jobs/{viz_job_id}")
            self.assertEqual(status_response.status_code, 200)
            status_data = status_response.json()
            
            status = status_data.get('status')
            print(f"Viz Status: {status}")
            
            if status == "completed":
                viz_result = status_data.get("result", {})
                break
            
            if status == "failed":
                self.fail(f"Visualization failed: {status_data.get('error')}")
                
            time.sleep(1)
            
        # Check if we got a path
        if "path" in viz_result:
            path = viz_result["path"]
        else:
            self.fail("No visualization path returned")
            
        print(f"Visualization Path: {path}")
        
        # 5. Verify Static File Access (if applicable)
        self.assertTrue(os.path.exists(path) or path.startswith("/"), "Visualization path invalid")
        print("Visualization path confirmed.")

if __name__ == "__main__":
    unittest.main()
