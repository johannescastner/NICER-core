#!/usr/bin/env python3
"""
Quick test script to verify GCS bucket creation permissions.

Usage:
    python3 test_gcs_permissions.py

This tests if your service account can create GCS buckets without doing a full deployment.
"""
import os
import sys
import base64
import json
from google.cloud import storage
from google.oauth2 import service_account

def test_gcs_permissions():
    """Test if service account can create GCS buckets."""
    
    # Load service account from environment
    try:
        sa_json_b64 = os.environ.get("GCP_SERVICE_ACCOUNT_BASE64")
        if not sa_json_b64:
            print("❌ GCP_SERVICE_ACCOUNT_BASE64 not set in environment")
            return False
            
        sa_json = base64.b64decode(sa_json_b64).decode("utf-8")
        sa_info = json.loads(sa_json)
        credentials = service_account.Credentials.from_service_account_info(sa_info)
        
        print(f"✅ Service account loaded: {sa_info.get('client_email')}")
    except Exception as e:
        print(f"❌ Failed to load service account: {e}")
        return False
    
    # Get project ID
    project_id = os.environ.get("GCP_PROJECT_ID") or os.environ.get("PROJECT_ID")
    if not project_id:
        project_id = sa_info.get("project_id")
    
    if not project_id:
        print("❌ PROJECT_ID not found")
        return False
    
    print(f"✅ Project ID: {project_id}")
    
    # Test bucket creation
    test_bucket_name = f"{project_id}-hf-models-test"
    region = os.environ.get("GCP_REGION", "europe-west1")
    
    try:
        client = storage.Client(project=project_id, credentials=credentials)
        print(f"✅ GCS client initialized")
        
        # Check if test bucket already exists
        bucket = client.bucket(test_bucket_name)
        if bucket.exists():
            print(f"⚠️  Test bucket {test_bucket_name} already exists, deleting...")
            bucket.delete(force=True)
            print(f"✅ Test bucket deleted")
        
        # Try to create bucket
        print(f"📦 Attempting to create bucket: {test_bucket_name}")
        bucket = client.create_bucket(test_bucket_name, location=region)
        print(f"✅ SUCCESS! Bucket created: {test_bucket_name}")
        
        # Clean up
        print(f"🧹 Cleaning up test bucket...")
        bucket.delete()
        print(f"✅ Test bucket deleted")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("Your service account has the necessary permissions.")
        print("You can now deploy and GCS caching will work!")
        return True
        
    except Exception as e:
        print(f"\n❌ PERMISSION TEST FAILED!")
        print(f"Error: {e}")
        print("\nThis means:")
        print("1. The IAM permissions haven't propagated yet (wait 60 seconds)")
        print("2. The grant script needs to be run again")
        print("3. There's an issue with the service account configuration")
        return False

if __name__ == "__main__":
    success = test_gcs_permissions()
    sys.exit(0 if success else 1)