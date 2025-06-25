#!/usr/bin/env python3
"""
NeuronMap Web Interface Launcher
===============================

Simple script to start the NeuronMap web interface.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def main():
    """Start the web interface"""
    try:
        from src.web.app import app, FLASK_AVAILABLE
        
        if not FLASK_AVAILABLE:
            print("❌ Flask is not available. Please install Flask:")
            print("   pip install flask")
            sys.exit(1)
        
        print("🚀 Starting NeuronMap Web Interface...")
        print("📊 Access the interface at: http://localhost:5000")
        print("🔄 Use Ctrl+C to stop the server")
        print()
        
        # Start Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n🛑 Web interface stopped.")
        
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
