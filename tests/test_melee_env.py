#!/usr/bin/env python3

import os
import sys
import time
from pathlib import Path
import numpy as np
# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "infrastructure"))

def test_melee_env():
    """Simple test of MeleeEnv functionality"""
    
    print("🎮 Testing MeleeEnv...")
    
    try:
        # Import after setting environment variables
        from melee_env import MeleeEnv, REPLAY_DIR
        import melee.enums as enums
        
        print(f"📁 Replay directory: {REPLAY_DIR}")
        
        # Count existing replays
        initial_replay_count = 0
        if REPLAY_DIR.exists():
            initial_replay_count = len(list(REPLAY_DIR.glob("*.slp")))
        print(f"Initial replay count: {initial_replay_count}")
        
        # Create environment
        print("🔧 Creating MeleeEnv...")
        env = MeleeEnv()
        
        print("✅ Environment created successfully!")
        print(f"   Frame count: {env.frame_count}")
        print(f"   Match started: {env.match_started}")
        print(f"   Game ended: {env.game_ended}")
        
        # Run game loop
        print("🚀 Starting game loop...")
        while env.gaming():
            # Create action array with first value 1, rest 0
            action = np.zeros(12)
            action[0] = 1
            env.step(action)
        
        # Check for replay generation
        print("🔍 Checking for replay files...")
        time.sleep(2)  # Give time for file writing
        
        if REPLAY_DIR.exists():
            current_replays = list(REPLAY_DIR.glob("*.slp"))
            new_replay_count = len(current_replays)
            
            print(f"Final replay count: {new_replay_count}")
            print(f"New replays: {new_replay_count - initial_replay_count}")
            
            if new_replay_count > initial_replay_count:
                newest_replay = max(current_replays, key=lambda f: f.stat().st_mtime)
                size = newest_replay.stat().st_size
                print(f"✅ New replay generated: {newest_replay}")
                print(f"   Size: {size:,} bytes ({size/1024:.1f} KB)")
                
                if size > 100:
                    print("✅ Replay file appears valid!")
                    return True
                else:
                    print("⚠️  Replay file is very small")
                    return False
            else:
                print("❌ No new replay file generated")
                return False
        else:
            print("❌ Replay directory not found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        env.stop()

if __name__ == "__main__":
    """Main test runner"""
    # Change to the project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    print(f"Working directory: {os.getcwd()}")
    
    success = test_melee_env()
    
    if success:
        print("\n✅ Test PASSED! MeleeEnv successfully generated a replay file.")
        sys.exit(0)
    else:
        print("\n❌ Test FAILED! No valid replay file was generated.")
        sys.exit(1)