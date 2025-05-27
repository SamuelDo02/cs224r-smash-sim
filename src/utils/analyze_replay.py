#!/usr/bin/env python3

import melee
import sys
from pathlib import Path

def analyze_replay(replay_path):
    """Analyze a .slp replay file and print information about it."""
    
    print(f"Analyzing replay: {replay_path}")
    
    # First, check basic file info
    file_size = Path(replay_path).stat().st_size
    print(f"File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    if file_size < 100:
        print("⚠️  Warning: File is very small, likely contains no game data")
        return
    
    # Try to read the file header to check if it's a valid Slippi file
    try:
        with open(replay_path, 'rb') as f:
            header = f.read(16)
            if not header.startswith(b'{U\x03raw'):
                print("❌ Error: File does not appear to be a valid Slippi replay file")
                print(f"   Header bytes: {header[:10]}")
                return
    except Exception as e:
        print(f"❌ Error reading file header: {e}")
        return
    
    # Create a console to read the replay
    console = melee.Console(path=str(replay_path), is_dolphin=False, allow_old_version=True)
    
    try:
        print("Attempting to connect to replay...")
        # Connect to the replay
        if not console.connect():
            print("❌ Failed to connect to replay file")
            return
        
        frame_count = 0
        game_started = False
        players_info = {}
        in_game_frames = 0
        
        print("✅ Successfully connected! Reading replay data...")
        
        while True:
            try:
                gamestate = console.step()
                if gamestate is None:
                    break
                    
                frame_count += 1
                
                # Print initial game info
                if frame_count == 1:
                    print(f"📍 Stage: {gamestate.stage}")
                    print(f"🎮 Menu state: {gamestate.menu_state}")
                    print(f"👥 Players: {len(gamestate.players)}")
                    
                # Track when the game actually starts
                if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                    if not game_started:
                        print(f"🚀 Game started at frame {frame_count}")
                        game_started = True
                    
                    in_game_frames += 1
                        
                    # Get player information
                    for port, player in gamestate.players.items():
                        if port not in players_info:
                            players_info[port] = {
                                'character': player.character,
                                'initial_stocks': player.stock,
                                'final_stocks': player.stock,
                                'max_percent': 0
                            }
                        
                        # Update player stats
                        players_info[port]['final_stocks'] = player.stock
                        players_info[port]['max_percent'] = max(
                            players_info[port]['max_percent'], 
                            player.percent
                        )
                
                # Print progress every 1800 frames (30 seconds)
                if frame_count % 1800 == 0:
                    print(f"⏱️  Processed {frame_count} frames ({frame_count/60:.1f}s)...")
                    
            except Exception as e:
                print(f"⚠️  Error at frame {frame_count}: {e}")
                break
                
        print(f"\n📊 Replay Analysis Complete!")
        print(f"   Total frames: {frame_count:,}")
        print(f"   Total duration: ~{frame_count/60:.1f} seconds")
        print(f"   In-game frames: {in_game_frames:,}")
        print(f"   In-game duration: ~{in_game_frames/60:.1f} seconds")
        
        if players_info:
            print(f"\n👥 Player Information:")
            for port, info in players_info.items():
                print(f"   Player {port}: {info['character'].name}")
                print(f"     Stocks: {info['initial_stocks']} → {info['final_stocks']}")
                print(f"     Max damage: {info['max_percent']:.1f}%")
        
            # Determine winner
            if len(players_info) >= 2:
                winner = max(players_info.keys(), key=lambda p: players_info[p]['final_stocks'])
                if players_info[winner]['final_stocks'] > 0:
                    print(f"\n🏆 Winner: Player {winner} ({players_info[winner]['character'].name})")
                else:
                    print(f"\n💀 Match ended with all players eliminated")
        else:
            print("\n⚠️  No player data found - replay may only contain menu navigation")
        
    except Exception as e:
        print(f"❌ Error reading replay: {e}")
        print(f"   This could mean:")
        print(f"   - The replay file is corrupted")
        print(f"   - The replay was created with an incompatible version")
        print(f"   - The file is not a complete Slippi replay")
        import traceback
        traceback.print_exc()
    finally:
        try:
            console.stop()
        except:
            pass

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze a Slippi replay (.slp) file")
    parser.add_argument("replay_file", nargs="?", help="Path to the .slp replay file")
    parser.add_argument("--list", "-l", action="store_true", help="List available replay files")
    
    args = parser.parse_args()
    
    # If --list flag is used, show available replays
    if args.list:
        replay_dir = Path("replays")
        if replay_dir.exists():
            print("Available replay files:")
            for slp_file in replay_dir.glob("*.slp"):
                size = slp_file.stat().st_size
                print(f"  {slp_file} ({size:,} bytes, {size/1024:.1f} KB)")
        else:
            print("No replays directory found")
        return
    
    # If no replay file specified, try to find the most recent one
    if not args.replay_file:
        replay_dir = Path("replays")
        if replay_dir.exists():
            slp_files = list(replay_dir.glob("*.slp"))
            if slp_files:
                # Get the most recent file
                replay_path = max(slp_files, key=lambda f: f.stat().st_mtime)
                print(f"No replay file specified, using most recent: {replay_path}")
            else:
                print("No replay file specified and no .slp files found in replays/")
                print("Usage: python analyze_replay.py <replay_file.slp>")
                print("   or: python analyze_replay.py --list")
                return
        else:
            print("No replay file specified and no replays/ directory found")
            print("Usage: python analyze_replay.py <replay_file.slp>")
            return
    else:
        replay_path = Path(args.replay_file)
    
    if not replay_path.exists():
        print(f"Replay file not found: {replay_path}")
        # List available replays as suggestion
        replay_dir = Path("replays")
        if replay_dir.exists():
            print("\nAvailable replay files:")
            for slp_file in replay_dir.glob("*.slp"):
                size = slp_file.stat().st_size
                print(f"  {slp_file} ({size:,} bytes)")
        return
    
    analyze_replay(replay_path)

if __name__ == "__main__":
    main() 