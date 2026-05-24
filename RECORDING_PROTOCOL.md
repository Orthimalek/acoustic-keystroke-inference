# Custom Dataset Recording Protocol

## Equipment
- **Keyboard:** MacBook Pro (scissor-switch) or Dell Latitude 5430 (chiclet membrane)
- **Microphone:** iPhone Pro Max placed 17cm to the left of keyboard on folded cloth
- **Recording app:** iPhone Voice Memos (default, uncompressed)

## File Naming Convention
```
{key}_{env}_{session}.m4a        # MacBook
{key}_{env}_D_{session}.m4a      # Dell

Examples:
  a_E1_01.m4a      → MacBook, key 'a', E1 clean, session 1
  a_E1_D_01.m4a    → Dell, key 'a', E1 clean, session 1
```

## Environments
| Code | Description |
|------|-------------|
| E1 | Quiet room, door closed, no ambient noise |
| E2 | Window open, outdoor ambient noise (traffic, conversation) |

## Recording Steps
1. Place iPhone 17cm to left of keyboard on folded cloth
2. Open Voice Memos → New Recording
3. Press each key 50 times at natural pace (~1 press/second)
4. Pause 2 seconds between keys
5. Record all 36 keys in order: `0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o p q r s t u v w x y z`

## Processing Pipeline

### 1. Convert m4a → wav (macOS)
```bash
cd ~/Downloads/CustomDataset/E1_clean/m4a
for f in *.m4a; do
  afconvert -f WAVE -d LEI16 "$f" "../wav/${f%.m4a}.wav"
done
```

### 2. Segment keystrokes
```bash
python segment.py \
  --input ~/Downloads/CustomDataset/E1_clean/wav \
  --output ~/Downloads/CustomDataset/E1_clean/segmented \
  --env E1
```

### 3. Verify
```bash
python custom_dataloader.py --env E1 --max_clips 50
# Expected: Samples=1800 | Classes=36
```

## Expected Output
- ~60–100 clips per key per environment
- Capped at 50 clips/class for balanced training
- Total: 1,800 samples per environment (36 classes × 50 clips)
