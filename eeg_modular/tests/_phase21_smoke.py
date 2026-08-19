"""Quick smoke test for Phase 2.1 AdaptiveFeedbackEngine changes."""
import os, sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "ui_prototype"))
sys.path.insert(0, str(_ROOT))

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import numpy as np
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

from services.dashboard_state import (
    DashboardState, DIFFICULTY_HARD, DIFFICULTY_MEDIUM,
    DIFFICULTY_EASY, AdaptiveAction
)
from services.adaptive_feedback_engine import (
    AdaptiveFeedbackEngine, AdaptiveDecision, apply_adaptive_decision
)

print('=== Engine basic test ===')
engine = AdaptiveFeedbackEngine(
    negative_threshold=0.60, sustain_seconds=3.0,
    cooldown_seconds=2.0, alpha=1.0
)

# Test 1: hard -> medium with accepted=True
print('Test 1: accepted=True, hard -> medium')
state = DashboardState()
state.warmup_progress = 1.0
state.quality_level = 'trusted'
state.poor_signal = 5
state.task_difficulty = DIFFICULTY_HARD
state.attention = 45.0

start = time.time()
for i in range(10):
    d = engine.decide(
        probabilities=np.array([0.1, 0.2, 0.7]),
        accepted=True,
        quality_level='trusted',
        attention=45.0,
        task_difficulty=DIFFICULTY_HARD,
        timestamp=start + i,
        warmup_complete=True,
    )
    if d.should_emit_event:
        apply_adaptive_decision(state, d)
        print(f'  Triggered! action={d.action}, new_diff={d.new_difficulty}')
        break

interventions = [e for e in state._events if e.category == 'intervention']
print(f'  task_difficulty={state.task_difficulty}')
print(f'  interventions={len(interventions)}')
assert state.task_difficulty == DIFFICULTY_MEDIUM, 'Hard should become medium'
assert len(interventions) == 1, 'Should have exactly 1 intervention'
print('  PASSED')

# Test 2: accepted=False, high negative -> NO action
print()
print('Test 2: accepted=False, negative=0.99 -> NO action')
engine2 = AdaptiveFeedbackEngine(
    negative_threshold=0.01, sustain_seconds=0.1,
    cooldown_seconds=2.0, alpha=1.0
)
state2 = DashboardState()
state2.warmup_progress = 1.0
state2.quality_level = 'trusted'
state2.task_difficulty = DIFFICULTY_HARD
state2.attention = 45.0

for i in range(20):
    d = engine2.decide(
        probabilities=np.array([0.0, 0.01, 0.99]),
        accepted=False,  # CRITICAL: Production rejected
        quality_level='trusted',
        attention=45.0,
        task_difficulty=DIFFICULTY_HARD,
        timestamp=time.time() + i,
        warmup_complete=True,
    )
    if d.should_emit_event:
        apply_adaptive_decision(state2, d)
        print(f'  UNEXPECTED TRIGGER!')
        break

interventions2 = [e for e in state2._events if e.category == 'intervention']
print(f'  task_difficulty={state2.task_difficulty}')
print(f'  adaptive_action={state2.adaptive_action}')
print(f'  interventions={len(interventions2)}')
assert state2.task_difficulty == DIFFICULTY_HARD, 'Should stay HARD'
assert state2.adaptive_action == AdaptiveAction.NONE, 'Should be NONE'
assert len(interventions2) == 0, 'Should have 0 interventions'
print('  PASSED')

# Test 3: accepted=True triggers, then accepted=False blocks
print()
print('Test 3: accepted=True triggers, then accepted=False blocks')
engine3 = AdaptiveFeedbackEngine(
    negative_threshold=0.60, sustain_seconds=3.0,
    cooldown_seconds=2.0, alpha=1.0
)
state3 = DashboardState()
state3.warmup_progress = 1.0
state3.quality_level = 'trusted'
state3.task_difficulty = DIFFICULTY_HARD
state3.attention = 45.0

start3 = time.time()
# First: accepted=True, should trigger
for i in range(10):
    d = engine3.decide(
        probabilities=np.array([0.1, 0.2, 0.7]),
        accepted=True,
        quality_level='trusted',
        attention=45.0,
        task_difficulty=DIFFICULTY_HARD,
        timestamp=start3 + i,
        warmup_complete=True,
    )
    if d.should_emit_event:
        apply_adaptive_decision(state3, d)
        print(f'  First trigger: action={d.action}')
        break

events_before = len([e for e in state3._events if e.category == 'intervention'])
# Second: accepted=False after trigger
for i in range(20):
    d = engine3.decide(
        probabilities=np.array([0.1, 0.2, 0.7]),
        accepted=False,  # rejected
        quality_level='trusted',
        attention=45.0,
        task_difficulty=DIFFICULTY_MEDIUM,
        timestamp=start3 + 100 + i,
        warmup_complete=True,
    )
    if d.should_emit_event:
        apply_adaptive_decision(state3, d)
        print(f'  UNEXPECTED: trigger with accepted=False!')
        break

events_after = len([e for e in state3._events if e.category == 'intervention'])
print(f'  Events before={events_before}, after={events_after}')
assert events_before == events_after, 'No new intervention with accepted=False'
print('  PASSED')

# Test 4: rejected quality still blocks even if accepted=True
print()
print('Test 4: quality=rejected blocks even if accepted=True')
engine4 = AdaptiveFeedbackEngine(
    negative_threshold=0.01, sustain_seconds=0.1,
    cooldown_seconds=2.0, alpha=1.0
)
state4 = DashboardState()
state4.warmup_progress = 1.0
state4.quality_level = 'rejected'
state4.task_difficulty = DIFFICULTY_HARD
state4.attention = 45.0

d = engine4.decide(
    probabilities=np.array([0.0, 0.01, 0.99]),
    accepted=True,
    quality_level='rejected',
    attention=45.0,
    task_difficulty=DIFFICULTY_HARD,
    timestamp=time.time(),
    warmup_complete=True,
)
print(f'  action={d.action}, should_emit={d.should_emit_event}')
assert d.action == AdaptiveAction.NONE, 'Should be NONE'
assert not d.should_emit_event, 'Should not emit'
print('  PASSED')

print()
print('All engine smoke tests passed!')
