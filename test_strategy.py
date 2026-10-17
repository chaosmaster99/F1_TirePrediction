"""Quick strategy test."""
from src.strategy import PitStopStrategy

strat = PitStopStrategy()
strat.load_models()
print("XGB ready:", strat._xgb_ready)

# Test LEC at Melbourne, starting on MEDIUM
print("\n=== LEC @ Melbourne, MEDIUM start ===")
result = strat.generate_strategy(
    starting_compound="MEDIUM",
    total_race_laps=58,
    driver="LEC",
    circuit="Melbourne",
    threshold=1.5,
)
for s in result["stints"]:
    print(f"  {s['compound']:6s}: laps {s['start_lap']}-{s['end_lap']} "
          f"({s['stint_length']} laps), max_deg={s['max_degradation']:.3f}s")

# Test starting on SOFT
print("\n=== LEC @ Melbourne, SOFT start ===")
result2 = strat.generate_strategy(
    starting_compound="SOFT",
    total_race_laps=58,
    driver="LEC",
    circuit="Melbourne",
    threshold=1.2,
)
for s in result2["stints"]:
    print(f"  {s['compound']:6s}: laps {s['start_lap']}-{s['end_lap']} "
          f"({s['stint_length']} laps), max_deg={s['max_degradation']:.3f}s")

# Test VER at Sakhir (high deg)
print("\n=== VER @ Sakhir, MEDIUM start ===")
result3 = strat.generate_strategy(
    starting_compound="MEDIUM",
    total_race_laps=57,
    driver="VER",
    circuit="Sakhir",
    threshold=1.5,
)
for s in result3["stints"]:
    print(f"  {s['compound']:6s}: laps {s['start_lap']}-{s['end_lap']} "
          f"({s['stint_length']} laps), max_deg={s['max_degradation']:.3f}s")

# Test HARD start
print("\n=== LEC @ Melbourne, HARD start ===")
result4 = strat.generate_strategy(
    starting_compound="HARD",
    total_race_laps=58,
    driver="LEC",
    circuit="Melbourne",
    threshold=2.0,
)
for s in result4["stints"]:
    print(f"  {s['compound']:6s}: laps {s['start_lap']}-{s['end_lap']} "
          f"({s['stint_length']} laps), max_deg={s['max_degradation']:.3f}s")
