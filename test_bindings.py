import sys
import os

# Add the directory containing the .pyd file to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'build_python', 'Release'))

import clash_royale_env as env

print("Successfully imported clash_royale_env!")

game = env.ClashRoyaleEnv(
    ai_deck=[15, 6, 0, 25, 7, 24, 34, 29],
    opp_deck=[8, 2, 38, 13, 37, 32, 29, 12]
)

print(f"Observation size: {game.observation_size()}")

obs = game.reset()
print(f"Reset returned observation of size: {len(obs)}")

for step in range(10):
    hand = game.get_hand()
    elixir = game.get_elixir()
    
    # Do nothing action
    res = game.step(-1, 0.0, 0.0)
    
    if step % 2 == 0:
        print(f"Step {step}: Elixir={elixir:.2f}, Hand={hand}, Reward={res.reward}, Done={res.done}")
    
    if res.done:
        print("Game over!")
        break

print("Python bindings test completed successfully.")
