"""
Script to evaluate the trained Decision Tree agent against other agents.
Runs tournaments to gather statistical data (wins, losses, draws) as requested.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from src.ai.agents import make_dt_agent, make_mcts_agent, random_agent
from src.utils.evaluation import tournament

DATA_DIR = Path('data')

def main():
    tree_path = DATA_DIR / 'popout_tree.pkl'
    if not tree_path.exists():
        print(f"Error: Could not find {tree_path}. Train the model first.")
        return

    print("Loading Decision Tree...")
    with open(tree_path, 'rb') as f:
        dt_model = pickle.load(f)

    # Instantiate agents
    dt_agent = make_dt_agent(dt_model)
    mcts_agent_fast = make_mcts_agent(iterations=50, rollout_strategy='random')
    mcts_agent_strong = make_mcts_agent(iterations=200, rollout_strategy='heuristic')
    
    agents = {
        'Decision Tree': dt_agent,
        'Random': random_agent,
        'MCTS (n=50)': mcts_agent_fast,
        'MCTS (n=200, heuristic)': mcts_agent_strong
    }

    # Define matchups
    matchups = [
        ('Decision Tree', 'Random', 50),
        ('Decision Tree', 'MCTS (n=50)', 20),
        ('Decision Tree', 'MCTS (n=200, heuristic)', 10),
    ]

    print("\n--- Starting Tournaments (Statistical Evaluation) ---")
    
    for agent1_name, agent2_name, n_games in matchups:
        print(f"\nTournament: {agent1_name} vs {agent2_name} ({n_games} games)")
        
        agent1 = agents[agent1_name]
        agent2 = agents[agent2_name]
        
        results = tournament(agent1, agent2, n_games=n_games, verbose=True)
        
        print(f"Results for {agent1_name}:")
        print(f"  Wins:  {results['wins_a']}")
        print(f"  Losses (Wins for {agent2_name}): {results['wins_b']}")
        print(f"  Draws: {results['draws']}")
        print(f"  Win Rate ({agent1_name}): {results['win_rate_a'] * 100:.1f}%")

if __name__ == '__main__':
    main()
