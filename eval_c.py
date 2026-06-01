import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from src.game.popout_board import PopOutBoard
from src.ai.agents import make_mcts_agent

FIGURES_DIR = os.path.join('outputs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

def play_match(agent1, agent2):
    board = PopOutBoard()
    agents = {1: agent1, 2: agent2}
    while not board.is_game_over:
        move = agents[board.current_player](board)
        board.apply_move(move)
    return board.winner

def run_tournament(c_values, num_games=10, iterations=200):
    results = []
    
    agent2 = make_mcts_agent(iterations=iterations, c=math.sqrt(2), rollout_strategy='random')
    
    for c in c_values:
        print(f"Testing C={c:.3f} vs C={math.sqrt(2):.3f}...")
        agent1 = make_mcts_agent(iterations=iterations, c=c, rollout_strategy='random')
        
        wins = 0
        draws = 0
        t0 = time.time()
        for i in range(num_games):
            # Half games player 1, half games player 2
            if i % 2 == 0:
                winner = play_match(agent1, agent2)
                if winner == 1: wins += 1
                elif winner == 0: draws += 1
            else:
                winner = play_match(agent2, agent1)
                if winner == 2: wins += 1
                elif winner == 0: draws += 1
                
        elapsed = time.time() - t0
        win_rate = wins / num_games
        print(f"  Win rate: {win_rate:.2f} (Time: {elapsed:.1f}s)")
        results.append(win_rate)
        
    return results

def main():
    c_values = [0.1, 0.5, 1.414, 2.5, 5.0, 10.0]
    num_games = 20  # Fast tournament for quick plot
    iterations = 300
    
    print(f"Starting tournament: MCTS(C) vs MCTS(C=1.414)")
    print(f"Games per C: {num_games}, Iterations: {iterations}")
    
    win_rates = run_tournament(c_values, num_games=num_games, iterations=iterations)
    
    # Plotting
    plt.figure(figsize=(8, 5))
    x_labels = [f"{c:.1f}" if c != 1.414 else "√2" for c in c_values]
    
    colors = ['#2ca02c' if c == 1.414 else '#1f77b4' for c in c_values]
    bars = plt.bar(x_labels, win_rates, color=colors, alpha=0.85)
    
    plt.axhline(0.5, linestyle='--', color='red', alpha=0.5, label='Empate Teórico (50%)')
    
    plt.title(f'Desempenho MCTS(C) vs MCTS(C=√2)\n({num_games} jogos cada, {iterations} iterações/jogada)', fontsize=12, fontweight='bold')
    plt.xlabel('Valor da Constante de Exploração (C)', fontsize=10)
    plt.ylabel('Win Rate (vs MCTS √2)', fontsize=10)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar, val in zip(bars, win_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha='center', fontsize=10, fontweight='bold')
                 
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(FIGURES_DIR, 'mcts_c_comparison.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nGráfico guardado em: {plot_path}")

if __name__ == '__main__':
    main()
