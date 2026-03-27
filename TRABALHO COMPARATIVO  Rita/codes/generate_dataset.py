import random
import numpy as np
from mcts import MCTS
from connected_four import ConnectState, GameMeta, MCTSMeta

def generate_mcts_dataset(num_games=1000):
    dataset = []

    for game_idx in range(num_games):
        print(f"\n🎮 Jogo {game_idx + 1} de {num_games}")
        state = ConnectState()
        mcts = MCTS(state)
        total_moves = 0  # contador de jogadas
        mcts.search(time_limit=3)  # mais tempo por jogada

        while not state.game_over():
            legal_moves = state.get_legal_moves()
            move = mcts.best_move()

            if move not in legal_moves:
                print(f"[!] Movimento ilegal sugerido: {move}")
                move = random.choice(legal_moves)

            try:
                board_flat = [cell for row in state.get_board() for cell in row]
                dataset.append((board_flat, move))

                state.move(move)
                mcts.move(move)
                total_moves += 1

                if not state.game_over():
                    time_limit = 3 if total_moves < 10 else 1.5
                    mcts.search(time_limit=time_limit)

            except ValueError as e:
                print(f"[Erro] Movimento inválido: {e}")
                break

        print(f"✔️ Jogo {game_idx + 1} concluído — total de jogadas salvas: {len(dataset)}")

        # Salvamento parcial
        if (game_idx + 1) % 10 == 0:
            save_dataset(dataset, filename='mcts_dataset_parcial.csv')
            print(f"💾 Dataset parcial salvo com {len(dataset)} jogadas.")

    return dataset

def save_dataset(dataset, filename='mcts_dataset.csv'):
    with open(filename, 'w') as f:
        f.write(','.join([f's{i}' for i in range(42)]) + ',move\n')
        for state, move in dataset:
            f.write(','.join(map(str, state)) + f',{move}\n')

# Geração e salvamento
if __name__ == "__main__":
    dataset = generate_mcts_dataset(num_games=1000)
    save_dataset(dataset)
    print(f"\n✅ Dataset final salvo com {len(dataset)} exemplos!")
