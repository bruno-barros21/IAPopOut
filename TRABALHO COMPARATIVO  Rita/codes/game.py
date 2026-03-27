from connected_four import ConnectState, GameMeta  # Representa o estado do jogo
from mcts import MCTS                   # Implementa o algoritmo MCTS
import pickle
from decision_tree_builder import predict  # Importa a função de previsão da árvore de decisão

# Exceções específicas para controlo de fluxo

class RestartGameException(Exception):
    pass

class QuitGameException(Exception):
    pass

with open("decision_tree.pkl", "rb") as f:
    decision_tree = pickle.load(f)


# Leitura de inteiro seguro
def safe_int_input(prompt):
    while True:
        try:
            value = input(prompt)
            if value.lower() == 'restart':
                raise RestartGameException
            if value.lower() == 'quit':
                raise QuitGameException
            return int(value)
        except ValueError:
            print("Entrada inválida. Por favor insira um número inteiro.")
        except EOFError:
            raise

# Leitura de jogada com reinício/saída
def safe_move_input(state, prompt="Escolha uma coluna (0-6), 'restart' ou 'quit': "):
    while True:
        try:
            user_input = input(prompt)
            if user_input.lower() == 'restart':
                raise RestartGameException
            if user_input.lower() == 'quit':
                raise QuitGameException
            move = int(user_input)
            if move in state.get_legal_moves():
                return move
            print("Movimento inválido. Tente novamente.")
        except ValueError:
            print("Entrada inválida. Insira um número de 0 a 6.")
        except EOFError:
            raise
def state_to_features(state):
    # Assumindo que o estado tem uma representação interna `board`
    # e que a árvore foi treinada com features nomeadas como pos_0, pos_1, ..., pos_41
    features = {}
    for i in range(6):  # 6 linhas
        for j in range(7):  # 7 colunas
            index = i * 7 + j
            features[f"s{index}"] = state.board[i][j]

    return features

# Funções de jogo com retorno de resultado

def play_player_vs_player(starting_player):
    state = ConnectState()
    state.to_play = starting_player
    while not state.game_over():
        state.print()
        print(f"Vez de {'X' if state.to_play==1 else 'O'}")
        move = safe_move_input(state)
        state.move(move)
    state.print()
    return state.get_outcome()

def play_player_vs_tree(player_piece, first, tree):
    state = ConnectState()
    tree_piece = 2 if player_piece == 1 else 1

    if first == 'tree':
        state.to_play = tree_piece
        state.print()
        features = state_to_features(state)
        mv = predict(tree, features)
        print(f"Árvore ({'X' if tree_piece==1 else 'O'}) escolheu: {mv}")
        state.move(mv)
    else:
        state.to_play = player_piece

    while not state.game_over():
        state.print()
        if state.to_play == player_piece:
            mv = safe_move_input(state)
        else:
            print("Árvore a pensar...")
            features = state_to_features(state)
            mv = predict(tree, features)
            print(f"Árvore ({'X' if tree_piece==1 else 'O'}) escolheu: {mv}")
        state.move(mv)

    state.print()
    return state.get_outcome()


def play_player_vs_ai(player_piece, first):
    state = ConnectState()
    ai_piece = 2 if player_piece == 1 else 1
    mcts = MCTS(state)
    if first == 'ai':
        state.to_play = ai_piece
        state.print()
        print("MCTS a começar...")
        mcts.search(10)
        mv = mcts.best_move()
        print(f"MCTS ({'X' if ai_piece==1 else 'O'}) escolheu: {mv}")
        state.move(mv)
        mcts.move(mv)
    else:
        state.to_play = player_piece
    while not state.game_over():
        state.print()
        if state.to_play == player_piece:
            mv = safe_move_input(state)
            state.move(mv)
            mcts.move(mv)
        else:
            print("MCTS a pensar...")
            mcts.search(5)
            mv = mcts.best_move()
            print(f"MCTS ({'X' if ai_piece==1 else 'O'}) escolheu: {mv}")
            state.move(mv)
            mcts.move(mv)
    state.print()
    return state.get_outcome()

def play_ai_vs_tree(starting_player, tree):
    state = ConnectState()
    mcts = MCTS(state)
    ai_piece = starting_player
    tree_piece = 2 if ai_piece == 1 else 1

    while not state.game_over():
        state.print()
        if state.to_play == ai_piece:
            print("MCTS a pensar...")
            mcts.search(5)
            mv = mcts.best_move()
            print(f"MCTS escolheu: {mv}")
        else:
            print("Árvore a pensar...")
            features = state_to_features(state)
            mv = predict(tree, features)
            print(f"Árvore escolheu: {mv}")
        state.move(mv)
        mcts.move(mv)

    state.print()
    return state.get_outcome()


def play_ai_vs_ai(starting_player):
    state = ConnectState()
    state.to_play = starting_player
    mcts1, mcts2 = MCTS(state), MCTS(state)
    while not state.game_over():
        state.print()
        # Indica qual IA está a jogar
        current_ai = 'X' if state.to_play == 1 else 'O'
        print(f"MCTS {current_ai} a pensar...")
        # Execução MCTS conforme a peça
        if state.to_play == 1:
            mcts1.search(5)
            mv = mcts1.best_move()
        else:
            mcts2.search(5)
            mv = mcts2.best_move()
        # Mostra a jogada escolhida
        print(f"MCTS {current_ai} escolheu a coluna: {mv}")
        # Aplica a jogada e atualiza ambas as árvores
        state.move(mv)
        mcts1.move(mv)
        mcts2.move(mv)
    state.print()
    return state.get_outcome()

# Menu pós-jogo: Play Again, Change Modes, Quit
def post_game_menu():
    print("O que deseja fazer a seguir?")
    print("1: Jogar novamente")
    print("2: Mudar modo")
    print("3: Sair")
    choice = safe_int_input("Escolha: ")
    if choice == 1:
        return 'again'
    if choice == 2:
        raise RestartGameException
    return 'quit'

# Bloco principal

def main():
    x_wins = o_wins = draws = 0
    try:
        while True:
            print("1: Jogador vs Jogador\n2: Jogador vs MCTS\n3: Jogador vs Árvore\n4: MCTS vs Árvore")
            mode = safe_int_input("Modo: ")
            # Definições iniciais do modo
            if mode == 1:
                print("Quem começa? \n1: X  \n2: O")
                first = 1 if safe_int_input("Escolha: ")==1 else 2
                while True:
                    outcome = play_player_vs_player(first)
                    if outcome == 1: x_wins += 1
                    elif outcome == 2: o_wins += 1
                    else: draws += 1
                    print(f"Placar: (X) {x_wins} - {o_wins} (O) (Empates: {draws})")
                    action = post_game_menu()
                    if action == 'again':
                        continue
                    elif action == 'quit':
                        raise QuitGameException
            elif mode == 2:
                print("Quem começa? \n1: Jogador  \n2: MCTS")
                first = 'player' if safe_int_input("Escolha: ") == 1 else 'ai'
                print("Que peça prefere? \n1: X  \n2: O")
                player_piece = 1 if safe_int_input("Escolha: ") == 1 else 2
                mcts_piece = 3 - player_piece

                while True:
                    outcome = play_player_vs_ai(player_piece, first)
                    # se outcome == player_piece → jogador humano ganhou
                    if outcome == player_piece:
                        if player_piece == 1:
                            x_wins += 1
                        else:
                            o_wins += 1
                    elif outcome == mcts_piece:
                        # MCTS ganhou
                        if mcts_piece == 1:
                            x_wins += 1
                        else:
                            o_wins += 1
                    else:
                        draws += 1

                    print(f"Placar: (X) {x_wins} - {o_wins} (O) (Empates: {draws})")
                    action = post_game_menu()
                    if action == 'again':
                        continue
                    elif action == 'quit':
                        raise QuitGameException

            elif mode == 3:
                print("Quem começa? \n1: Jogador  \n2: Árvore")
                first = 'player' if safe_int_input("Escolha: ") == 1 else 'tree'
                print("Que peça prefere? \n1: X  \n2: O")
                player_piece = 1 if safe_int_input("Escolha: ") == 1 else 2
                tree_piece = 3 - player_piece

                while True:
                    outcome = play_player_vs_tree(player_piece, first, decision_tree)
                    if outcome == player_piece:
                        # Jogador humano ganhou
                        if player_piece == 1:
                            x_wins += 1
                        else:
                            o_wins += 1
                    elif outcome == tree_piece:
                        # Árvore ganhou
                        if tree_piece == 1:
                            x_wins += 1
                        else:
                            o_wins += 1
                    else:
                        draws += 1

                    print(f"Placar: (X) {x_wins} - {o_wins} (O) (Empates: {draws})")
                    action = post_game_menu()
                    if action == 'again':
                        continue
                    elif action == 'quit':
                        raise QuitGameException

                    
            elif mode == 4:
                print("Quem começa? \n1: MCTS  \n2: Árvore")
                mcts_piece = 1 if safe_int_input("Escolha: ") == 1 else 2
                tree_piece = 3 - mcts_piece

                while True:
                    outcome = play_ai_vs_tree(mcts_piece, decision_tree)
                    if outcome == mcts_piece:
                        # MCTS ganhou
                        if mcts_piece == 1:
                            x_wins += 1
                        else:
                            o_wins += 1
                    elif outcome == tree_piece:
                        # Árvore ganhou
                        if tree_piece == 1:
                            x_wins += 1
                        else:
                            o_wins += 1
                    else:
                        draws += 1

                    print(f"Placar: (X) {x_wins} - {o_wins} (O) (Empates: {draws})")
                    action = post_game_menu()
                    if action == 'again':
                        continue
                    elif action == 'quit':
                        raise QuitGameException



            else:
                print("Modo inválido.")
    except RestartGameException:
        print("Mudando modo...")
        main()
    except QuitGameException:
        print("Programa encerrado.")
    except EOFError:
        print("Entrada finalizada.")

if __name__ == "__main__":
    main()
# Este código implementa um jogo de Conecta 4 com várias opções de modo de jogo.