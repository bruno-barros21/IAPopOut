import numpy as np

class PopOutBoard:
    def __init__(self):
        self.ROWS = 6
        self.COLS = 7
        # 0 = Vazio, 1 = Jogador 1, 2 = Jogador 2
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.current_player = 1
        
        # Histórico para gerir a Regra 3 (Empate por repetição)
        self.state_history = {} 
        self._record_state()
        
        self.is_game_over = False
        self.winner = None  # None = jogo a decorrer, 0 = Empate, 1 = Jogador 1, 2 = Jogador 2

    def copy(self):
        """Cria uma cópia rápida do estado atual (essencial para o MCTS)."""
        new_board = PopOutBoard()
        new_board.board = np.copy(self.board)
        new_board.current_player = self.current_player
        new_board.state_history = self.state_history.copy()
        new_board.is_game_over = self.is_game_over
        new_board.winner = self.winner
        return new_board

    def _record_state(self):
        """Guarda o estado atual no histórico para verificar repetições."""
        # Convertemos a matriz para bytes para poder usar como chave no dicionário
        state_bytes = self.board.tobytes()
        self.state_history[state_bytes] = self.state_history.get(state_bytes, 0) + 1

    def get_legal_moves(self):
        """Retorna uma lista de jogadas possíveis: [('drop', col), ('pop', col), ('draw', None)]"""
        if self.is_game_over:
            return []

        moves = []
        is_board_full = True

        for col in range(self.COLS):
            # Verifica Drops (se a linha do topo, índice 0, estiver vazia)
            if self.board[0][col] == 0:
                moves.append(('drop', col))
                is_board_full = False
            
            # Verifica Pops (se a linha do fundo, índice 5, tem uma peça do jogador atual)
            if self.board[self.ROWS - 1][col] == self.current_player:
                moves.append(('pop', col))

        # REGRA 2: Se o tabuleiro estiver cheio, o jogador pode declarar empate ou fazer pop
        if is_board_full:
            moves.append(('draw', None))

        # REGRA 3: Se o estado se repetiu 3 vezes, o jogador pode declarar empate
        state_bytes = self.board.tobytes()
        if self.state_history.get(state_bytes, 0) >= 3:
            if ('draw', None) not in moves:
                moves.append(('draw', None))

        return moves

    def apply_move(self, move):
        """Aplica a jogada, atualiza o tabuleiro e verifica vitórias."""
        move_type, col = move

        if move_type == 'draw':
            self.is_game_over = True
            self.winner = 0
            return

        if move_type == 'drop':
            # Encontra a linha mais baixa vazia
            for row in range(self.ROWS - 1, -1, -1):
                if self.board[row][col] == 0:
                    self.board[row][col] = self.current_player
                    break

        elif move_type == 'pop':
            # Move todas as peças dessa coluna um espaço para baixo
            self.board[1:self.ROWS, col] = self.board[0:self.ROWS - 1, col]
            # O espaço do topo fica vazio
            self.board[0][col] = 0

        # Verifica vitória ANTES de mudar de jogador para saber quem fez o movimento
        self._check_winner(move_type)

        if not self.is_game_over:
            # Troca o jogador
            self.current_player = 3 - self.current_player  # Matemática simples: 3-1=2, 3-2=1
            self._record_state()

    def _check_winner(self, move_type):
        """Verifica se há 4 em linha."""
        p1_wins = self._has_four_in_a_row(1)
        p2_wins = self._has_four_in_a_row(2)

        if p1_wins and p2_wins:
            # REGRA 1: Vitórias simultâneas através de Pop
            # O vencedor é o jogador que acabou de fazer a jogada
            self.winner = self.current_player
            self.is_game_over = True
        elif p1_wins:
            self.winner = 1
            self.is_game_over = True
        elif p2_wins:
            self.winner = 2
            self.is_game_over = True

    def _has_four_in_a_row(self, player):
        """Lógica de verificação de 4 em linha (horizontal, vertical, diagonal)."""
        # Verifica Horizontal
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                if self.board[r][c] == player and self.board[r][c+1] == player and \
                   self.board[r][c+2] == player and self.board[r][c+3] == player:
                    return True

        # Verifica Vertical
        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                if self.board[r][c] == player and self.board[r+1][c] == player and \
                   self.board[r+2][c] == player and self.board[r+3][c] == player:
                    return True

        # Verifica Diagonal (Positiva)
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                if self.board[r][c] == player and self.board[r+1][c+1] == player and \
                   self.board[r+2][c+2] == player and self.board[r+3][c+3] == player:
                    return True

        # Verifica Diagonal (Negativa)
        for r in range(3, self.ROWS):
            for c in range(self.COLS - 3):
                if self.board[r][c] == player and self.board[r-1][c+1] == player and \
                   self.board[r-2][c+2] == player and self.board[r-3][c+3] == player:
                    return True

        return False