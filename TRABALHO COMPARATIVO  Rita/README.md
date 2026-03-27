# 🎯 Connected Four

Uma implementação do clássico jogo **Connect Four** (Quatro em Linha), com agentes de **IA** baseados em:

- **Monte Carlo Tree Search (MCTS)**  
- **Árvore de Decisão (ID3)**  

Desenvolvido no contexto académico para avaliar e comparar desempenho dessas duas abordagens em jogos de tabuleiro.

---

## 📚 Conteúdo do Repositório

```text
ConnectedFour/
├── Enunciado/            ← Enunciado ou especificação do trabalho (PDF ou Markdown)
    |── enunciado_IA.pdf
├── codes/                    ← Código-fonte e notebooks
|   ├── gráficos/             ← gráficos auxiliares para a análise de resultados (notebook)
|        ├── acuraciaxprofundidade.png ← Relação entre acurácia e profundidade do ID3
|        ├── numeroJogadas.png ← gráfico com análise do número de jogadas em cada coluna usando o mcts_dataset.csv
|        └── impactoAlpha.png      ← Impacto do alpha na taxa de vitória e tempo médio de decisão do MCTS
|   ├── connected_four.py     ← Regras do jogo e representação do tabuleiro
│   ├── ConnectFour.ipynb     ← Notebook principal (interface, visualizações, testes)
│   ├── game.py               ← Jogo final com menus auxiliares, exceções para erros e uso de MCTS e Árvore
│   ├── mcts.py               ← Implementação do Monte Carlo Tree Search
|   ├── mcts_dataset.csv      ← dataset gerado por 1000 jogos com MCTS
│   ├── decision_tree_builder.py  ← Implementação do algoritmo ID3
|   ├── decision_tree.pkl     ← árvore de decisão treinada pelo dataset mcts_dataset.csv
│   ├── generate_dataset.py   ← Gerador de dataset para treinar a árvore de decisão
|   ├── iris.csv              ← dataset de treino
|   ├── iris_tree.pkl         ← árvore de decisão treinada pelo dataset iris.csv
│   └── iris_test.py          ← Treino auxiliar de árvores de decisão usando dataset iris.csv
├── .gitignore
└── README.md                 ← Este ficheiro
```

## 📊 Funcionalidades 

__Agente MCTS__
- Escolha de jogadas via simulações
- Parâmetros configuráveis: número de simulações, função de seleção, exploração vs exploração
  
__Agente ID3__
- Treino de árvore de decisão com dados de partidas
- Visualização da árvore e regras extraídas
  
__Comparação de desempenho__ 
- Taxa de vitória
- Tempo de decisão
- Qualidade das jogadas (heurísticas)

## 📈 Trabalhos Futuros

- Integrar visualização gráfica em tempo real (por exemplo, pygame ou tkinter)
- Experimentar algoritmos adversariais avançados (Minimax com α-β)
- Ajustar hiperparâmetros do MCTS com técnicas de otimização automática
- Ampliar base de dados de treino para ID3
  
## 🧑‍💻 Autoria

Desenvolvido por Rita Moreira, Pedro Gilvaia e Gonçalo Correia

## 📝 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).


