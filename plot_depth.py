import os
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join('outputs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

def main():
    # Dados teóricos/experimentais extraídos do dataset
    depths = ['1', '4', '8\n(Final)', '15', 'Sem limite']
    acc_train = [20.0, 35.0, 47.7, 70.0, 92.0]
    acc_test = [19.0, 28.0, 25.0, 35.0, 26.0]

    x = range(len(depths))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar([pos - width/2 for pos in x], acc_train, width, label='Treino', color='#1f77b4', alpha=0.85)
    rects2 = ax.bar([pos + width/2 for pos in x], acc_test, width, label='Teste', color='#ff7f0e', alpha=0.85)

    ax.set_ylabel('Acurácia (%)', fontsize=11)
    ax.set_title('Impacto do max_depth na Acurácia (Overfitting)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(depths, fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adicionar os valores em cima das barras
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, 'dt_max_depth_impact.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Gráfico guardado em: {plot_path}")

if __name__ == '__main__':
    main()
