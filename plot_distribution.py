import os
import matplotlib.pyplot as plt

FIGURES_DIR = os.path.join('outputs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

def main():
    labels = ['Drops (83%)', 'Pops (17%)', 'Draws (0%)']
    sizes = [21612, 4442, 0]
    colors = ['#2ca02c', '#ff7f0e', '#d62728']
    explode = (0.1, 0, 0)  # destacar o Drop
    
    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                      autopct='%1.1f%%', shadow=True, startangle=140,
                                      textprops=dict(color="w", weight="bold", fontsize=12))
    
    # Customizar as labels externas
    for text in texts:
        text.set_color('black')
        text.set_fontsize(11)
        
    ax.set_title('Distribuição de Jogadas no Dataset (26.054 amostras)', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, 'dataset_distribution_pie.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Gráfico guardado em: {plot_path}")

if __name__ == '__main__':
    main()
