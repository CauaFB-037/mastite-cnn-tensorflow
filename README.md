# 📄 Classificação de Mastite Bovina com CNN

Este projeto utiliza redes neurais convolucionais (CNN) para detectar automaticamente casos de **mastite bovina** por meio da análise de imagens.  
O modelo foi construído com TensorFlow/Keras e treina em imagens organizadas em duas pastas: `treino/` e `validacao/`.

---

## 📂 Estrutura do Projeto

Projeto_Mastite/

│

├── Data/ # Dados de entrada

│ ├── treino/ # Imagens para treino

│ │ ├── saudavel/ # Classe 1

│ │ └── doente/ # Classe 2

│ └── validacao/ # Imagens para validação

│ ├── saudavel/ # Classe 1

│ └── doente/ # Classe 2

│

├── treino_rede.py # Script de treinamento da CNN

├── modelo.h5 # Modelo treinado salvo

---

## 🚀 Como executar

### 1. Instale os pacotes necessários

Recomenda-se Python 3.8+ e os seguintes pacotes:

```bash
pip install tensorflow matplotlib
```

2. Execute o script
Certifique-se de que os caminhos das pastas treino e validacao estão corretos no código, e então execute:
```
python treino_rede.py
```
---

## 🧠 Sobre o modelo
A CNN utilizada possui a seguinte arquitetura:

* Camadas de convolução (Conv2D) com ativação ReLU

* Camadas de pooling (MaxPooling2D)

* Camada Flatten e Dropout

* Camadas densas (Dense)

* Saída com sigmoid para classificação binária

Técnicas aplicadas:
* Normalização de imagens (Rescaling)

* Aumento de dados (Data Augmentation)

* Pré-processamento com cache() e prefetch()

* Função de perda: binary_crossentropy

* Otimizador: adam

* Métrica: accuracy

---

## 📊 Resultados
Ao final do treinamento, um gráfico de acurácia é exibido mostrando o desempenho em cada época.
O modelo final é salvo como modelo.h5.

---

## 📝 Observações
As imagens devem estar organizadas em subpastas com o nome da classe (ex: saudavel/, doente/)

O número de épocas e outras configurações podem ser ajustados no código.

---

## 📋 Licença
Este projeto é livre para uso educacional e está sob a Licença MIT.
