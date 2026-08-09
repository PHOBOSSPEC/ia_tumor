#  Reconhecimento de Tumores Cerebrais

Projeto de Inteligência Artificial para reconhecimento e classificação de tumores cerebrais a partir de imagens de ressonância magnética (RM) e tomografia computadorizada (TC).

O projeto contém o código utilizado para treinamento dos modelos de Deep Learning, realizando o treinamento e salvando os melhores modelos obtidos durante o processo.

O treinamento é realizado utilizando PyTorch e a arquitetura MobileNetV2 pré-treinada.

##  Estrutura do Dataset

O dataset deve seguir a seguinte estrutura:

```text
dataset/
├── train/
│   ├── normal_RM/
│   ├── normal_TC/
│   ├── tumor_RM/
│   └── tumor_TC/
│
└── val/
    ├── normal_RM/
    ├── normal_TC/
    ├── tumor_RM/
    └── tumor_TC/
