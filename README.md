# PhotoPobre 2.0
O Photoshop de pobre | programa criado para processamento de imagens simples 





## Instalação

Tenha python instalado:

### Windows:

#### 1. Vá para o site oficial do Python em https://www.python.org/downloads/.
#### 2. Role para baixo até encontrar as versões disponíveis do Python.
#### 3. Escolha a versão mais recente do Python 3 (por exemplo, Python 3.9) que seja apropriada para o seu sistema operacional (32 bits ou 64 bits). Recomenda-se usar a versão mais recente.
#### 4. Clique no link de download para essa versão.
#### 5. Na parte inferior da janela do navegador, você deve ver um arquivo de instalação sendo baixado (por exemplo, python-3.9.6.exe). Aguarde até que o download seja concluído.
#### 6. Execute o arquivo de instalação que você baixou.
#### 7. Certifique-se de marcar a opção "Add Python X.Y to PATH" (substitua X.Y pela versão do Python que você está instalando) durante a instalação para facilitar o uso do Python a partir da linha de comando.
#### 8. Clique em "Install Now" para começar a instalação.
#### 9. A instalação será concluída em breve. Você pode fechar o instalador após a conclusão.

### Linux:

```bash
  sudo apt install python3
```

Verificar se foi instalado
```bash
  python3 --version
```

### MacOs:

Instale o homebrew:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Agora instale o python:
```bash
  brew install python@3
```

Verificar se foi instalado
```bash
  python3 --version
```


Instale as seguintes bibliotecas se não tiver:

```bash
  pip install PySimpleGUI
```
```bash
  pip install numpy
```
```bash
  pip install matplotlib
```
```bash
  pip install opencv-python
```
```bash
  pip install tk
```




## Funcionalidades

- Temas DarkAmber do PySimpleGui
- Realiza
    - Transformações de intensidade pontuais:
        * Negativos.
        * Aumento uniforme de brilho.
        * Correções de contraste: logarítimica, gamma e linear por partes.
    - Histogramas.
        - Histogramas de canal monocromático e RGB
        - Correção de contraste por equalização de histogramas.
        - Especificação de histogramas.
    - Reamostragem por interpolação bilinear
    - Filtros espaciais de suavização: 
        - Filtro Box, Gaussiano e Mediana (Passa Baixas) 
            - Suavização de imagens e ruidos 
            - Tratamento de ruido sal pimenta com filtro mediana
        - Filtro Laplaciano e Sobel (Passa Altas) -> Operações de Aguçamento da imagem




