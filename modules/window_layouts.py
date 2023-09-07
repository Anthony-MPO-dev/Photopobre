import tkinter as tk
import PySimpleGUI as sg
import matplotlib.pyplot as plt

"""

    Esse Modulo é Responsavel pelo Layout das Janelas

    
"""

# Carrega janela Principal
def main_window():

    # Obter as dimensões da tela
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    # Calcular o tamanho da janela proporcional a 50% em todas as dimensões
    window_width = int(screen_width * 0.5)
    window_height = int(screen_height * 0.5)

    # Calcular as coordenadas para centralizar a janela
    x_pos = (screen_width - window_width) // 2
    y_pos = (screen_height - window_height) // 2

    sg.theme('DarkAmber')

    # Defina o layout da janela
    layout = [
        [sg.Menu([['Arquivo', ['Carregar Imagem', 'Desfazer', 'Salvar', 'Salvar como', 'Confirmar alterações', 'Reset imagem']], ['Transformações de Intensidade', ['Negativa', 'Monocromática','Correções de Contrastes', ['Logarítimica', 'Exponencial', 'GAMMA', 'Limiarizar'], 'Histograma', ['Equalização de hist.', 'Especificação de hist.', 'Hist. Monocromatico', 'Hist. Mult. Canais RGB'], 'Ajustar Brilho']],['Reamostragem', ['Zoom +', 'Zoom -', 'Ferramentas', ['Ajustar Zoom']]], ['Filtros Espaciais', ['Filtros Passa-Baixa',['Filtro Box', 'Filtro Gaussiano', 'Filtro da Mediana'], 'Filtros Passa-Alta', ['Aguçamento Laplaciano', 'Aguçamento de Sobel']]]])],
        [sg.Image(key='-IMAGE-')],  # Local onde a imagem será exibida
    ]

    # Crie a janela
    return sg.Window('Photopobre 1.0', layout, size=(window_width, window_height), location=(x_pos, y_pos), resizable=True)




# Função para criar a janela da barra de rolagem de zoom
def create_zoom_slider_window():
    layout = [
        [sg.Text('Ajustar Zoom ')],
        [sg.Slider(range=(-50, 50), default_value=0, orientation='h', key='-ZOOM-')],
        [sg.Button('Aplicar')],
    ]
    return sg.Window('Ajustar Zoom', layout, finalize=True)  # Adicione finalize=True para iniciar o loop de eventos

# Função para criar a janela da barra de rolagem de brilho
def create_brilho_slider_window():
    layout = [
        [sg.Text('Escolha o Fator de Intensidade do Brilho')],
        [sg.Slider(range=(0.1, 1000), default_value=0, orientation='vertical', key='-BRILHO-')],
        [sg.Button('Aplicar')],
    ]
    return sg.Window('Ajustar Brilho', layout, finalize=True)  # Adicione finalize=True para iniciar o loop de eventos

# Função para criar a janela do input da transformação GAMMA
def create_gamma_input_window():
    layout = [
        [sg.Text('Digite o Valor do Parâmetro de Gama:')],
        [sg.InputText(key='-GAMMA-')],
        [sg.Button('Aplicar')],
    ]
    return sg.Window('Ajustar Parâmetro de Gama', layout, finalize=True)

def create_limiarizacao_input_window():
    layout = [
        [sg.Text('Digite o Valor do Parâmetro do Thresholding da Limiarização:')],
        [sg.InputText(key='-Thresholding-')],
        [sg.Button('Aplicar')],
    ]
    return sg.Window('Ajustar Parâmetro do Thresholding', layout, finalize=True)

# Função para criar a janela do histograma
def create_hist_monocromatico_window(hist, intensity_values):
    # Crie um gráfico do histograma
    plt.figure(figsize=(6, 4))
    plt.bar(intensity_values, hist.ravel(), width=1, color='gray')
    plt.title('Histograma de Intensidade')
    plt.xlabel('Intensidade')
    plt.ylabel('Frequência')

    # Salve o gráfico em um arquivo temporário
    plt.savefig('temp_hist.png', bbox_inches='tight')
    plt.close()

    # Exiba o gráfico na janela PySimpleGUI
    layout = [
        [sg.Text('Histograma de imagem Monocromática:')],
        [sg.Image(filename='temp_hist.png')],
    ]
    return sg.Window('Histograma de Intensidade', layout, finalize=True)


def create_hist_rgb_window(intensity_values, hist_r, hist_g, hist_b, combined_hist):

    # Crie um gráfico para cada canal
    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    plt.bar(intensity_values, hist_r.ravel(), width=1, color='red')
    plt.title('Histograma de R')
    plt.xlabel('Intensidade')
    plt.ylabel('Frequência')

    plt.subplot(132)
    plt.bar(intensity_values, hist_g.ravel(), width=1, color='green')
    plt.title('Histograma de G')
    plt.xlabel('Intensidade')
    plt.ylabel('Frequência')

    plt.subplot(133)
    plt.bar(intensity_values, hist_b.ravel(), width=1, color='blue')
    plt.title('Histograma de B')
    plt.xlabel('Intensidade')
    plt.ylabel('Frequência')

    # Salve o gráfico em um arquivo temporário
    plt.savefig('temp_hist_rgb.png', bbox_inches='tight')

    # Crie um gráfico de histograma usando Matplotlib
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(combined_hist)
    ax.set_title('Histograma RGB da Imagem')
    ax.set_xlabel('Valor de Pixel')
    ax.set_ylabel('Frequência')
    ax.set_xlim(0, 255)
    ax.legend(['B', 'R', 'G'])

    # Salve a figura em um arquivo temporário
    plt.savefig('temp_combined_rgb_hist.png', bbox_inches='tight')

    plt.close()

    # Exiba o gráfico na janela PySimpleGUI
    layout = [
        [sg.Text('Histogramas dos Canais de Cores da imagem RGB:')],
        [sg.Image(filename='temp_hist_rgb.png')],
        [sg.Image(filename='temp_combined_rgb_hist.png')],
    ]

    return sg.Window('Histogramas de Cores', layout, finalize=True)

# Função para criar a janela do gistograma de especificacao
def create_compare_window(button_txt):
    layout = [
        [sg.Button(button_txt)],
        [sg.Text(key='-TXT-IMG-')],
        [sg.Image(key='-ORI-IMAGE-')],
        [sg.Text(key='-TXT-IMG-REF-')],
        [sg.Image(key='-REF-IMAGE-')],

    ]
    return sg.Window('Gerar Transformacao', layout, finalize=True)

# Função para criar janela de definicao do filtro box
def create_filtro_window(title, text):
    layout = [
        [sg.Text(text)],
        [sg.InputText(key='-INPUT-')],
        [sg.Button('Aplicar'), sg.Button('Cancelar')]
    ]

    return sg.Window(title, layout, finalize=True)


