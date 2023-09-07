from modules.window_layouts import *
import numpy as np
import cv2
import os 

global zoom_factor, brilho_factor, img_scaled, negativa

zoom_slider_window = None
brilho_slider_window = None
gamma_input_window = None
thresholding_input_window = None
hist_monocromatico_window = None
window_hist_especificacao = None
window_filtro_box = None
window_gauss = None
window_mediana = None
window_sobel = None
hist_rgb_window = None
window_laplaciana = None
zoom_factor = 1.0
brilho_factor = 1.0
negativa = False
img_scaled = None



"""

    Funções dos eventos

"""


def carregar_img(image_path):
    global zoom_factor, img_scaled, original_img

    if image_path:
        print(f'Loading image: {image_path}')
        img = cv2.imread(image_path)
        original_img = img
        
        img_scaled = cv2.resize(img, (int(img.shape[1] * zoom_factor), int(img.shape[0] * zoom_factor)))       
        img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

        return img_bytes, img, original_img

def zoom_plus(img):
    global zoom_factor, img_scaled
    
    if zoom_factor < 3.1:
        zoom_factor += zoom_factor*0.1
        img_scaled = cv2.resize(img, (int(img.shape[1] * zoom_factor), int(img.shape[0] * zoom_factor)), interpolation=cv2.INTER_LINEAR)
        img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()
    
    else:
        img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

    return img_bytes

def zoom_minus(img):
    global zoom_factor, img_scaled

    if zoom_factor <= 0.1:
        zoom_factor = 0.1
    else:
        zoom_factor -= zoom_factor*0.1
    

    img_scaled = cv2.resize(img, (int(img.shape[1] * zoom_factor), int(img.shape[0] * zoom_factor)), interpolation=cv2.INTER_LINEAR)
    img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

    return img_bytes

def Negativa(img):
    global negativa, img_scaled
    # Verificar se a imagem é negativa
    if negativa:
        img_scaled = 255 - img_scaled
    else:
        img_scaled = 255 - img
        
    negativa = True

    img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

    return img_bytes

def monocromatica(img):
    global img_scaled
    
    if len(img.shape) <= 2:

        # A conversão falhou, assuma que a imagem já está em escala de cinza
        print("A imagem já está em escala de cinza.")
        try:
            raise ValueError("A imagem já é Monocromática.")
        except ValueError as e:
                sg.popup_error(str(e))

        img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

        return img_bytes
    
    else:
        # Transformar a imagem em escala de cinza
        img_scaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

        return img_bytes
    

    

def equalizar_hist(img):
    global img_scaled

    if len(img.shape) <= 2:
        img_scaled = cv2.equalizeHist(img)

        img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

    else:
        # Separar os canais de cor
        b, g, r = cv2.split(img)
        
        # Aplicar equalização de histograma em cada canal
        b_equalized = cv2.equalizeHist(b)
        g_equalized = cv2.equalizeHist(g)
        r_equalized = cv2.equalizeHist(r)
        
        # Juntar os canais novamente
        img_scaled = cv2.merge((b_equalized, g_equalized, r_equalized))
        img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

        

    return img_bytes

def ajuste_brilho(img):
    global brilho_slider_window, brilho_factor, img_scaled
    if brilho_slider_window is None:      
        print('Janela de brilho ativa!')
        brilho_slider_window = create_brilho_slider_window()
        while True:  # Adicione este loop de eventos para a janela de ajuste de brilho
            event, values = brilho_slider_window.read()

            if event == sg.WIN_CLOSED:
                brilho_slider_window.close()
                brilho_slider_window = 1.0
                brilho_slider_window = None
                break
            elif event == 'Aplicar':
                if brilho_slider_window:
                    print("Brilho aplicado!")
                    brilho_factor = values['-BRILHO-']*0.1

                    brilho_slider_window.close()
                    brilho_slider_window = None
                    break
        
        img_scaled = np.clip(img * brilho_factor, 0, 255).astype(np.uint8)
        img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

        return img_bytes

def ajuste_zoom(img):
    global img_scaled, zoom_factor, zoom_slider_window

    if zoom_slider_window is None:      
        print('Janela de zoom ativa!')
        zoom_slider_window = create_zoom_slider_window()
        while True:  # Adicione este loop de eventos para a janela de ajuste de zoom
            event, values = zoom_slider_window.read()
            if event == sg.WIN_CLOSED:
                zoom_slider_window.close()
                zoom_factor = 1.0
                zoom_slider_window = None
                break
            elif event == 'Aplicar':
                if zoom_slider_window:
                    print("Aplicação de zoom!")
                    zoom_factor += values['-ZOOM-']*0.1

                    if zoom_factor <= 0:
                        zoom_factor = 0.1
                    if zoom_factor > 3.1:
                        zoom_factor = 3
                    zoom_slider_window.close()
                    zoom_slider_window = None
                    break
        
        img_scaled = cv2.resize(img, (int(img.shape[1] * zoom_factor), int(img.shape[0] * zoom_factor)), interpolation=cv2.INTER_LINEAR)
        img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

        return img_bytes
    
def salvar(image_path):
    global img_scaled
    cv2.imwrite(image_path, img_scaled)
    print('Imagem salva!')
    
def salvar_como(image_path):
    global img_scaled
    # Extrair a extensão do arquivo original
    _, ext = os.path.splitext(image_path)
    save_path = sg.popup_get_file('Salvar Imagem Como', save_as=True, default_extension=ext)
    if save_path:
        cv2.imwrite(save_path, img_scaled)
        print(f'Imagem salva em: {save_path}')

def reset_image():
    global img_scaled, zoom_factor, original_img

    zoom_factor = 1.0
    img_scaled = cv2.resize(original_img, (int(original_img.shape[1]), int(original_img.shape[0])), interpolation=cv2.INTER_LINEAR)
    img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

    return img_scaled, img_bytes

def desfazer_alteracao(img):

    global zoom_factor, brilho_factor, negativa

    zoom_factor = 1.0
    brilho_factor = 1.0
    negativa = False

    img_bytes = cv2.imencode('.png', img)[1].tobytes()

    return img_bytes

def confirmar_alteracoes():

    global zoom_factor, brilho_factor, negativa, img_scaled

    zoom_factor = 1.0
    brilho_factor = 1.0
    negativa = False

    return img_scaled

def log_transf(img):
    global img_scaled

    c = 255.0 / np.log(1 + 255)

    # Separar os canais de cor da imagem
    b, g, r = cv2.split(img)

    # Aplicar a transformação logarítmica a cada canal
    b_transformed = c * np.log(b.astype(np.float64) + 1)
    g_transformed = c * np.log(g.astype(np.float64) + 1)
    r_transformed = c * np.log(r.astype(np.float64) + 1)

    # Juntar os canais novamente
    transformed_image = cv2.merge((b_transformed, g_transformed, r_transformed))

    img_scaled = transformed_image.astype(np.uint8)

    img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

    return img_bytes

def exp_transf(img):
    global img_scaled

    # Constante de normalização para imagens de 8 bits
    c = 255 / np.log(256)

    # Transformação exponencial para cada canal de cor
    transformed_channels = []

    for channel in cv2.split(img):
        exp_transformed = np.exp(channel / c)
        transformed_channels.append(exp_transformed)
    
    # Juntar os canais transformados
    transformed_image = cv2.merge(transformed_channels)
    
    # Normalizar os valores para o intervalo [0, 255]
    transformed_image = (transformed_image / np.max(transformed_image)) * 255
    
    # Converter para o tipo de dados uint8
    img_scaled = transformed_image.astype(np.uint8)

    img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

    return img_bytes

def gamma(img):

    global img_scaled, gamma_input_window

    gamma_input_window = create_gamma_input_window()
    while True:
        gamma_event, gamma_values = gamma_input_window.read()
        if gamma_event == sg.WIN_CLOSED:
            gamma_input_window.close()
            break
        elif gamma_event == 'Aplicar':
            if gamma_input_window:
                gamma = float(gamma_values['-GAMMA-'])
                gamma_input_window.close()

                # Constante de normalização para imagens de 8 bits
                c = 255 / (255 ** gamma)
                
                # Transformação gama para cada canal de cor
                transformed_channels = []
                for channel in cv2.split(img):
                    gamma_transformed = c * (channel ** gamma)
                    transformed_channels.append(gamma_transformed)
                
                # Juntar os canais transformados
                transformed_image = cv2.merge(transformed_channels)
                
                # Normalizar os valores para o intervalo [0, 255]
                transformed_image = (transformed_image / np.max(transformed_image)) * 255
                
                # Converter para o tipo de dados uint8
                img_scaled = transformed_image.astype(np.uint8)

                img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

                

                return img_bytes

def limiariza(img):

    global img_scaled, thresholding_input_window

    thresholding_input_window = create_limiarizacao_input_window()
    while True:
        thresholding_event, thresholding_values = thresholding_input_window.read()
        if thresholding_event == sg.WIN_CLOSED:
            thresholding_input_window.close()
            break
        elif thresholding_event == 'Aplicar':
            if thresholding_input_window:
                th = int(thresholding_values['-Thresholding-'])
                thresholding_input_window.close()

                img_scaled = np.zeros(img.shape, dtype=np.uint8)
                img_scaled[img > th] = 255                

                img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

                return img_bytes
                

def hist_monocromatico(img_path):
        # Carregue a imagem monocromática
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Calcule o histograma
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = np.array(hist)

    # Crie um array de valores de intensidade de 0 a 255
    intensity_values = np.arange(256)

    hist_monocromatico_window = create_hist_monocromatico_window(hist, intensity_values)

    while True:
        event, values = hist_monocromatico_window.read()
        if event == sg.WIN_CLOSED:
            break
        
    hist_monocromatico_window.close()

def hist_rgb(img):
    # Converter a imagem de BGR para RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Divida a imagem em canais R, G e B
    r_channel, g_channel, b_channel = cv2.split(img_rgb)

    # Calcule os histogramas de cada canal
    hist_r = cv2.calcHist([r_channel], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g_channel], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b_channel], [0], None, [256], [0, 256])

    # Crie um array de valores de intensidade de 0 a 255
    intensity_values = np.arange(256)

    # Normalize os histogramas
    hist_r = hist_r / (r_channel.shape[0] * r_channel.shape[1])
    hist_g = hist_g / (g_channel.shape[0] * g_channel.shape[1])
    hist_b = hist_b / (b_channel.shape[0] * b_channel.shape[1])

    # Combine os histogramas dos canais R, G e B em um único gráfico
    combined_hist = np.column_stack((hist_r, hist_g, hist_b))

    hist_rgb_window = create_hist_rgb_window(intensity_values, hist_r, hist_g, hist_b, combined_hist)

    while True:
        event, values = hist_rgb_window.read()
        if event == sg.WIN_CLOSED:
            break

    hist_rgb_window.close()

def especificar_hist(img_original, reference_img, bytes_img_ref):

    global original_img, img_scaled

    original_img = img_original

    window_hist_especificacao = create_compare_window('Gerar Transformação')
    window_hist_especificacao['-TXT-IMG-REF-'].update('Imagem de Referencia:')
    window_hist_especificacao['-REF-IMAGE-'].update(data=bytes_img_ref)

    while True:
        event, values = window_hist_especificacao.read()

        if event == sg.WIN_CLOSED:
            break
        elif event == 'Gerar Transformação':

            #Carrega Imagem original para a Janela
            new_text = 'Imagem Original:'
            window_hist_especificacao['-TXT-IMG-'].update(new_text)

            img_bytes = cv2.imencode('.png', img_original)[1].tobytes() 
            window_hist_especificacao['-ORI-IMAGE-'].update(data=img_bytes)

            # Realize a especificação do histograma para cada canal de cor
            chans_img = cv2.split(img_original) # separa os canais de cores
            chans_ref = cv2.split(reference_img) # separa os canais de cores

            # iterage nos canais da imagem de entrada e calcula o histograma
            pr = np.zeros((256, 3))
            for chan, n in zip(chans_img, np.arange(3)):
                pr[:,n] = cv2.calcHist([chan], [0], None, [256], [0, 256]).ravel()

            # iterage nos canais da imagem de referencia e calcula o histograma
            pz = np.zeros((256, 3))
            for chan, n in zip(chans_ref, np.arange(3)):
                pz[:,n] = cv2.calcHist([chan], [0], None, [256], [0, 256]).ravel()
            
            # calcula as CDFs para a imagem de entrada
            cdf_input = np.zeros((256, 3))
            for i in range(3):
                cdf_input[:,i] = np.cumsum(pr[:,i]) # referencia
            
            # calcula as CDFs para a imagem de referencia
            cdf_ref = np.zeros((256,3))
            for i in range(3):
                cdf_ref[:,i] = np.cumsum(pz[:,i]) # referencia
            

            img_scaled = np.zeros(img_original.shape) # imagem de saida

            for c in range(3): # percorre os planas de cores da imagem 
                for i in range(256): # corre ba cdf de cada plano da imagem
                    diff = np.absolute(cdf_ref[:,c] - cdf_input[i,c])
                    indice = diff.argmin() # indice da saida que tem o minimo de diferença
                    img_scaled[img_original[:,:,c] == i, c] = indice

            

            img_scaled = img_scaled.astype(np.uint8)

            img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

            return img_bytes
        

def while_window_open(window_filtro):
    while True:
        event, values = window_filtro.read()

        if event == sg.WIN_CLOSED or event == 'Cancelar':
            tamanho_kernel = 0
            break
        elif event == 'Aplicar':
            try:
                tamanho_kernel = int(values['-INPUT-'])
                if tamanho_kernel % 2 == 0 or tamanho_kernel < 3:
                    raise ValueError("O tamanho do kernel deve ser ímpar e maior ou igual a '3'.")
                
                # Valor válido, pode sair do loop
                break
            except ValueError as e:
                sg.popup_error(str(e))

    window_filtro.close()

    if 'tamanho_kernel' in locals():
        print(f'Tamanho do Kernel: {tamanho_kernel}')

    return tamanho_kernel

def filtro_box(img):

    global img_scaled

    window_filtro_box = create_filtro_window('Filtro Box - Tamanho do Kernel', 'Tamanho do Kernel para Filtro Box:')

    tamanho_kernel = while_window_open(window_filtro_box)

    if tamanho_kernel is 0:
        img_scaled = img

        img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()
    else:

        # Crie um kernel separável Box para o eixo X
        box_kernel_x = np.ones((1, tamanho_kernel), np.float32) / tamanho_kernel

        # Crie um kernel separável Box para o eixo Y
        box_kernel_y = np.ones((tamanho_kernel, 1), np.float32) / tamanho_kernel

        # Aplique o filtro Box separável à imagem
        img_scaled = cv2.sepFilter2D(img, -1, box_kernel_x, box_kernel_y)

        img_scaled = img_scaled.astype(np.uint8)

        img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

    return img_bytes
        
def filtro_gaussiano(img):

    global img_scaled

    window_gauss = create_filtro_window('Filtro Gaussiano - Tamanho do Kernel', 'Tamanho do Kernel para Filtro Gaussiano:')

    tamanho_kernel = while_window_open(window_gauss)

    if tamanho_kernel is 0:
        img_scaled = img

        img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()
    else:

        # Aplique o filtro Gaussiano separável à imagem
        img_scaled = cv2.GaussianBlur(img, (tamanho_kernel, tamanho_kernel), 0) # terceiro parametro 0 ira definir o melhor valor de sigma para o tamanho do kernel

        img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

    return img_bytes

def filtro_mediana(img):

    global img_scaled

    window_mediana = create_filtro_window('Filtro da Mediana - Tamanho do Kernel', 'Tamanho do Kernel para Filtro Mediana:')

    tamanho_kernel = while_window_open(window_mediana)

    if tamanho_kernel is 0:
        img_scaled = img

        img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

    else: 

        # Aplique o filtro da mediana separável à imagem
        img_scaled = cv2.medianBlur(img, tamanho_kernel)

        img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

    return img_bytes
        

def filtro_laplaciano(img):

    global img_scaled

    img = img_scaled

    window_laplaciana = create_compare_window('Gerar Aguçamento')
    window_laplaciana['-TXT-IMG-REF-'].update('Imagem de bordas e ruidos capiturados:')

    # Verifique se a imagem é colorida (3 canais) ou em tons de cinza (1 canal)
    if len(img.shape) == 3:  # Imagem colorida
        # Separe os canais de cores
        b, g, r = cv2.split(img)

        # Aplique o filtro Laplaciano a cada canal individualmente
        laplacian_b = cv2.Laplacian(b, cv2.CV_64F)
        laplacian_g = cv2.Laplacian(g, cv2.CV_64F)
        laplacian_r = cv2.Laplacian(r, cv2.CV_64F)

        # Converta os resultados para imagens em tons de cinza
        laplacian_b_abs = np.uint8(np.absolute(laplacian_b))
        laplacian_g_abs = np.uint8(np.absolute(laplacian_g))
        laplacian_r_abs = np.uint8(np.absolute(laplacian_r))

        # Combine os canais para criar uma imagem RGB resultante
        imagem_laplacian = cv2.merge((laplacian_b_abs, laplacian_g_abs, laplacian_r_abs))
    else:  # Imagem em tons de cinza
        # Aplique o filtro Laplaciano diretamente
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        laplacian_abs = np.uint8(np.absolute(laplacian))
        imagem_laplacian = laplacian_abs

    img_laplacian_bytes = cv2.imencode('.png', imagem_laplacian)[1].tobytes()
    window_laplaciana['-REF-IMAGE-'].update(data=img_laplacian_bytes)


    save_path = sg.popup_get_file('Se quiser salvar a imagem de borda', save_as=True, default_extension='.png')
    if save_path:
        cv2.imwrite(save_path, imagem_laplacian)
        print(f'Imagem salva em: {save_path}')

    while True:
        event, values = window_laplaciana.read()

        if event == sg.WIN_CLOSED:
            break
        elif event == 'Gerar Aguçamento':

            #Carrega Imagem original para a Janela
            img_bytes = cv2.imencode('.png', img)[1].tobytes() 
            window_laplaciana['-TXT-IMG-REF-'].update('Imagem Original:')
            window_laplaciana['-REF-IMAGE-'].update(data=img_bytes)

            # Aplique o efeito de aguçamento somando a imagem Laplaciana à imagem original
            img_scaled = cv2.add(img, imagem_laplacian)   

            img_scaled = img_scaled.astype(np.uint8)

            img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

            return img_bytes

def filtro_sobel(img):

    global img_scaled

    img = img_scaled

    window_sobel = create_compare_window('Gerar Aguçamento')
    window_sobel['-TXT-IMG-REF-'].update('Imagem de bordas e ruidos capiturados:')

    # Defina o tamanho da borda que deseja adicionar
    largura_da_borda = 1

    # Adicione uma borda à imagem com padding por replicação
    imagem_com_borda = cv2.copyMakeBorder(img, largura_da_borda, largura_da_borda, largura_da_borda, largura_da_borda, cv2.BORDER_REPLICATE)

    # Inicialize a matriz gradiente para cada canal de cor
    gradientes_canais = [np.zeros_like(img) for _ in range(3)]

    # Verifique se a imagem é colorida (3 canais) ou em tons de cinza (1 canal)
    if len(imagem_com_borda.shape) == 3:  # Imagem colorida
        # Separe os canais de cor R, G e B
        canal_r, canal_g, canal_b = cv2.split(imagem_com_borda)

        # Aplique o filtro de Sobel em x e y a cada canal
        for i, canal in enumerate([canal_r, canal_g, canal_b]):
            sobel_x = cv2.Sobel(canal, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(canal, cv2.CV_64F, 0, 1, ksize=3)

            # Calcule o gradiente da imagem combinando as derivadas em x e y
            gradiente = np.sqrt(sobel_x**2 + sobel_y**2)

            # Normalize o gradiente para exibir
            gradientes_canais[i] = cv2.normalize(gradiente, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Combine os canais de gradiente para obter a imagem gradiente colorida
        gradiente_colorido = cv2.merge(gradientes_canais)
        
    else:  # Imagem em tons de cinza
        # Aplique o filtro de Sobel diretamente a cada canal

        sobel_x = cv2.Sobel(imagem_com_borda, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(imagem_com_borda, cv2.CV_64F, 0, 1, ksize=3)

        # Calcule o gradiente da imagem combinando as derivadas em x e y
        gradiente = np.sqrt(sobel_x**2 + sobel_y**2)

        # Normalize o gradiente para exibir
        gradiente_colorido = cv2.normalize(gradiente, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        

    # Redimensione o gradiente para ter as mesmas dimensões da imagem original
    img_shape = img.shape[:2]
    gradiente_redimensionado = cv2.resize(gradiente_colorido, (img_shape[1], img_shape[0]))

    img_sobel_bytes = cv2.imencode('.png', gradiente_redimensionado)[1].tobytes()
    window_sobel['-REF-IMAGE-'].update(data=img_sobel_bytes)

    save_path = sg.popup_get_file('Se quiser salvar a imagem de borda', save_as=True, default_extension='.png')
    if save_path:
        cv2.imwrite(save_path, gradiente_redimensionado)
        print(f'Imagem salva em: {save_path}')

    while True:
        event, values = window_sobel.read()

        if event == sg.WIN_CLOSED:
            break
        elif event == 'Gerar Aguçamento':

            #Carrega Imagem original para a Janela
            img_bytes = cv2.imencode('.png', img)[1].tobytes() 
            window_sobel['-TXT-IMG-REF-'].update('Imagem Original:')
            window_sobel['-REF-IMAGE-'].update(data=img_bytes)

            # Aplique o efeito de aguçamento somando a imagem Laplaciana à imagem original
            img_scaled = cv2.add(img, gradiente_redimensionado)   

            img_scaled = img_scaled.astype(np.uint8)

            img_bytes = cv2.imencode('.png', img_scaled)[1].tobytes()

            return img_bytes