from modules.window_layouts import main_window
from modules.funcoes import *



image_path = None

window = None
original_img = None
img = None

"""
    Esse modulo Controla Todos os Eventos do app 

    Menu:
        Opções de Arquivo
        Ferramentas
        Transformações de Intensidade
        ...

"""



# Event Loop para processar os eventos e obter os valores de entrada
def eventos():

    global image_path, window, original_img, img
    
    window = main_window()

    while True: 
            event, values = window.read()
            if event == sg.WIN_CLOSED:
                break

            # FUNÇÔES DE CARREGAMENTO DE IMAGEM, RESET, SAVE, ETC...

            elif event == 'Carregar Imagem':
                
                image_path = sg.popup_get_file('Escolher Imagem')  # Abrir diálogo para escolher a imagem
                img_bytes, img, original_img = carregar_img(image_path)

                window['-IMAGE-'].update(data=img_bytes)

            elif event == 'Reset imagem':
                if image_path:
                    img, img_bytes = reset_image()

                    window['-IMAGE-'].update(data=img_bytes)
            
            elif event == 'Desfazer' and img is not None:
                if image_path:
                    img_bytes = desfazer_alteracao(img)

                    window['-IMAGE-'].update(data=img_bytes)
                    
            elif event == 'Confirmar alterações':
                if image_path:
                    img = confirmar_alteracoes()
            
            elif event == 'Salvar': # Salva a imagem por cima da original
                if image_path:
                    salvar(image_path)
            
            elif event == 'Salvar como': # oferece opção de salvar o imagem em outro diretorio
                if image_path:
                    salvar_como(image_path)

            ## ZOOM
               
            elif event == 'Zoom +':
                if image_path:

                    img_bytes = zoom_plus(img)
                    
                    window['-IMAGE-'].update(data=img_bytes)

            elif event == 'Zoom -':
                if image_path:

                    img_bytes = zoom_minus(img)
                    
                    window['-IMAGE-'].update(data=img_bytes)
            
            # NEGATIVA e MONOCROMATICA
            #         
            elif event == 'Negativa': 
                if image_path:
                    img_bytes = Negativa(img)

                    window['-IMAGE-'].update(data=img_bytes)

            elif event == 'Monocromática':
                if image_path:
                    img_bytes = monocromatica(img)

                    window['-IMAGE-'].update(data=img_bytes)
            
            # HISTOGRAMAS
            
            elif event == 'Equalização de hist.': 
                if image_path:                  
                    img_bytes = equalizar_hist(img)

                    window['-IMAGE-'].update(data=img_bytes)

            elif event == 'Especificação de hist.': 
                if image_path: 
                    image_ref_path = sg.popup_get_file('Escolher Imagem')  # Abrir diálogo para escolher a imagem
                    if image_ref_path:
                        img_bytes, img_ref, _ = carregar_img(image_ref_path)                 
                        img_bytes = especificar_hist(img, img_ref, img_bytes)

                        window['-IMAGE-'].update(data=img_bytes)
            
            elif event == 'Hist. Monocromatico': 
                if image_path:                  
                    hist_monocromatico(image_path)
            
            elif event == 'Hist. Mult. Canais RGB': 
                if image_path:                  
                    hist_rgb(img)

            # AJUSTE DE BRILHO e ZOOM

            elif event == 'Ajustar Brilho':
                if image_path:           
                    img_bytes = ajuste_brilho(img)

                    window['-IMAGE-'].update(data=img_bytes)

                        
            elif event == 'Ajustar Zoom':
                if image_path:
                    img_bytes = ajuste_zoom(img)

                    window['-IMAGE-'].update(data=img_bytes)

            # REAMOSTRAGEM LOG E EXPONENCIAL

            elif event == 'Logarítimica':
                if image_path:
                    img_bytes = log_transf(img)

                    window['-IMAGE-'].update(data=img_bytes)

            elif event == 'Exponencial':
                if image_path:
                    img_bytes = exp_transf(img)
                    
                    window['-IMAGE-'].update(data=img_bytes)

            # GAMME E LIMIARIZAR

            elif event == 'GAMMA':
                if image_path:
                    img_bytes = gamma(img)
                    
                    window['-IMAGE-'].update(data=img_bytes)

            elif event == 'Limiarizar':
                if image_path:
                    img_bytes = limiariza(img)
                    
                    window['-IMAGE-'].update(data=img_bytes)

            ## FILTROS ESPACIAIS

            ## FILTROS PASSA BAIXA

            elif event == 'Filtro Box':
                if image_path:
                    img_bytes = filtro_box(img)
                    
                    window['-IMAGE-'].update(data=img_bytes)
                    
            elif event == 'Filtro Gaussiano':
                if image_path:
                    img_bytes = filtro_gaussiano(img)
                    
                    window['-IMAGE-'].update(data=img_bytes)
            
            elif event == 'Filtro da Mediana':
                if image_path:
                    img_bytes = filtro_mediana(img)
                    
                    window['-IMAGE-'].update(data=img_bytes)

            # Filtro Passa Alta

            # Filtros Sobel e Laplaciana

            elif event == 'Aguçamento Laplaciano':
                if image_path:
                    img_bytes = filtro_laplaciano(img)
                    
                    window['-IMAGE-'].update(data=img_bytes)

            elif event == 'Aguçamento de Sobel':
                if image_path:
                    img_bytes = filtro_sobel(img)

                    window['-IMAGE-'].update(data=img_bytes)

    # Certifique-se de fechar todas as janelas antes de sair
    if zoom_slider_window:
        zoom_slider_window.close()
    
    if brilho_slider_window:
        zoom_slider_window.close()
    
    if thresholding_input_window:
        thresholding_input_window.close()
    
    if gamma_input_window:
        gamma_input_window.close()
    
    if hist_monocromatico_window:
        hist_monocromatico_window.close()
    
    if window_hist_especificacao:
        window_hist_especificacao.close()

    if window_filtro_box:
        window_filtro_box.close()
    
    if window_gauss:
        window_gauss.close()

    if window_mediana:
        window_mediana.close()

    if window_laplaciana:
        window_laplaciana.close()
    
    if window_sobel:
        window_sobel.close()


    window.close()                           
