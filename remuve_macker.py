import streamlit as st
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Watermark Remover", layout="centered")

st.title("Limpiador de Marcas de Agua")
st.write("Sube tu video y ajusta las coordenadas para borrar logotipos usando Inpainting.")

# 1. Carga del archivo
uploaded_file = st.file_uploader("Sube tu archivo de video (MP4)", type=["mp4"])

if uploaded_file is not None:
    # Guardar el video subido en un archivo temporal
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Leer el primer fotograma para ayudar con las coordenadas
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    
    if ret:
        st.write("### Ajuste de Coordenadas")
        st.write("Define la posición de la marca de agua. Valores por defecto aproximados para la esquina superior izquierda.")
        
        col1, col2 = st.columns(2)
        with col1:
            x = st.number_input("Posición X (Izquierda a Derecha)", min_value=0, value=40)
            w = st.number_input("Ancho (Pixels)", min_value=10, value=250)
        with col2:
            y = st.number_input("Posición Y (Arriba a Abajo)", min_value=0, value=120)
            h = st.number_input("Alto (Pixels)", min_value=10, value=100)

        # Mostrar una previsualización del primer fotograma con la máscara dibujada
        preview_frame = first_frame.copy()
        cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        preview_frame_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
        st.image(preview_frame_rgb, caption="Previsualización del área a borrar (Caja Roja)", use_column_width=True)

    if st.button("Procesar Video"):
        # Preparar archivo de salida
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        # Propiedades del video original
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Configurar el escritor de video (Usando codec H264 compatible con web)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # UI: Barra de progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Resetear el video al principio
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Crear la máscara (imagen negra con un rectángulo blanco en la marca de agua)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            
            # Aplicar el algoritmo TELEA para rellenar la zona
            # El valor '3' es el radio de análisis de píxeles
            frame_limpio = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            
            out.write(frame_limpio)
            
            frame_count += 1
            # Actualizar barra de progreso
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Procesando: {progress}% ({frame_count}/{total_frames} fotogramas)")
            
        cap.release()
        out.release()
        
        st.success("¡Video procesado con éxito!")
        
        # Botón de descarga
        with open(output_path, 'rb') as f:
            st.download_button(
                label="Descargar Video Limpio",
                data=f,
                file_name="video_sin_marca.mp4",
                mime="video/mp4"
            )
            
        # Limpieza de archivos temporales del servidor
        os.remove(video_path)
        os.remove(output_path)
