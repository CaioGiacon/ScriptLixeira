"""
Teste do modelo da lixeira inteligente (YOLOv8n-cls: papel, plastico, vidro)
usando a webcam do notebook.

Instalação:
    pip install ultralytics opencv-python

Uso:
    python testar_modelo.py

Teclas na janela da câmera:
    ESPAÇO -> captura o objeto e classifica
    S      -> salva a última captura na pasta "capturas"
    Q      -> sai
"""

import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

CAMINHO_MODELO = "TCC_lixeira/best.pt"
INDICE_CAMERA = 0          # webcam embutida do notebook
LIMIAR = 0.60              # abaixo disso, o resultado é "incerto"
PASTA_CAPTURAS = Path("capturas")

# Textos sem acento, pois o cv2.putText não desenha caracteres acentuados
LIXEIRAS = {
    "papel": "Lixeira AZUL (papel)",
    "plastico": "Lixeira VERMELHA (plastico)",
    "vidro": "Lixeira VERDE (vidro)",
}


def abrir_camera():
    # No Windows, CAP_DSHOW abre a webcam mais rápido e evita avisos
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(INDICE_CAMERA, backend)
    if not cap.isOpened():
        raise RuntimeError(
            "Não foi possível abrir a webcam. Feche outros apps que usem a câmera "
            "e confira as permissões do sistema."
        )
    # Descarta os primeiros quadros (a câmera ajusta exposição ao abrir)
    for _ in range(10):
        cap.read()
    return cap


def classificar(modelo, frame):
    """Classifica um quadro BGR e retorna (classe, confiança, top3)."""
    r = modelo.predict(frame, imgsz=224, verbose=False)[0]
    nomes, probs = r.names, r.probs
    classe = nomes[probs.top1]
    conf = float(probs.top1conf)
    top3 = [(nomes[i], float(probs.data[i])) for i in probs.top5[:3]]
    if conf < LIMIAR:
        classe = "incerto"
    return classe, conf, top3


def escrever(img, texto, y, cor=(255, 255, 255), escala=0.7):
    # contorno preto + texto colorido para ler em qualquer fundo
    cv2.putText(img, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 4)
    cv2.putText(img, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 2)


def desenhar_resultado(frame, classe, conf, top3):
    img = frame.copy()
    cor = (0, 200, 0) if classe != "incerto" else (0, 0, 255)
    destino = LIXEIRAS.get(classe, "Nenhuma (confianca baixa)")
    escrever(img, f"{classe.upper()} ({conf:.0%})", 35, cor, 1.0)
    escrever(img, destino, 70, cor)
    for i, (nome, p) in enumerate(top3):
        escrever(img, f"{nome}: {p:.1%}", 105 + i * 28, escala=0.6)
    escrever(img, "ESPACO: nova captura | S: salvar | Q: sair", img.shape[0] - 15, escala=0.55)
    return img


def main():
    modelo = YOLO(CAMINHO_MODELO)
    print(f"Modelo carregado. Classes: {list(modelo.names.values())}")
    cap = abrir_camera()
    print("Câmera aberta. Posicione o objeto e tecle ESPAÇO para capturar.")

    janela = "Teste - Lixeira Inteligente"
    ultimo_frame = None          # última captura (sem texto desenhado)
    resultado_img = None         # captura com o resultado desenhado

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Falha ao ler a câmera.")
            break

        if resultado_img is not None:
            exibir = resultado_img          # mantém o resultado na tela
        else:
            exibir = frame.copy()
            escrever(exibir, "ESPACO: capturar | Q: sair", exibir.shape[0] - 15, escala=0.55)

        cv2.imshow(janela, exibir)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            break

        if tecla == ord(" "):
            ultimo_frame = frame.copy()
            classe, conf, top3 = classificar(modelo, ultimo_frame)
            resultado_img = desenhar_resultado(ultimo_frame, classe, conf, top3)
            detalhes = " | ".join(f"{n}: {p:.1%}" for n, p in top3)
            print(f"-> {classe} ({conf:.1%}) | {LIXEIRAS.get(classe, 'sem destino')} | [{detalhes}]")

        if tecla == ord("s") and ultimo_frame is not None:
            PASTA_CAPTURAS.mkdir(exist_ok=True)
            arq = PASTA_CAPTURAS / f"captura_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(arq), ultimo_frame)
            print(f"Captura salva em {arq}")

        # qualquer outra tecla (ex.: ESC) volta para a visualização ao vivo
        if tecla == 27:
            resultado_img = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
