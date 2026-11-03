"""
JOVI Camera - Simulador de Camera com Visao Computacional
Sprint 4 - Computational Thinking with Python
FIAP 2026

Integrantes:
    Douglas Ferreira Santos Patriota - RM: 569713
    Octavio Kal Oliveira Lima        - RM: 571838
    Pedro Henrique Santos de Brito   - RM: 573187
    Vitor Santos Cintra              - RM: 569416

Tecnologias: Python 3.10+, OpenCV, PyQt5, NumPy, Requests


"""

import sys
import os
import re
import json
import unicodedata
from datetime import datetime

import cv2
import numpy as np
import requests
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDialog, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QSlider, QComboBox, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap

# =====================================================================
# CONSTANTES E CONFIGURACOES GLOBAIS
# =====================================================================

APP_NAME = "JOVI Camera"
VERSION = "4.0"

# Todos os arquivos ficam ao lado do script, independente de onde o
# programa for executado.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "fotos_jovi")
CONFIG_FILE = os.path.join(BASE_DIR, "config_jovi.json")
GALERIA_FILE = os.path.join(BASE_DIR, "galeria_jovi.json")
MAX_FOTOS = 50

FILTROS = {
    0: "Normal",
    1: "Preto e Branco",
    2: "Sépia",
    3: "Desfoque",
    4: "Detecção de Bordas",
    5: "Negativo",
    6: "Brilho Alto",
}

RESOLUCOES = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}

TIMERS_DISPONIVEIS = {
    "Sem Timer": 0,
    "3 segundos": 3,
    "5 segundos": 5,
    "10 segundos": 10,
}

CONFIG_PADRAO = {
    "resolucao": "720p",
    "qualidade": 85,
    "timer": "Sem Timer",
    "cidade": "São Paulo",
}

# --- API externa: Open-Meteo (https://open-meteo.com) - gratuita, sem chave ---
URL_GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
URL_PREVISAO = "https://api.open-meteo.com/v1/forecast"
API_TIMEOUT = 5  # segundos

# Codigos de tempo (WMO) devolvidos pela API -> descricao em portugues
DESCRICAO_CLIMA = {
    0: "Céu limpo", 1: "Predominantemente limpo", 2: "Parcialmente nublado",
    3: "Nublado", 45: "Neblina", 48: "Neblina com geada",
    51: "Garoa fraca", 53: "Garoa moderada", 55: "Garoa forte",
    56: "Garoa congelante", 57: "Garoa congelante forte",
    61: "Chuva fraca", 63: "Chuva moderada", 65: "Chuva forte",
    66: "Chuva congelante", 67: "Chuva congelante forte",
    71: "Neve fraca", 73: "Neve moderada", 75: "Neve forte", 77: "Grãos de neve",
    80: "Pancadas de chuva fracas", 81: "Pancadas de chuva moderadas",
    82: "Pancadas de chuva fortes", 85: "Pancadas de neve fracas",
    86: "Pancadas de neve fortes", 95: "Trovoada",
    96: "Trovoada com granizo", 99: "Trovoada com granizo forte",
}

# Garante que a pasta de fotos exista
os.makedirs(SAVE_DIR, exist_ok=True)


# =====================================================================
# EXCECAO PROPRIA
# =====================================================================

class ErroAPI(Exception):
    """Falha ao consultar a API externa (rede, resposta invalida, cidade inexistente)."""


# =====================================================================
# FUNCOES DE VALIDACAO (subrotinas puras: recebem parametro, retornam valor)
# =====================================================================

def validar_nome_foto(nome, nomes_existentes):
    """Valida o nome de uma foto antes de salvar. Retorna (valido: bool, mensagem: str)."""
    nome = nome.strip()
    caracteres_invalidos = set('/\\:*?"<>|')
    if len(nome) < 3:
        return False, "O nome precisa ter pelo menos 3 caracteres."
    if len(nome) > 40:
        return False, "O nome pode ter no máximo 40 caracteres."
    if any(c in caracteres_invalidos for c in nome):
        return False, "O nome não pode conter caracteres especiais (/ \\ : * ? \" < > |)."
    if nome.endswith(".") or nome.startswith("."):
        return False, "O nome não pode começar nem terminar com ponto."
    if nome.lower() in (n.lower() for n in nomes_existentes):
        return False, f"Já existe uma foto chamada '{nome}'."
    return True, "OK"


def validar_cidade(texto):
    """Valida o nome de cidade digitado pelo usuario. Retorna (valido: bool, mensagem: str)."""
    if not isinstance(texto, str):
        return False, "Informe o nome da cidade."
    texto = texto.strip()
    if len(texto) < 2 or len(texto) > 60:
        return False, "O nome da cidade deve ter entre 2 e 60 caracteres."
    if not any(c.isalpha() for c in texto):
        return False, "O nome da cidade deve conter letras."
    if not all(c.isalpha() or c in " -'." for c in texto):
        return False, "Use apenas letras, espaços, hífen, apóstrofo ou ponto no nome da cidade."
    return True, "OK"


def texto_ascii(texto):
    """Remove acentos. Necessario porque o cv2.putText so desenha caracteres ASCII."""
    normalizado = unicodedata.normalize("NFKD", texto)
    return normalizado.encode("ascii", "ignore").decode("ascii")


# =====================================================================
# FUNCOES DE PROCESSAMENTO DE IMAGEM
# =====================================================================

def aplicar_filtro(frame, filtro_id):
    """Recebe um frame (numpy array) e o id do filtro, retorna o frame processado."""
    match filtro_id:
        case 0:  # Normal
            return frame
        case 1:  # Preto e Branco
            cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)
        case 2:  # Sepia
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            sepia = cv2.transform(frame, kernel)
            return np.clip(sepia, 0, 255).astype(np.uint8)
        case 3:  # Desfoque
            return cv2.GaussianBlur(frame, (21, 21), 0)
        case 4:  # Deteccao de bordas
            bordas = cv2.Canny(frame, 100, 200)
            return cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
        case 5:  # Negativo
            return cv2.bitwise_not(frame)
        case 6:  # Brilho alto
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.6, 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        case _:
            return frame


def ajustar_brilho_contraste(frame, brilho=0, contraste=0):
    """Aplica ajuste linear de brilho/contraste via cv2.addWeighted. Retorna o frame ajustado."""
    fator = (259 * (contraste + 255)) / (255 * (259 - contraste)) if contraste != -255 else 1
    return cv2.addWeighted(frame, fator, frame, 0, brilho - 128 * (fator - 1))


# Cascade carregado uma unica vez (evita recriar o classificador a cada frame)
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_CLASSIFICADOR_FACES = cv2.CascadeClassifier(_CASCADE_PATH)


def detectar_faces(frame):
    """Detecta rostos no frame usando Haar Cascade. Retorna (frame_com_retangulos, quantidade)."""
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _CLASSIFICADOR_FACES.detectMultiScale(cinza, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame, len(faces)


def salvar_imagem(caminho, frame, qualidade):
    """Grava o frame como JPEG. Usa imencode + arquivo para funcionar com acentos no
    caminho (o cv2.imwrite falha silenciosamente com acentos no Windows).
    Retorna True/False conforme sucesso."""
    try:
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(qualidade)])
        if not ok:
            return False
        buffer.tofile(caminho)
        return True
    except (OSError, cv2.error):
        return False


def carregar_imagem(caminho):
    """Le uma imagem do disco (compativel com acentos). Retorna o frame ou None se falhar."""
    try:
        dados = np.fromfile(caminho, dtype=np.uint8)
        if dados.size == 0:
            return None
        return cv2.imdecode(dados, cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None


# =====================================================================
# FUNCOES DE API EXTERNA (Open-Meteo)
# =====================================================================

def buscar_coordenadas(cidade):
    """Consulta a API de geocodificacao e devolve um dicionario com nome, pais, latitude
    e longitude da cidade. Levanta ErroAPI em caso de falha ou cidade nao encontrada."""
    try:
        resposta = requests.get(
            URL_GEOCODING,
            params={"name": cidade, "count": 1, "language": "pt", "format": "json"},
            timeout=API_TIMEOUT,
        )
        resposta.raise_for_status()
        resultados = resposta.json().get("results")
        if not resultados:
            raise ErroAPI(f"Cidade '{cidade}' não encontrada.")
        primeiro = resultados[0]
        return {
            "nome": primeiro["name"],
            "pais": primeiro.get("country", ""),
            "latitude": float(primeiro["latitude"]),
            "longitude": float(primeiro["longitude"]),
        }
    except requests.exceptions.Timeout:
        raise ErroAPI("A consulta demorou demais. Verifique sua conexão.")
    except requests.exceptions.ConnectionError:
        raise ErroAPI("Sem conexão com a internet.")
    except requests.exceptions.HTTPError as erro:
        raise ErroAPI(f"O serviço de clima retornou erro ({erro.response.status_code}).")
    except (ValueError, KeyError, TypeError, AttributeError):
        raise ErroAPI("Resposta inesperada do serviço de clima.")


def buscar_clima(latitude, longitude):
    """Consulta a temperatura e a condicao do tempo atuais para uma coordenada.
    Retorna {"temperatura": float, "descricao": str}. Levanta ErroAPI em caso de falha."""
    try:
        resposta = requests.get(
            URL_PREVISAO,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,weather_code",
                "timezone": "auto",
            },
            timeout=API_TIMEOUT,
        )
        resposta.raise_for_status()
        atual = resposta.json()["current"]
        codigo = int(atual["weather_code"])
        return {
            "temperatura": round(float(atual["temperature_2m"]), 1),
            "descricao": DESCRICAO_CLIMA.get(codigo, "Condição desconhecida"),
        }
    except requests.exceptions.Timeout:
        raise ErroAPI("A consulta demorou demais. Verifique sua conexão.")
    except requests.exceptions.ConnectionError:
        raise ErroAPI("Sem conexão com a internet.")
    except requests.exceptions.HTTPError as erro:
        raise ErroAPI(f"O serviço de clima retornou erro ({erro.response.status_code}).")
    except (ValueError, KeyError, TypeError):
        raise ErroAPI("Resposta inesperada do serviço de clima.")


def obter_clima_cidade(cidade):
    """Une as duas consultas: acha a cidade e devolve o clima atual dela.
    Retorna {"cidade", "pais", "temperatura", "descricao", "consultado_em"}."""
    local = buscar_coordenadas(cidade)
    clima = buscar_clima(local["latitude"], local["longitude"])
    return {
        "cidade": local["nome"],
        "pais": local["pais"],
        "temperatura": clima["temperatura"],
        "descricao": clima["descricao"],
        "consultado_em": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    }


def formatar_clima(clima):
    """Texto amigavel para exibir na interface."""
    if not clima:
        return "Clima indisponível"
    return f"{clima['cidade']}: {clima['temperatura']:.1f} °C, {clima['descricao']}"


# =====================================================================
# FUNCOES DE PERSISTENCIA (JSON)
# =====================================================================

def _gravar_json(caminho, dados):
    """Grava JSON de forma segura: escreve em arquivo temporario e so entao substitui o
    original, para nao corromper os dados se o programa fechar no meio da gravacao."""
    temporario = caminho + ".tmp"
    try:
        with open(temporario, "w", encoding="utf-8") as arquivo:
            json.dump(dados, arquivo, ensure_ascii=False, indent=2)
        os.replace(temporario, caminho)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _isolar_arquivo_corrompido(caminho):
    """Renomeia um JSON ilegivel para .corrompido, preservando o conteudo para analise."""
    try:
        os.replace(caminho, caminho + ".corrompido")
    except OSError:
        pass


def sanitizar_configuracoes(dados):
    """Recebe qualquer coisa lida do JSON e devolve um dicionario de configuracoes valido,
    usando o padrao para cada campo ausente ou invalido."""
    config = dict(CONFIG_PADRAO)
    if not isinstance(dados, dict):
        return config

    resolucao = dados.get("resolucao")
    if isinstance(resolucao, str) and resolucao in RESOLUCOES:
        config["resolucao"] = resolucao

    try:
        qualidade = int(dados.get("qualidade"))
        if 10 <= qualidade <= 100:
            config["qualidade"] = qualidade
    except (TypeError, ValueError):
        pass

    timer = dados.get("timer")
    if isinstance(timer, str) and timer in TIMERS_DISPONIVEIS:
        config["timer"] = timer

    cidade = dados.get("cidade")
    if validar_cidade(cidade)[0]:
        config["cidade"] = cidade.strip()
    return config


def carregar_configuracoes():
    """Le config_jovi.json. Se nao existir ou estiver corrompido, usa a configuracao padrao.
    Retorna sempre um dicionario valido."""
    if not os.path.exists(CONFIG_FILE):
        return dict(CONFIG_PADRAO)
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as arquivo:
            return sanitizar_configuracoes(json.load(arquivo))
    except json.JSONDecodeError:
        _isolar_arquivo_corrompido(CONFIG_FILE)
        return dict(CONFIG_PADRAO)
    except OSError:
        return dict(CONFIG_PADRAO)


def salvar_configuracoes(config):
    """Grava o dicionario de configuracoes em disco. Retorna True/False conforme sucesso."""
    return _gravar_json(CONFIG_FILE, config)


def _normalizar_foto(item):
    """Valida um registro de foto lido do JSON. Retorna o dicionario completo (com valores
    padrao nos campos opcionais) ou None se o registro for invalido."""
    if not isinstance(item, dict):
        return None
    nome, caminho = item.get("nome"), item.get("caminho")
    if not isinstance(nome, str) or not isinstance(caminho, str) or not nome or not caminho:
        return None

    clima = item.get("clima")
    if isinstance(clima, dict):
        try:
            clima = {
                "cidade": str(clima["cidade"]),
                "temperatura": float(clima["temperatura"]),
                "descricao": str(clima["descricao"]),
            }
        except (KeyError, TypeError, ValueError):
            clima = None
    else:
        clima = None

    def numero(chave, padrao=0):
        try:
            return int(item.get(chave, padrao))
        except (TypeError, ValueError):
            return padrao

    return {
        "nome": nome,
        "caminho": caminho,  # relativo a BASE_DIR
        "timestamp": str(item.get("timestamp", "")),
        "filtro": numero("filtro") if numero("filtro") in FILTROS else 0,
        "brilho": numero("brilho"),
        "contraste": numero("contraste"),
        "favorita": bool(item.get("favorita", False)),
        "clima": clima,
    }


def caminho_absoluto(caminho_relativo):
    """Converte o caminho guardado no JSON (relativo) para um caminho completo no disco."""
    return os.path.join(BASE_DIR, caminho_relativo)


def carregar_galeria():
    """Le galeria_jovi.json e devolve (lista_de_fotos, aviso). Registros invalidos ou cujo
    arquivo de imagem nao existe mais sao descartados; 'aviso' explica o que aconteceu
    (string vazia quando esta tudo certo)."""
    if not os.path.exists(GALERIA_FILE):
        return [], ""
    try:
        with open(GALERIA_FILE, "r", encoding="utf-8") as arquivo:
            dados = json.load(arquivo)
    except json.JSONDecodeError:
        _isolar_arquivo_corrompido(GALERIA_FILE)
        return [], ("O arquivo da galeria estava corrompido e foi guardado como "
                    "'galeria_jovi.json.corrompido'. A galeria começou vazia.")
    except OSError:
        return [], "Não foi possível ler o arquivo da galeria."

    if not isinstance(dados, list):
        _isolar_arquivo_corrompido(GALERIA_FILE)
        return [], "O arquivo da galeria tinha um formato inválido e foi isolado."

    fotos, descartadas = [], 0
    for item in dados:
        foto = _normalizar_foto(item)
        if foto is None or not os.path.exists(caminho_absoluto(foto["caminho"])):
            descartadas += 1
            continue
        fotos.append(foto)

    aviso = ""
    if descartadas:
        aviso = (f"{descartadas} foto(s) da galeria foram ignoradas por estarem com "
                 "dados inválidos ou com o arquivo de imagem ausente.")
    return fotos, aviso


def salvar_galeria(fotos_lista):
    """Grava a lista de fotos (somente metadados + caminho da imagem) em galeria_jovi.json.
    Retorna True/False conforme sucesso."""
    return _gravar_json(GALERIA_FILE, fotos_lista)


def calcular_estatisticas(fotos_lista):
    """Recebe a lista de fotos (dicionarios) e retorna um dicionario com estatisticas:
    total, favoritas, uso por filtro, medias de brilho/contraste e temperatura media."""
    total = len(fotos_lista)
    favoritas = sum(1 for foto in fotos_lista if foto.get("favorita"))

    uso_filtros = {nome: 0 for nome in FILTROS.values()}
    for foto in fotos_lista:
        nome_filtro = FILTROS.get(foto.get("filtro", 0), "Normal")
        uso_filtros[nome_filtro] += 1

    if total > 0:
        brilho_medio = sum(f.get("brilho", 0) for f in fotos_lista) / total
        contraste_medio = sum(f.get("contraste", 0) for f in fotos_lista) / total
    else:
        brilho_medio = 0
        contraste_medio = 0

    temperaturas = [f["clima"]["temperatura"] for f in fotos_lista if f.get("clima")]
    temperatura_media = round(sum(temperaturas) / len(temperaturas), 1) if temperaturas else None

    return {
        "total": total,
        "favoritas": favoritas,
        "uso_filtros": uso_filtros,
        "brilho_medio": round(brilho_medio, 1),
        "contraste_medio": round(contraste_medio, 1),
        "com_clima": len(temperaturas),
        "temperatura_media": temperatura_media,
    }


# =====================================================================
# ESTILOS E HELPERS DE INTERFACE (evitam repeticao de codigo)
# =====================================================================

def btn_style(cor_fundo="#2b2b2b", cor_texto="#ffffff"):
    return (f"QPushButton {{ background-color: {cor_fundo}; color: {cor_texto}; "
            "border-radius: 8px; padding: 10px; font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background-color: #3d3d3d; }")


def combo_style():
    return ("QComboBox { background-color: #2b2b2b; color: white; padding: 6px; "
            "border-radius: 6px; }")


def slider_style():
    return ("QSlider::groove:horizontal { background: #444; height: 6px; border-radius: 3px; }"
            "QSlider::handle:horizontal { background: #00c8ff; width: 14px; "
            "margin: -5px 0; border-radius: 7px; }")


def input_style():
    return "background-color: #2b2b2b; color: white; padding: 8px; border-radius: 6px;"


def criar_label(texto, estilo="color: white;"):
    """Cria um QLabel ja estilizado."""
    label = QLabel(texto)
    label.setStyleSheet(estilo)
    return label


def criar_botao(texto, funcao, cor="#2b2b2b"):
    """Cria um QPushButton estilizado e ja conectado a uma funcao."""
    botao = QPushButton(texto)
    botao.setStyleSheet(btn_style(cor))
    botao.clicked.connect(funcao)
    return botao


def frame_para_pixmap(frame, largura, altura, suave=False):
    """Converte um frame do OpenCV (BGR) em QPixmap redimensionado para o QLabel."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, canais = rgb.shape
    imagem = QImage(rgb.data, w, h, canais * w, QImage.Format_RGB888)
    modo = Qt.SmoothTransformation if suave else Qt.FastTransformation
    return QPixmap.fromImage(imagem).scaled(largura, altura, Qt.KeepAspectRatio, modo)


# =====================================================================
# THREADS (para nao travar a interface)
# =====================================================================

class CameraThread(QThread):
    """Thread dedicada a captura continua de frames."""
    frame_pronto = pyqtSignal(np.ndarray)
    erro_camera = pyqtSignal(str)

    def __init__(self, resolucao=(1280, 720)):
        super().__init__()
        self.brilho = 0
        self.contraste = 0
        self.filtro_id = 0
        self.detectar_face = False
        self.rodando = True
        self.resolucao = resolucao
        self.ultimo_frame = None
        self.clima_texto = ""  # preenchido pela interface quando a API responde

    def run(self):
        captura = cv2.VideoCapture(0)
        if not captura.isOpened():
            self.erro_camera.emit(
                "Não foi possível acessar a câmera. Verifique se ela está conectada "
                "e se nenhum outro programa está utilizando-a.")
            return

        captura.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolucao[0])
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolucao[1])

        falhas_seguidas = 0
        while self.rodando:
            ok, frame = captura.read()
            if not ok:
                falhas_seguidas += 1
                if falhas_seguidas > 50:
                    self.erro_camera.emit("A câmera parou de responder.")
                    break
                continue
            falhas_seguidas = 0

            frame = ajustar_brilho_contraste(frame, self.brilho, self.contraste)
            frame = aplicar_filtro(frame, self.filtro_id)

            if self.detectar_face:
                frame, _ = detectar_faces(frame)

            # marca d'agua: nome do app, clima (se a API respondeu) e data/hora
            partes = [APP_NAME]
            if self.clima_texto:
                partes.append(self.clima_texto)
            partes.append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            cv2.putText(frame, " | ".join(partes), (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            self.ultimo_frame = frame
            self.frame_pronto.emit(frame)
            self.msleep(30)

        captura.release()

    def parar(self):
        self.rodando = False
        self.wait()


class ClimaThread(QThread):
    """Consulta a API de clima em segundo plano (a internet pode demorar)."""
    clima_pronto = pyqtSignal(dict)
    clima_erro = pyqtSignal(str)

    def __init__(self, cidade):
        super().__init__()
        self.cidade = cidade

    def run(self):
        try:
            self.clima_pronto.emit(obter_clima_cidade(self.cidade))
        except ErroAPI as erro:
            self.clima_erro.emit(str(erro))


# Mantem as threads de clima vivas ate terminarem, mesmo se a janela que as
# iniciou for fechada antes da resposta chegar.
_THREADS_CLIMA_ATIVAS = set()


def consultar_clima_async(cidade, ao_receber, ao_falhar):
    """Dispara a consulta de clima sem bloquear a interface.
    'ao_receber(dict)' e 'ao_falhar(str)' sao chamadas na thread da interface."""
    thread = ClimaThread(cidade)
    thread.clima_pronto.connect(ao_receber)
    thread.clima_erro.connect(ao_falhar)
    _THREADS_CLIMA_ATIVAS.add(thread)
    thread.finished.connect(lambda: _THREADS_CLIMA_ATIVAS.discard(thread))
    thread.start()
    return thread


# =====================================================================
# JANELA PRINCIPAL (MENU)
# =====================================================================

class JanelaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{VERSION}")
        self.setFixedSize(420, 640)
        self.setStyleSheet("background-color: #1b1b1b;")

        # Sprint 4: a galeria e as configuracoes vem do disco
        self.fotos_capturadas, self.aviso_inicial = carregar_galeria()
        self.configuracoes = carregar_configuracoes()

        self._montar_interface()

    def _montar_interface(self):
        central = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(30, 30, 30, 30)

        titulo = criar_label(APP_NAME, "color: white; font-size: 26px; font-weight: bold;")
        titulo.setAlignment(Qt.AlignCenter)
        layout.addWidget(titulo)

        self.label_contador = criar_label(self._texto_contador(), "color: #00c8ff; font-size: 13px;")
        self.label_contador.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_contador)

        botoes = [
            ("Câmera ao Vivo", self.abrir_camera),
            ("Filtros & Efeitos", self.abrir_filtros),
            ("Galeria de Fotos", self.abrir_galeria),
            ("Estatísticas da Galeria", self.abrir_estatisticas),
            ("Clima Atual", self.abrir_clima),
            ("Configurações", self.abrir_configuracoes),
            ("Sair", self.sair),
        ]
        for texto, funcao in botoes:
            botao = criar_botao(texto, funcao)
            botao.setMinimumHeight(46)
            layout.addWidget(botao)

        central.setLayout(layout)
        self.setCentralWidget(central)

    def _texto_contador(self):
        favoritas = sum(1 for f in self.fotos_capturadas if f.get("favorita"))
        return f"Fotos na galeria: {len(self.fotos_capturadas)}/{MAX_FOTOS}  |  Favoritas: {favoritas}"

    def atualizar_contador(self):
        self.label_contador.setText(self._texto_contador())

    # --- navegacao: cada tela abre como dialogo modal e, ao fechar, volta pro menu ---

    def _abrir_dialogo(self, janela):
        """Exibe o dialogo, espera ele fechar, libera memoria e atualiza o contador."""
        janela.exec_()
        janela.deleteLater()
        self.atualizar_contador()

    def abrir_camera(self):
        if len(self.fotos_capturadas) >= MAX_FOTOS:
            QMessageBox.warning(self, "Limite atingido",
                                f"A galeria já tem {MAX_FOTOS} fotos. Exclua alguma para capturar outra.")
            return
        self._abrir_dialogo(JanelaCamera(self.fotos_capturadas, self.configuracoes, self))

    def abrir_filtros(self):
        self._abrir_dialogo(JanelaFiltros(self.configuracoes, self))

    def abrir_galeria(self):
        if not self.fotos_capturadas:
            QMessageBox.information(self, "Galeria vazia",
                                    "Você ainda não capturou nenhuma foto.")
            return
        self._abrir_dialogo(JanelaGaleria(self.fotos_capturadas, self))

    def abrir_estatisticas(self):
        if not self.fotos_capturadas:
            QMessageBox.information(self, "Sem dados",
                                    "Capture algumas fotos para ver as estatísticas da galeria.")
            return
        self._abrir_dialogo(JanelaEstatisticas(self.fotos_capturadas, self))

    def abrir_clima(self):
        self._abrir_dialogo(JanelaClima(self.configuracoes, self))

    def abrir_configuracoes(self):
        self._abrir_dialogo(JanelaConfiguracoes(self.configuracoes, self))

    def sair(self):
        resposta = QMessageBox.question(
            self, "Sair", "Deseja realmente encerrar o programa?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if resposta == QMessageBox.Yes:
            self.close()


# =====================================================================
# JANELA: CAMERA AO VIVO
# =====================================================================

class JanelaCamera(QDialog):
    def __init__(self, fotos_lista, configuracoes, parent=None):
        super().__init__(parent)
        self.fotos_lista = fotos_lista
        self.configuracoes = configuracoes
        self.clima_atual = None   # ultimo clima recebido da API (ou None)
        self.setWindowTitle("Câmera ao Vivo")
        self.setFixedSize(720, 700)
        self.setStyleSheet("background-color: #1b1b1b;")

        self.segundos_restantes = 0
        self.timer_contagem = QTimer()
        self.timer_contagem.timeout.connect(self._tick_contagem)

        resolucao = RESOLUCOES.get(self.configuracoes.get("resolucao", "720p"), (1280, 720))
        self.camera_thread = CameraThread(resolucao)
        self.camera_thread.frame_pronto.connect(self._atualizar_frame)
        self.camera_thread.erro_camera.connect(self._exibir_erro_camera)

        self._montar_interface()
        self.camera_thread.start()
        self._buscar_clima()

    def _exibir_erro_camera(self, mensagem):
        QMessageBox.critical(self, "Erro de câmera", mensagem)
        self.close()

    def _montar_interface(self):
        layout = QVBoxLayout()

        self.label_video = QLabel()
        self.label_video.setFixedSize(680, 400)
        self.label_video.setStyleSheet("background-color: black; border-radius: 8px;")
        layout.addWidget(self.label_video, alignment=Qt.AlignCenter)

        self.label_contagem = criar_label("", "color: #ff5555; font-size: 22px; font-weight: bold;")
        self.label_contagem.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_contagem)

        # clima (API externa)
        linha_clima = QHBoxLayout()
        self.label_clima = criar_label("Clima: consultando...", "color: #00c8ff; font-size: 12px;")
        linha_clima.addWidget(self.label_clima, 1)
        botao_clima = criar_botao("Atualizar clima", self._buscar_clima)
        botao_clima.setStyleSheet(btn_style() + "QPushButton { padding: 4px 10px; font-size: 12px; }")
        linha_clima.addWidget(botao_clima)
        layout.addLayout(linha_clima)

        # sliders de brilho e contraste
        grid = QGridLayout()
        grid.addWidget(criar_label("Brilho"), 0, 0)
        self.slider_brilho = QSlider(Qt.Horizontal)
        self.slider_brilho.setRange(-100, 100)
        self.slider_brilho.setValue(0)
        self.slider_brilho.setStyleSheet(slider_style())
        self.slider_brilho.valueChanged.connect(lambda v: setattr(self.camera_thread, "brilho", v))
        grid.addWidget(self.slider_brilho, 0, 1)

        grid.addWidget(criar_label("Contraste"), 1, 0)
        self.slider_contraste = QSlider(Qt.Horizontal)
        self.slider_contraste.setRange(-100, 100)
        self.slider_contraste.setValue(0)
        self.slider_contraste.setStyleSheet(slider_style())
        self.slider_contraste.valueChanged.connect(lambda v: setattr(self.camera_thread, "contraste", v))
        grid.addWidget(self.slider_contraste, 1, 1)
        layout.addLayout(grid)

        linha_opcoes = QHBoxLayout()
        self.check_faces = QCheckBox("Detectar rostos")
        self.check_faces.setStyleSheet("color: white;")
        self.check_faces.stateChanged.connect(
            lambda v: setattr(self.camera_thread, "detectar_face", v == Qt.Checked))
        linha_opcoes.addWidget(self.check_faces)

        self.combo_timer = QComboBox()
        self.combo_timer.addItems(TIMERS_DISPONIVEIS.keys())
        self.combo_timer.setCurrentText(self.configuracoes.get("timer", "Sem Timer"))
        self.combo_timer.setStyleSheet(combo_style())
        linha_opcoes.addWidget(self.combo_timer)
        layout.addLayout(linha_opcoes)

        linha_nome = QHBoxLayout()
        self.input_nome = QLineEdit()
        self.input_nome.setPlaceholderText("Nome da foto...")
        self.input_nome.setMaxLength(40)
        self.input_nome.setStyleSheet(input_style())
        self.input_nome.returnPressed.connect(self.iniciar_captura)
        linha_nome.addWidget(self.input_nome)
        linha_nome.addWidget(criar_botao("Capturar Foto", self.iniciar_captura, "#00a86b"))
        layout.addLayout(linha_nome)

        layout.addWidget(criar_botao("Voltar ao Menu", self.close))
        self.setLayout(layout)

    def _atualizar_frame(self, frame):
        self.label_video.setPixmap(
            frame_para_pixmap(frame, self.label_video.width(), self.label_video.height()))

    # --- clima via API ---

    def _buscar_clima(self):
        cidade = self.configuracoes.get("cidade", CONFIG_PADRAO["cidade"])
        self.label_clima.setText(f"Clima: consultando {cidade}...")
        consultar_clima_async(cidade, self._clima_recebido, self._clima_falhou)

    def _clima_recebido(self, clima):
        self.clima_atual = clima
        self.label_clima.setText("Clima: " + formatar_clima(clima))
        self.camera_thread.clima_texto = texto_ascii(
            f"{clima['cidade']} {clima['temperatura']:.0f}C")

    def _clima_falhou(self, mensagem):
        self.clima_atual = None
        self.label_clima.setText(f"Clima indisponível: {mensagem} (as fotos serão salvas sem clima)")
        self.camera_thread.clima_texto = ""

    # --- captura com timer ---

    def iniciar_captura(self):
        if self.timer_contagem.isActive():
            return  # ja existe uma contagem regressiva em andamento

        # valida o nome ANTES de iniciar a contagem, para o usuario nao esperar a toa
        nomes_existentes = [f["nome"] for f in self.fotos_lista]
        valido, mensagem = validar_nome_foto(self.input_nome.text(), nomes_existentes)
        if not valido:
            QMessageBox.warning(self, "Nome inválido", mensagem)
            return

        segundos = TIMERS_DISPONIVEIS.get(self.combo_timer.currentText(), 0)
        if segundos == 0:
            self._salvar_foto()
            return
        self.segundos_restantes = segundos
        self.label_contagem.setText(str(self.segundos_restantes))
        self.timer_contagem.start(1000)

    def _tick_contagem(self):
        self.segundos_restantes -= 1
        if self.segundos_restantes > 0:
            self.label_contagem.setText(str(self.segundos_restantes))
        else:
            self.timer_contagem.stop()
            self.label_contagem.setText("")
            self._salvar_foto()

    def _salvar_foto(self):
        nome = self.input_nome.text().strip()
        nomes_existentes = [f["nome"] for f in self.fotos_lista]
        valido, mensagem = validar_nome_foto(nome, nomes_existentes)  # revalida (a lista pode ter mudado)
        if not valido:
            QMessageBox.warning(self, "Nome inválido", mensagem)
            return

        frame = self.camera_thread.ultimo_frame
        if frame is None:
            QMessageBox.warning(self, "Sem imagem", "Aguarde a câmera inicializar.")
            return

        agora = datetime.now()
        nome_arquivo = f"{nome}_{agora.strftime('%Y%m%d_%H%M%S')}.jpg"
        caminho_relativo = os.path.join("fotos_jovi", nome_arquivo)
        qualidade = self.configuracoes.get("qualidade", 85)

        if not salvar_imagem(caminho_absoluto(caminho_relativo), frame, qualidade):
            QMessageBox.critical(self, "Erro ao salvar",
                                 "Não foi possível gravar a imagem no disco. "
                                 "Verifique o espaço disponível e as permissões da pasta.")
            return

        clima = None
        if self.clima_atual:
            clima = {
                "cidade": self.clima_atual["cidade"],
                "temperatura": self.clima_atual["temperatura"],
                "descricao": self.clima_atual["descricao"],
            }

        foto = {
            "nome": nome,
            "caminho": caminho_relativo,
            "timestamp": agora.strftime("%d/%m/%Y %H:%M:%S"),
            "filtro": self.camera_thread.filtro_id,
            "brilho": self.camera_thread.brilho,
            "contraste": self.camera_thread.contraste,
            "favorita": False,
            "clima": clima,
        }
        self.fotos_lista.append(foto)
        self.input_nome.clear()

        if salvar_galeria(self.fotos_lista):
            QMessageBox.information(self, "Foto salva", f"Foto '{nome}' salva com sucesso!")
        else:
            QMessageBox.warning(self, "Foto salva com aviso",
                                f"A foto '{nome}' foi salva, mas não foi possível atualizar "
                                "o arquivo da galeria. Ela pode não aparecer na próxima execução.")

    def closeEvent(self, event):
        self.timer_contagem.stop()
        self.camera_thread.parar()
        event.accept()


# =====================================================================
# JANELA: FILTROS & EFEITOS
# =====================================================================

class JanelaFiltros(QDialog):
    def __init__(self, configuracoes, parent=None):
        super().__init__(parent)
        self.configuracoes = configuracoes
        self.setWindowTitle("Filtros & Efeitos")
        self.setFixedSize(680, 560)
        self.setStyleSheet("background-color: #1b1b1b;")

        resolucao = RESOLUCOES.get(self.configuracoes.get("resolucao", "720p"), (1280, 720))
        self.camera_thread = CameraThread(resolucao)
        self.camera_thread.frame_pronto.connect(self._atualizar_frame)
        self.camera_thread.erro_camera.connect(self._exibir_erro_camera)

        self._montar_interface()
        self.camera_thread.start()

    def _exibir_erro_camera(self, mensagem):
        QMessageBox.critical(self, "Erro de câmera", mensagem)
        self.close()

    def _montar_interface(self):
        layout = QVBoxLayout()

        self.label_video = QLabel()
        self.label_video.setFixedSize(640, 380)
        self.label_video.setStyleSheet("background-color: black; border-radius: 8px;")
        layout.addWidget(self.label_video, alignment=Qt.AlignCenter)

        self.combo_filtro = QComboBox()
        self.combo_filtro.addItems(FILTROS.values())
        self.combo_filtro.setStyleSheet(combo_style())
        self.combo_filtro.currentIndexChanged.connect(
            lambda i: setattr(self.camera_thread, "filtro_id", i))
        layout.addWidget(self.combo_filtro)

        atalhos = QGridLayout()
        for posicao, (filtro_id, nome) in enumerate(FILTROS.items()):
            botao = criar_botao(nome, lambda _, i=filtro_id: self.combo_filtro.setCurrentIndex(i))
            botao.setStyleSheet(btn_style() + "QPushButton { padding: 6px; font-size: 12px; }")
            atalhos.addWidget(botao, posicao // 4, posicao % 4)
        layout.addLayout(atalhos)

        layout.addWidget(criar_botao("Voltar ao Menu", self.close))
        self.setLayout(layout)

    def _atualizar_frame(self, frame):
        self.label_video.setPixmap(
            frame_para_pixmap(frame, self.label_video.width(), self.label_video.height()))

    def closeEvent(self, event):
        self.camera_thread.parar()
        event.accept()


# =====================================================================
# JANELA: GALERIA DE FOTOS
# =====================================================================

class JanelaGaleria(QDialog):
    def __init__(self, fotos_lista, parent=None):
        super().__init__(parent)
        self.fotos_lista = fotos_lista
        self.indice_atual = 0
        self.somente_favoritas = False
        self.setWindowTitle("Galeria de Fotos")
        self.setFixedSize(780, 560)
        self.setStyleSheet("background-color: #1b1b1b;")

        self._montar_interface()
        self._recarregar_lista()

    def _montar_interface(self):
        layout_geral = QHBoxLayout()

        # painel esquerdo: lista
        painel_esq = QVBoxLayout()
        self.check_favoritas = QCheckBox("Mostrar somente favoritas")
        self.check_favoritas.setStyleSheet("color: white;")
        self.check_favoritas.stateChanged.connect(self._alternar_filtro_favoritas)
        painel_esq.addWidget(self.check_favoritas)

        self.lista_widget = QListWidget()
        self.lista_widget.setStyleSheet("background-color: #2b2b2b; color: white;")
        self.lista_widget.currentRowChanged.connect(self._selecionar_por_indice)
        painel_esq.addWidget(self.lista_widget)
        layout_geral.addLayout(painel_esq, 2)

        # painel direito: preview + metadados
        painel_dir = QVBoxLayout()
        self.label_preview = QLabel()
        self.label_preview.setFixedSize(400, 300)
        self.label_preview.setAlignment(Qt.AlignCenter)
        self.label_preview.setStyleSheet("background-color: black; border-radius: 8px; color: #888;")
        painel_dir.addWidget(self.label_preview, alignment=Qt.AlignCenter)

        self.label_metadados = criar_label("", "color: #cccccc; font-size: 12px;")
        painel_dir.addWidget(self.label_metadados)

        linha_nav = QHBoxLayout()
        linha_nav.addWidget(criar_botao("< Anterior", self._foto_anterior))
        linha_nav.addWidget(criar_botao("Próxima >", self._foto_proxima))
        painel_dir.addLayout(linha_nav)

        linha_acoes = QHBoxLayout()
        self.botao_favoritar = criar_botao("Favoritar", self._alternar_favorita, "#c9a227")
        linha_acoes.addWidget(self.botao_favoritar)
        linha_acoes.addWidget(criar_botao("Excluir", self._excluir_foto, "#b03030"))
        painel_dir.addLayout(linha_acoes)

        painel_dir.addWidget(criar_botao("Voltar ao Menu", self.close))

        layout_geral.addLayout(painel_dir, 3)
        self.setLayout(layout_geral)

    def _fotos_visiveis(self):
        if self.somente_favoritas:
            return [f for f in self.fotos_lista if f.get("favorita")]
        return self.fotos_lista

    def _foto_selecionada(self):
        """Retorna a foto atualmente selecionada, ou None se nao houver."""
        visiveis = self._fotos_visiveis()
        if not visiveis or not 0 <= self.indice_atual < len(visiveis):
            return None
        return visiveis[self.indice_atual]

    def _persistir(self):
        """Grava a galeria em disco e avisa o usuario se falhar."""
        if not salvar_galeria(self.fotos_lista):
            QMessageBox.warning(self, "Erro ao salvar",
                                "Não foi possível atualizar o arquivo da galeria no disco.")

    def _alternar_filtro_favoritas(self, estado):
        self.somente_favoritas = estado == Qt.Checked
        self._recarregar_lista()

    def _recarregar_lista(self, linha=0):
        self.lista_widget.blockSignals(True)
        self.lista_widget.clear()
        for foto in self._fotos_visiveis():
            estrela = "★ " if foto.get("favorita") else ""
            self.lista_widget.addItem(QListWidgetItem(f"{estrela}{foto['nome']}  ({foto['timestamp']})"))
        self.lista_widget.blockSignals(False)

        total = self.lista_widget.count()
        if total > 0:
            linha = max(0, min(linha, total - 1))
            self.lista_widget.setCurrentRow(linha)
            self.indice_atual = linha
            self._atualizar_preview()
        else:
            self.indice_atual = 0
            self.label_preview.clear()
            self.label_metadados.setText("Nenhuma foto para exibir.")

    def _selecionar_por_indice(self, indice):
        if indice < 0:
            return
        self.indice_atual = indice
        self._atualizar_preview()

    def _atualizar_preview(self):
        foto = self._foto_selecionada()
        if foto is None:
            return

        frame = carregar_imagem(caminho_absoluto(foto["caminho"]))
        if frame is None:
            self.label_preview.clear()
            self.label_preview.setText("Imagem não encontrada no disco")
        else:
            self.label_preview.setPixmap(frame_para_pixmap(
                frame, self.label_preview.width(), self.label_preview.height(), suave=True))

        nome_filtro = FILTROS.get(foto.get("filtro", 0), "Normal")
        favorita_txt = "Sim" if foto.get("favorita") else "Não"
        clima_txt = formatar_clima(foto.get("clima")) if foto.get("clima") else "não registrado"
        self.label_metadados.setText(
            f"Nome: {foto['nome']}\n"
            f"Data/Hora: {foto['timestamp']}\n"
            f"Filtro: {nome_filtro}\n"
            f"Brilho: {foto.get('brilho', 0)}  |  Contraste: {foto.get('contraste', 0)}\n"
            f"Clima: {clima_txt}\n"
            f"Favorita: {favorita_txt}"
        )
        self.botao_favoritar.setText("Desfavoritar" if foto.get("favorita") else "Favoritar")

    def _foto_anterior(self):
        total = len(self._fotos_visiveis())
        if total:
            self.lista_widget.setCurrentRow((self.indice_atual - 1) % total)

    def _foto_proxima(self):
        total = len(self._fotos_visiveis())
        if total:
            self.lista_widget.setCurrentRow((self.indice_atual + 1) % total)

    def _alternar_favorita(self):
        foto = self._foto_selecionada()
        if foto is None:
            return
        foto["favorita"] = not foto.get("favorita", False)
        self._persistir()
        # com o filtro "somente favoritas" ligado, a foto pode sair da lista
        self._recarregar_lista(self.indice_atual)

    def _excluir_foto(self):
        foto = self._foto_selecionada()
        if foto is None:
            return
        resposta = QMessageBox.question(
            self, "Confirmar exclusão",
            f"Deseja excluir a foto '{foto['nome']}'?\nO arquivo de imagem também será apagado.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if resposta != QMessageBox.Yes:
            return

        try:
            os.remove(caminho_absoluto(foto["caminho"]))
        except FileNotFoundError:
            pass  # ja nao existia: o objetivo (sumir do disco) esta cumprido
        except OSError:
            QMessageBox.warning(self, "Erro ao excluir",
                                "Não foi possível apagar o arquivo da imagem. "
                                "Ele pode estar aberto em outro programa.")
            return

        self.fotos_lista.remove(foto)
        self._persistir()
        self._recarregar_lista(self.indice_atual)


# =====================================================================
# JANELA: ESTATISTICAS DA GALERIA
# =====================================================================

class JanelaEstatisticas(QDialog):
    def __init__(self, fotos_lista, parent=None):
        super().__init__(parent)
        self.fotos_lista = fotos_lista
        self.setWindowTitle("Estatísticas da Galeria")
        self.setFixedSize(420, 560)
        self.setStyleSheet("background-color: #1b1b1b;")
        self._montar_interface()

    def _montar_interface(self):
        stats = calcular_estatisticas(self.fotos_lista)

        layout = QVBoxLayout()
        titulo = criar_label("Estatísticas da Galeria", "color: white; font-size: 18px; font-weight: bold;")
        titulo.setAlignment(Qt.AlignCenter)
        layout.addWidget(titulo)

        temperatura = (f"{stats['temperatura_media']} °C" if stats["temperatura_media"] is not None
                       else "sem dados")
        layout.addWidget(criar_label(
            f"Total de fotos: {stats['total']}\n"
            f"Favoritas: {stats['favoritas']}\n"
            f"Brilho médio: {stats['brilho_medio']}\n"
            f"Contraste médio: {stats['contraste_medio']}\n"
            f"Fotos com clima registrado: {stats['com_clima']}\n"
            f"Temperatura média das fotos: {temperatura}",
            "color: #00c8ff; font-size: 13px;"))

        layout.addWidget(criar_label(
            "Uso por filtro:", "color: white; font-size: 14px; font-weight: bold; margin-top: 10px;"))
        for nome_filtro, quantidade in stats["uso_filtros"].items():
            layout.addWidget(criar_label(f"{nome_filtro}: {quantidade}", "color: #cccccc; font-size: 12px;"))

        layout.addStretch()
        layout.addWidget(criar_botao("Voltar ao Menu", self.close))
        self.setLayout(layout)


# =====================================================================
# JANELA: CLIMA ATUAL (consome a API externa)
# =====================================================================

class JanelaClima(QDialog):
    def __init__(self, configuracoes, parent=None):
        super().__init__(parent)
        self.configuracoes = configuracoes
        self.setWindowTitle("Clima Atual")
        self.setFixedSize(400, 340)
        self.setStyleSheet("background-color: #1b1b1b;")
        self._montar_interface()
        self._atualizar()

    def _montar_interface(self):
        layout = QVBoxLayout()

        self.label_cidade = criar_label("", "color: white; font-size: 20px; font-weight: bold;")
        self.label_cidade.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_cidade)

        self.label_temperatura = criar_label("--", "color: #00c8ff; font-size: 44px; font-weight: bold;")
        self.label_temperatura.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_temperatura)

        self.label_descricao = criar_label("", "color: #cccccc; font-size: 16px;")
        self.label_descricao.setAlignment(Qt.AlignCenter)
        self.label_descricao.setWordWrap(True)
        layout.addWidget(self.label_descricao)

        self.label_rodape = criar_label("", "color: #888; font-size: 11px;")
        self.label_rodape.setAlignment(Qt.AlignCenter)
        self.label_rodape.setWordWrap(True)
        layout.addWidget(self.label_rodape)

        layout.addStretch()
        self.botao_atualizar = criar_botao("Atualizar", self._atualizar, "#00a86b")
        layout.addWidget(self.botao_atualizar)
        layout.addWidget(criar_botao("Voltar ao Menu", self.close))
        self.setLayout(layout)

    def _atualizar(self):
        cidade = self.configuracoes.get("cidade", CONFIG_PADRAO["cidade"])
        self.botao_atualizar.setEnabled(False)
        self.label_cidade.setText(cidade)
        self.label_temperatura.setText("...")
        self.label_descricao.setText("Consultando o serviço de clima...")
        self.label_rodape.setText("")
        consultar_clima_async(cidade, self._exibir_clima, self._exibir_erro)

    def _exibir_clima(self, clima):
        local = f"{clima['cidade']}, {clima['pais']}" if clima["pais"] else clima["cidade"]
        self.label_cidade.setText(local)
        self.label_temperatura.setText(f"{clima['temperatura']:.1f} °C")
        self.label_descricao.setText(clima["descricao"])
        self.label_rodape.setText(f"Consultado em {clima['consultado_em']}\nFonte: Open-Meteo (open-meteo.com)")
        self.botao_atualizar.setEnabled(True)

    def _exibir_erro(self, mensagem):
        self.label_temperatura.setText("--")
        self.label_descricao.setText(mensagem)
        self.label_rodape.setText("Dica: confira a cidade em Configurações e sua conexão.")
        self.botao_atualizar.setEnabled(True)


# =====================================================================
# JANELA: CONFIGURACOES
# =====================================================================

class JanelaConfiguracoes(QDialog):
    def __init__(self, configuracoes, parent=None):
        super().__init__(parent)
        self.configuracoes = configuracoes
        self.setWindowTitle("Configurações")
        self.setFixedSize(380, 500)
        self.setStyleSheet("background-color: #1b1b1b;")
        self._montar_interface()

    def _montar_interface(self):
        layout = QVBoxLayout()

        layout.addWidget(criar_label("Resolução da câmera"))
        self.combo_resolucao = QComboBox()
        self.combo_resolucao.addItems(RESOLUCOES.keys())
        self.combo_resolucao.setCurrentText(self.configuracoes.get("resolucao", "720p"))
        self.combo_resolucao.setStyleSheet(combo_style())
        layout.addWidget(self.combo_resolucao)

        layout.addWidget(criar_label("Qualidade JPEG (10-100%)"))
        self.slider_qualidade = QSlider(Qt.Horizontal)
        self.slider_qualidade.setRange(10, 100)
        self.slider_qualidade.setValue(self.configuracoes.get("qualidade", 85))
        self.slider_qualidade.setStyleSheet(slider_style())
        self.label_qualidade = criar_label(f"{self.slider_qualidade.value()}%")
        self.slider_qualidade.valueChanged.connect(
            lambda v: self.label_qualidade.setText(f"{v}%"))
        layout.addWidget(self.slider_qualidade)
        layout.addWidget(self.label_qualidade)

        layout.addWidget(criar_label("Timer de captura padrão"))
        self.combo_timer = QComboBox()
        self.combo_timer.addItems(TIMERS_DISPONIVEIS.keys())
        self.combo_timer.setCurrentText(self.configuracoes.get("timer", "Sem Timer"))
        self.combo_timer.setStyleSheet(combo_style())
        layout.addWidget(self.combo_timer)

        layout.addWidget(criar_label("Cidade (para o clima das fotos)"))
        self.input_cidade = QLineEdit(self.configuracoes.get("cidade", CONFIG_PADRAO["cidade"]))
        self.input_cidade.setMaxLength(60)
        self.input_cidade.setPlaceholderText("Ex.: São Paulo")
        self.input_cidade.setStyleSheet(input_style())
        layout.addWidget(self.input_cidade)

        layout.addWidget(criar_botao("Salvar Configurações", self._salvar, "#00a86b"))
        layout.addWidget(criar_botao("Restaurar Padrões", self._restaurar_padroes))
        layout.addWidget(criar_botao("Voltar ao Menu", self.close))
        self.setLayout(layout)

    def _salvar(self):
        valida, mensagem = validar_cidade(self.input_cidade.text())
        if not valida:
            QMessageBox.warning(self, "Cidade inválida", mensagem)
            return

        self.configuracoes["resolucao"] = self.combo_resolucao.currentText()
        self.configuracoes["qualidade"] = self.slider_qualidade.value()
        self.configuracoes["timer"] = self.combo_timer.currentText()
        self.configuracoes["cidade"] = self.input_cidade.text().strip()

        if salvar_configuracoes(self.configuracoes):
            QMessageBox.information(self, "Configurações salvas",
                                    "Suas preferências foram salvas e serão\n"
                                    "carregadas automaticamente na próxima vez que abrir o app.")
        else:
            QMessageBox.warning(self, "Erro ao salvar",
                                "Não foi possível gravar o arquivo de configurações em disco.")

    def _restaurar_padroes(self):
        self.combo_resolucao.setCurrentText(CONFIG_PADRAO["resolucao"])
        self.slider_qualidade.setValue(CONFIG_PADRAO["qualidade"])
        self.combo_timer.setCurrentText(CONFIG_PADRAO["timer"])
        self.input_cidade.setText(CONFIG_PADRAO["cidade"])
        QMessageBox.information(self, "Padrões restaurados",
                                "Os valores padrão foram restaurados. Clique em\n"
                                "'Salvar Configurações' para gravar em disco.")


# =====================================================================
# PONTO DE ENTRADA
# =====================================================================

def main():
    print(f"Iniciando {APP_NAME} v{VERSION}...")
    app = QApplication(sys.argv)
    janela = JanelaPrincipal()
    janela.show()
    if janela.aviso_inicial:
        QMessageBox.warning(janela, "Aviso ao carregar a galeria", janela.aviso_inicial)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
