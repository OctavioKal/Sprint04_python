# JOVI Camera

Simulador de câmera para smartphone desenvolvido em Python, com interface gráfica em PyQt5, processamento de imagem via OpenCV e consulta de clima por API externa.

Projeto acadêmico da disciplina **Computational Thinking with Python** — FIAP 2026.

## Integrantes

- Douglas Ferreira Santos Patriota — RM 569713
- Octavio Kal Oliveira Lima — RM 571838
- Pedro Henrique Santos de Brito — RM 573187
- Vitor Santos Cintra — RM 569416

## Funcionalidades

- **Câmera ao Vivo** — feed em tempo real com brilho, contraste, detecção de rostos e timer de captura (3s, 5s ou 10s)
- **Filtros & Efeitos** — 7 filtros aplicados em tempo real (Normal, Preto e Branco, Sépia, Desfoque, Detecção de Bordas, Negativo, Brilho Alto)
- **Galeria de Fotos** — visualização, navegação, exclusão e marcação de favoritas; **persistente entre execuções**
- **Estatísticas da Galeria** — total de fotos, favoritas, uso por filtro, médias de brilho/contraste e temperatura média
- **Clima Atual** *(novo na Sprint 4)* — consulta a temperatura e a condição do tempo da cidade configurada
- **Configurações persistentes** — resolução, qualidade JPEG, timer e cidade salvos em `config_jovi.json`

## Novidades da Sprint 4

### Persistência de dados (JSON)

| Arquivo | Conteúdo |
|---|---|
| `config_jovi.json` | Preferências do usuário (resolução, qualidade, timer, cidade) |
| `galeria_jovi.json` | Metadados de cada foto: nome, caminho da imagem, data/hora, filtro, brilho, contraste, favorita e clima |
| `fotos_jovi/` | Imagens `.jpg` capturadas |

- A galeria é carregada ao abrir o programa; favoritar ou excluir uma foto atualiza o arquivo na hora.
- Se um JSON estiver corrompido, ele é preservado como `*.corrompido` e o programa inicia com valores padrão, sem travar.
- Registros inválidos ou com a imagem apagada do disco são ignorados e o usuário é avisado.
- A gravação é segura: escreve em arquivo temporário e só depois substitui o original.

### Integração com API externa — Open-Meteo

- Usa a [Open-Meteo](https://open-meteo.com) (gratuita, **sem chave de API**):
  - `geocoding-api.open-meteo.com` converte o nome da cidade em coordenadas;
  - `api.open-meteo.com` devolve temperatura e condição do tempo atuais.
- O clima aparece na tela da câmera, na marca d'água da foto e nos metadados da galeria.
- As consultas rodam em uma thread separada, então a interface nunca trava. Sem internet, a câmera continua funcionando e as fotos são salvas sem clima.
- Tratamento de erros: timeout, falta de conexão, erro HTTP, cidade inexistente e resposta inesperada.

### Validações e tratamento de exceções

- **Nome da foto:** de 3 a 40 caracteres, sem caracteres proibidos, sem repetição (ignora maiúsculas/minúsculas). É validado *antes* da contagem do timer.
- **Cidade:** apenas letras, espaços, hífen, apóstrofo e ponto.
- **Arquivos:** `try/except` em toda leitura/gravação de JSON e imagem (`JSONDecodeError`, `OSError`, `cv2.error`).
- **Imagens com acento no nome** funcionam no Windows (uso de `imencode`/`imdecode` no lugar de `imwrite`/`imread`).

## Organização do código (`script.py`)

O código é dividido em subrotinas com parâmetros e retorno bem definidos:

- **Validação:** `validar_nome_foto`, `validar_cidade`
- **Imagem:** `aplicar_filtro`, `ajustar_brilho_contraste`, `detectar_faces`, `salvar_imagem`, `carregar_imagem`
- **API:** `buscar_coordenadas`, `buscar_clima`, `obter_clima_cidade`, `formatar_clima`
- **Persistência:** `carregar_configuracoes`, `salvar_configuracoes`, `carregar_galeria`, `salvar_galeria`, `calcular_estatisticas`
- **Interface (PyQt5):** `JanelaPrincipal` (menu), `JanelaCamera`, `JanelaFiltros`, `JanelaGaleria`, `JanelaEstatisticas`, `JanelaClima`, `JanelaConfiguracoes`
- **Threads:** `CameraThread`, `ClimaThread`

## Tecnologias

- Python 3.10+
- OpenCV (`opencv-python`)
- PyQt5
- NumPy
- Requests

## Como executar

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/jovi-camera.git
cd jovi-camera
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python -m venv venv
```

Ativar no Windows (Git Bash):
```bash
source venv/Scripts/activate
```

Ativar no Windows (PowerShell):
```powershell
venv\Scripts\Activate.ps1
```

Ativar no Linux/macOS:
```bash
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Execute o programa

```bash
python script.py
```

## Observações

- É necessário ter uma webcam conectada ao computador. O clima exige conexão com a internet (opcional).
- Todos os arquivos gerados (`fotos_jovi/`, `config_jovi.json`, `galeria_jovi.json`) ficam na mesma pasta do `script.py`.
- O limite da galeria é de 50 fotos.
- O projeto fixa `opencv-python<5` no `requirements.txt` porque o OpenCV 5.0 moveu o `CascadeClassifier` (usado na detecção de rosto) para o pacote `opencv-contrib-python`, quebrando compatibilidade com a versão 4.x usada aqui.
