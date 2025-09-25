# Detector Inteligente API v7.3.4

API Node.js/Express para análise completa de imagens incluindo detecção NSFW, jogos, software e análise de ociosidade.

## Funcionalidades

### Análise Completa
- **Detecção NSFW**: 93% precisão usando NSFWJS
- **Detecção de Jogos**: Algoritmo híbrido customizado
- **Detecção de Software**: OCR com Tesseract.js
- **Análise de Ociosidade**: Comparação temporal de screenshots

### Processamento
- Suporte a arquivos ZIP com múltiplas imagens
- Processamento paralelo configurável
- Análise temporal detalhada
- Sanitização automática de dados

## Instalação

```bash
# Instalar dependências
npm install

# Configurar ambiente
cp .env.example .env

# Executar em desenvolvimento
npm run dev

# Executar em produção
npm start
```

## Uso

### Endpoint Principal

```http
POST /analyze
Content-Type: multipart/form-data
```

**Parâmetros:**
- `file`: Arquivo ZIP ou imagem única (JPG, PNG, WebP, BMP, TIFF)

### Exemplo com cURL

```bash
# Analisar arquivo ZIP
curl -X POST http://localhost:3000/analyze \
  -F "file=@screenshots.zip"

# Analisar imagem única
curl -X POST http://localhost:3000/analyze \
  -F "file=@imagem.jpg"
```

### Exemplo com JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:3000/analyze', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);
```

## Resposta da API

```json
{
  "success": true,
  "timestamp": "2025-09-23T13:30:45.123Z",
  "processingTime": 2500,
  "version": "7.3.4",
  
  "fileInfo": {
    "originalName": "screenshots.zip",
    "size": 2523207,
    "type": "application/zip",
    "isZip": true
  },
  
  "extraction": {
    "totalFiles": 15,
    "imageCount": 12,
    "errorCount": 0,
    "errors": []
  },
  
  "images": [
    {
      "filename": "screenshot_20250923133045.jpg",
      "index": 0,
      "size": 245760,
      "width": 1920,
      "height": 1080,
      "format": "jpeg",
      "timestamp": "2025-09-23T13:30:45.000Z",
      "processingTime": 1250,
      
      "nsfw": {
        "success": true,
        "isNSFW": false,
        "confidence": 0.95,
        "classifications": [
          {
            "className": "Neutral",
            "probability": 0.85,
            "percentage": 85.0
          }
        ],
        "details": {
          "isPorn": false,
          "isHentai": false,
          "isSexy": false,
          "primaryCategory": "Neutral",
          "scores": {
            "porn": 0.02,
            "hentai": 0.01,
            "sexy": 0.12,
            "drawing": 0.15,
            "neutral": 0.85
          }
        }
      },
      
      "games": {
        "success": true,
        "isGaming": true,
        "confidence": 0.78,
        "gameScore": 0.78,
        "detectedGame": "League of Legends",
        "features": {
          "hudElements": 0.82,
          "gameColors": 0.75,
          "gameUI": 0.68,
          "visualComplexity": 0.71,
          "gameText": 0.45
        }
      },
      
      "software": {
        "success": true,
        "detected": true,
        "confidence": 0.65,
        "softwareList": [
          {
            "name": "Steam",
            "confidence": 0.85,
            "type": "gaming_platform",
            "matches": 3
          }
        ],
        "urls": [
          "https://store.steampowered.com"
        ],
        "domains": [
          "steampowered.com"
        ],
        "ocrText": "Steam Library Games Store..."
      },
      
      "errors": []
    }
  ],
  
  "idleness": {
    "success": true,
    "totalImages": 12,
    "idlenessAnalysis": {
      "totalPeriods": 11,
      "idlePeriods": 3,
      "activePeriods": 8,
      "averageIdleness": 35.5,
      "maxIdleness": 85.2,
      "minIdleness": 5.1,
      "idlenessPercentage": 27.27
    },
    "timeAnalysis": {
      "13": {
        "screenshots": 5,
        "averageIdleness": 25.4,
        "activityLevel": "moderate"
      },
      "14": {
        "screenshots": 7,
        "averageIdleness": 42.1,
        "activityLevel": "low"
      }
    },
    "productivityAnalysis": {
      "productiveTime": 65.5,
      "unproductiveTime": 20.0,
      "neutralTime": 14.5,
      "productivityScore": 72.75
    },
    "summary": {
      "overallIdleness": 35.5,
      "productivityLevel": "Alta",
      "mostActiveHour": 13,
      "leastActiveHour": 14,
      "recommendation": "Nível de atividade moderado. Há espaço para melhorias."
    }
  },
  
  "statistics": {
    "totalImages": 12,
    "nsfw": {
      "detected": 0,
      "porn": 0,
      "hentai": 0,
      "sexy": 0,
      "averageConfidence": 0.92
    },
    "games": {
      "detected": 8,
      "averageConfidence": 0.65,
      "detectedGames": ["League of Legends", "Steam"]
    },
    "software": {
      "detected": 5,
      "totalSoftware": 12,
      "totalUrls": 8,
      "totalDomains": 6,
      "averageConfidence": 0.58
    },
    "processing": {
      "averageTime": 1250,
      "totalTime": 15000,
      "errors": 0
    },
    "idleness": {
      "averageIdleness": 35.5,
      "idlePercentage": 27.27,
      "productivityScore": 72.75,
      "mostActiveHour": 13,
      "leastActiveHour": 14
    }
  }
}
```

## Endpoints Adicionais

### Status da API
```http
GET /status
```

### Informações da API
```http
GET /
```

## Configuração

### Variáveis de Ambiente (.env)

```env
# Servidor
PORT=3000
MAX_FILE_SIZE_MB=50

# Thresholds NSFW
NSFW_PORN_THRESHOLD=0.5
NSFW_HENTAI_THRESHOLD=0.5
NSFW_SEXY_THRESHOLD=0.7

# Detecção de jogos
GAME_DETECTION_THRESHOLD=0.3
GAME_DRAWING_THRESHOLD=0.6

# Análise de ociosidade
IDLENESS_CHANGE_THRESHOLD=15
IDLENESS_HIGH_THRESHOLD=70
IDLENESS_MODERATE_THRESHOLD=40

# Processamento paralelo
ENABLE_PARALLEL_PROCESSING=true
MAX_PARALLEL_PROCESSES=4
```

## Estrutura do Projeto

```
detector-api/
├── server.js              # Servidor principal
├── package.json           # Dependências
├── .env                   # Configurações
├── README.md              # Documentação
├── utils/
│   └── zipProcessor.js    # Processamento de ZIP
└── detectors/
    ├── nsfwDetector.js    # Detector NSFW
    ├── gameDetector.js    # Detector de jogos
    ├── softwareDetector.js # Detector de software
    └── idlenessDetector.js # Detector de ociosidade
```

## Tecnologias

- **Node.js** 18+
- **Express** 5.x
- **TensorFlow.js** + NSFWJS
- **Tesseract.js** (OCR)
- **Sharp** (processamento de imagem)
- **Canvas** (análise de imagem)
- **JSZip** (processamento de ZIP)

## Limitações

- Tamanho máximo de arquivo: 50MB (configurável)
- Formatos suportados: ZIP, JPG, PNG, WebP, BMP, TIFF
- Processamento local (sem envio para servidores externos)
- Requer Node.js 18+ para compatibilidade com TensorFlow.js

## Desenvolvimento

```bash
# Modo desenvolvimento com watch
npm run dev

# Logs detalhados
DEBUG_ENABLED=true npm run dev
```

---

**Versão**: 7.3.4  
**Data**: 23/09/2025  
**Compatibilidade**: Node.js 18+
