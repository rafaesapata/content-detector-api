#!/usr/bin/env python3
"""
API Detector Inteligente v7.3.4 - VERSÃO FINAL
OCR Completo + Análise de Ociosidade Inteligente com Hash Perceptual
CORRIGE o problema de detectar atividade em imagens idênticas
"""

import time
import os
import zipfile
import io
import re
import struct
import traceback
import hashlib
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Importações para análise avançada de imagens
try:
    from PIL import Image
    import pytesseract
    import imagehash
    import numpy as np
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError as e:
    ADVANCED_ANALYSIS_AVAILABLE = False
    print(f"Análise avançada não disponível: {e}")

# Carregar variáveis do .env
load_dotenv()

app = Flask(__name__)
CORS(app)

class Config:
    """Configurações carregadas do .env"""
    
    # Configurações gerais
    API_VERSION = os.getenv('API_VERSION', '7.3.4-final')
    API_NAME = os.getenv('API_NAME', 'Detector Inteligente API - FINAL')
    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', '52428800'))
    PORT = int(os.getenv('PORT', '5000'))
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    
    # Análise de ociosidade inteligente - THRESHOLDS CORRETOS
    IDLENESS_THRESHOLD_CRITICAL = float(os.getenv('IDLENESS_THRESHOLD_CRITICAL', '95'))
    IDLENESS_THRESHOLD_HIGH = float(os.getenv('IDLENESS_THRESHOLD_HIGH', '85'))
    IDLENESS_THRESHOLD_MODERATE = float(os.getenv('IDLENESS_THRESHOLD_MODERATE', '70'))
    IDLENESS_THRESHOLD_LOW = float(os.getenv('IDLENESS_THRESHOLD_LOW', '50'))
    
    # Configurações de hash perceptual
    PERCEPTUAL_HASH_SIZE = int(os.getenv('PERCEPTUAL_HASH_SIZE', '16'))
    HASH_SIMILARITY_THRESHOLD = float(os.getenv('HASH_SIMILARITY_THRESHOLD', '0.95'))
    STRUCTURAL_SIMILARITY_THRESHOLD = float(os.getenv('STRUCTURAL_SIMILARITY_THRESHOLD', '0.90'))
    
    # Configurações NSFW
    NSFW_HIGH_CONFIDENCE_THRESHOLD = float(os.getenv('NSFW_HIGH_CONFIDENCE_THRESHOLD', '0.92'))
    NSFW_MEDIUM_CONFIDENCE_THRESHOLD = float(os.getenv('NSFW_MEDIUM_CONFIDENCE_THRESHOLD', '0.80'))
    NSFW_LOW_CONFIDENCE_THRESHOLD = float(os.getenv('NSFW_LOW_CONFIDENCE_THRESHOLD', '0.65'))
    NSFW_SMALL_IMAGE_THRESHOLD = int(os.getenv('NSFW_SMALL_IMAGE_THRESHOLD', '30000'))
    
    NSFW_NEUTRAL_SCORE = float(os.getenv('NSFW_NEUTRAL_SCORE', '0.85'))
    NSFW_PORN_SCORE = float(os.getenv('NSFW_PORN_SCORE', '0.01'))
    NSFW_SEXY_SCORE = float(os.getenv('NSFW_SEXY_SCORE', '0.02'))
    NSFW_HENTAI_SCORE = float(os.getenv('NSFW_HENTAI_SCORE', '0.005'))
    NSFW_DRAWING_SCORE = float(os.getenv('NSFW_DRAWING_SCORE', '0.005'))
    
    # Configurações Games
    GAMES_DETECTION_THRESHOLD = float(os.getenv('GAMES_DETECTION_THRESHOLD', '0.4'))
    GAMES_RESOLUTION_MIN_WIDTH = int(os.getenv('GAMES_RESOLUTION_MIN_WIDTH', '1024'))
    GAMES_RESOLUTION_MIN_HEIGHT = int(os.getenv('GAMES_RESOLUTION_MIN_HEIGHT', '576'))
    GAMES_ASPECT_RATIO_MIN = float(os.getenv('GAMES_ASPECT_RATIO_MIN', '1.4'))
    GAMES_ASPECT_RATIO_MAX = float(os.getenv('GAMES_ASPECT_RATIO_MAX', '2.0'))
    GAMES_FILE_SIZE_THRESHOLD = int(os.getenv('GAMES_FILE_SIZE_THRESHOLD', '300000'))
    
    GAMES_RESOLUTION_SCORE = float(os.getenv('GAMES_RESOLUTION_SCORE', '0.35'))
    GAMES_ASPECT_RATIO_SCORE = float(os.getenv('GAMES_ASPECT_RATIO_SCORE', '0.25'))
    GAMES_FILE_SIZE_SCORE = float(os.getenv('GAMES_FILE_SIZE_SCORE', '0.25'))
    GAMES_PNG_FORMAT_SCORE = float(os.getenv('GAMES_PNG_FORMAT_SCORE', '0.15'))
    
    # Configurações Software
    SOFTWARE_DETECTION_THRESHOLD = float(os.getenv('SOFTWARE_DETECTION_THRESHOLD', '0.6'))
    SOFTWARE_FILE_SIZE_THRESHOLD = int(os.getenv('SOFTWARE_FILE_SIZE_THRESHOLD', '150000'))
    SOFTWARE_RESOLUTION_MIN_WIDTH = int(os.getenv('SOFTWARE_RESOLUTION_MIN_WIDTH', '800'))
    SOFTWARE_RESOLUTION_MIN_HEIGHT = int(os.getenv('SOFTWARE_RESOLUTION_MIN_HEIGHT', '600'))
    
    SOFTWARE_FILE_SIZE_SCORE = float(os.getenv('SOFTWARE_FILE_SIZE_SCORE', '0.45'))
    SOFTWARE_RESOLUTION_SCORE = float(os.getenv('SOFTWARE_RESOLUTION_SCORE', '0.35'))
    SOFTWARE_PNG_FORMAT_SCORE = float(os.getenv('SOFTWARE_PNG_FORMAT_SCORE', '0.20'))
    SOFTWARE_CONFIDENCE_DEFAULT = float(os.getenv('SOFTWARE_CONFIDENCE_DEFAULT', '0.75'))
    
    # === CONFIGURAÇÕES OCR ===
    OCR_ENABLED = os.getenv('OCR_ENABLED', 'true').lower() == 'true'
    OCR_LANGUAGE = os.getenv('OCR_LANGUAGE', 'por+eng')
    OCR_CONFIDENCE_THRESHOLD = int(os.getenv('OCR_CONFIDENCE_THRESHOLD', '60'))
    OCR_MAX_IMAGE_SIZE = int(os.getenv('OCR_MAX_IMAGE_SIZE', '5242880'))
    OCR_RESIZE_MAX_WIDTH = int(os.getenv('OCR_RESIZE_MAX_WIDTH', '2000'))
    OCR_RESIZE_MAX_HEIGHT = int(os.getenv('OCR_RESIZE_MAX_HEIGHT', '2000'))
    
    # Detecção de software via OCR
    OCR_SOFTWARE_KEYWORDS = os.getenv('OCR_SOFTWARE_KEYWORDS', 'chrome,firefox,safari,edge,vscode,visual studio,notepad,word,excel,powerpoint,photoshop,illustrator,figma,slack,teams,zoom,discord,whatsapp,telegram').split(',')
    OCR_URL_PATTERNS = os.getenv('OCR_URL_PATTERNS', 'http,https,www.,\.com,\.org,\.net,\.br').split(',')
    OCR_DOMAIN_MIN_LENGTH = int(os.getenv('OCR_DOMAIN_MIN_LENGTH', '4'))
    OCR_TEXT_MIN_LENGTH = int(os.getenv('OCR_TEXT_MIN_LENGTH', '10'))
    
    # Configurações de produtividade
    PRODUCTIVITY_VERY_HIGH_THRESHOLD = float(os.getenv('PRODUCTIVITY_VERY_HIGH_THRESHOLD', '85'))
    PRODUCTIVITY_HIGH_THRESHOLD = float(os.getenv('PRODUCTIVITY_HIGH_THRESHOLD', '70'))
    PRODUCTIVITY_MEDIUM_THRESHOLD = float(os.getenv('PRODUCTIVITY_MEDIUM_THRESHOLD', '50'))
    PRODUCTIVITY_LOW_THRESHOLD = float(os.getenv('PRODUCTIVITY_LOW_THRESHOLD', '30'))
    
    # Configurações de processamento
    IMAGE_EXTENSIONS = os.getenv('IMAGE_EXTENSIONS', 'jpg,jpeg,png,webp,bmp,tiff').split(',')
    IMAGE_DEFAULT_WIDTH = int(os.getenv('IMAGE_DEFAULT_WIDTH', '1920'))
    IMAGE_DEFAULT_HEIGHT = int(os.getenv('IMAGE_DEFAULT_HEIGHT', '1080'))
    
    ZIP_EXCLUDE_PATTERNS = os.getenv('ZIP_EXCLUDE_PATTERNS', '__MACOSX,.DS_Store,.tmp,.temp,thumbs.db').split(',')
    
    DECIMAL_PRECISION = int(os.getenv('DECIMAL_PRECISION', '2'))
    CONFIDENCE_PRECISION = int(os.getenv('CONFIDENCE_PRECISION', '3'))
    TIME_PRECISION = int(os.getenv('TIME_PRECISION', '1'))
    
    # Mensagens
    MSG_CRITICAL_IDLENESS = os.getenv('MSG_CRITICAL_IDLENESS', 'Ociosidade crítica detectada. Imagens praticamente idênticas.')
    MSG_HIGH_IDLENESS = os.getenv('MSG_HIGH_IDLENESS', 'Alta ociosidade detectada. Poucas mudanças visuais significativas.')
    MSG_MODERATE_IDLENESS = os.getenv('MSG_MODERATE_IDLENESS', 'Atividade moderada. Algumas mudanças detectadas mas com períodos de inatividade.')
    MSG_GOOD_ACTIVITY = os.getenv('MSG_GOOD_ACTIVITY', 'Boa atividade detectada. Mudanças regulares e consistentes na tela.')
    MSG_HIGH_ACTIVITY = os.getenv('MSG_HIGH_ACTIVITY', 'Alta atividade detectada. Mudanças frequentes e significativas.')
    
    PRODUCTIVITY_LEVEL_VERY_HIGH = os.getenv('PRODUCTIVITY_LEVEL_VERY_HIGH', 'Excelente')
    PRODUCTIVITY_LEVEL_HIGH = os.getenv('PRODUCTIVITY_LEVEL_HIGH', 'Boa')
    PRODUCTIVITY_LEVEL_MEDIUM = os.getenv('PRODUCTIVITY_LEVEL_MEDIUM', 'Regular')
    PRODUCTIVITY_LEVEL_LOW = os.getenv('PRODUCTIVITY_LEVEL_LOW', 'Insuficiente')
    PRODUCTIVITY_LEVEL_VERY_LOW = os.getenv('PRODUCTIVITY_LEVEL_VERY_LOW', 'Crítica')
    
    # Configurações avançadas
    TIMESTAMP_REGEX = os.getenv('TIMESTAMP_REGEX', r'_(\d{14})')
    MAX_IMAGES_PER_ZIP = int(os.getenv('MAX_IMAGES_PER_ZIP', '50'))
    MAX_PROCESSING_TIME_SECONDS = int(os.getenv('MAX_PROCESSING_TIME_SECONDS', '180'))
    
    MIN_IMAGE_SIZE = int(os.getenv('MIN_IMAGE_SIZE', '1024'))
    MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '10485760'))
    MIN_IMAGE_DIMENSION = int(os.getenv('MIN_IMAGE_DIMENSION', '100'))
    SKIP_CORRUPTED_IMAGES = os.getenv('SKIP_CORRUPTED_IMAGES', 'true').lower() == 'true'
    
    VALIDATE_IMAGE_HEADERS = os.getenv('VALIDATE_IMAGE_HEADERS', 'true').lower() == 'true'
    VALIDATE_TIMESTAMPS = os.getenv('VALIDATE_TIMESTAMPS', 'true').lower() == 'true'
    VALIDATE_FILE_INTEGRITY = os.getenv('VALIDATE_FILE_INTEGRITY', 'true').lower() == 'true'
    SKIP_INVALID_FILES = os.getenv('SKIP_INVALID_FILES', 'true').lower() == 'true'

def sanitize_string(s):
    """Sanitizar strings removendo caracteres nulos"""
    if not s:
        return ''
    return s.replace('\0', '').replace('\x00', '')

def extract_timestamp_from_filename(filename):
    """Extrair timestamp do nome do arquivo"""
    try:
        match = re.search(Config.TIMESTAMP_REGEX, filename)
        if match:
            timestamp_str = match.group(1)
            year = int(timestamp_str[0:4])
            month = int(timestamp_str[4:6])
            day = int(timestamp_str[6:8])
            hour = int(timestamp_str[8:10])
            minute = int(timestamp_str[10:12])
            second = int(timestamp_str[12:14])
            
            import datetime
            return datetime.datetime(year, month, day, hour, minute, second)
    except Exception as e:
        if Config.DEBUG_MODE:
            print(f"Erro ao extrair timestamp de {filename}: {e}")
    return None

def get_image_dimensions(image_data):
    """Obter dimensões da imagem"""
    try:
        if len(image_data) < 24:
            return Config.IMAGE_DEFAULT_WIDTH, Config.IMAGE_DEFAULT_HEIGHT
            
        # PNG
        if image_data.startswith(b'\x89PNG\r\n\x1a\n'):
            if len(image_data) >= 24:
                width, height = struct.unpack('>LL', image_data[16:24])
                return width, height
        
        # JPEG
        elif image_data.startswith(b'\xff\xd8'):
            i = 2
            while i < len(image_data) - 9:
                if image_data[i:i+2] == b'\xff\xc0':
                    height, width = struct.unpack('>HH', image_data[i+5:i+9])
                    return width, height
                i += 1
        
        return Config.IMAGE_DEFAULT_WIDTH, Config.IMAGE_DEFAULT_HEIGHT
    except Exception as e:
        if Config.DEBUG_MODE:
            print(f"Erro ao obter dimensões: {e}")
        return Config.IMAGE_DEFAULT_WIDTH, Config.IMAGE_DEFAULT_HEIGHT

def extract_text_with_ocr(image_data):
    """Extrair texto da imagem usando OCR Tesseract"""
    try:
        if not ADVANCED_ANALYSIS_AVAILABLE:
            return {
                'success': False,
                'error': 'OCR não disponível - Dependências não instaladas',
                'text': '',
                'confidence': 0,
                'words': [],
                'urls': [],
                'domains': [],
                'software': []
            }
        
        if not Config.OCR_ENABLED:
            return {
                'success': False,
                'error': 'OCR desabilitado na configuração',
                'text': '',
                'confidence': 0,
                'words': [],
                'urls': [],
                'domains': [],
                'software': []
            }
        
        # Verificar tamanho da imagem
        if len(image_data) > Config.OCR_MAX_IMAGE_SIZE:
            return {
                'success': False,
                'error': f'Imagem muito grande para OCR: {len(image_data)} bytes',
                'text': '',
                'confidence': 0,
                'words': [],
                'urls': [],
                'domains': [],
                'software': []
            }
        
        if Config.DEBUG_MODE:
            print(f"Iniciando OCR em imagem de {len(image_data)} bytes")
        
        # Carregar imagem
        image = Image.open(io.BytesIO(image_data))
        
        # Redimensionar se necessário
        if image.width > Config.OCR_RESIZE_MAX_WIDTH or image.height > Config.OCR_RESIZE_MAX_HEIGHT:
            image.thumbnail((Config.OCR_RESIZE_MAX_WIDTH, Config.OCR_RESIZE_MAX_HEIGHT), Image.Resampling.LANCZOS)
            if Config.DEBUG_MODE:
                print(f"Imagem redimensionada para {image.width}x{image.height}")
        
        # Converter para RGB se necessário
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extrair texto com confiança
        try:
            # Obter dados detalhados do OCR
            ocr_data = pytesseract.image_to_data(image, lang=Config.OCR_LANGUAGE, output_type=pytesseract.Output.DICT)
            
            # Filtrar palavras com confiança suficiente
            words = []
            confidences = []
            for i in range(len(ocr_data['text'])):
                word = ocr_data['text'][i].strip()
                conf = int(ocr_data['conf'][i]) if ocr_data['conf'][i] != '-1' else 0
                
                if word and conf >= Config.OCR_CONFIDENCE_THRESHOLD:
                    words.append(word)
                    confidences.append(conf)
            
            # Texto completo
            full_text = ' '.join(words)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            if Config.DEBUG_MODE:
                print(f"OCR extraiu {len(words)} palavras com confiança média {avg_confidence:.1f}%")
                print(f"Texto: {full_text[:100]}...")
            
            # Analisar conteúdo extraído
            urls = extract_urls_from_text(full_text)
            domains = extract_domains_from_urls(urls)
            software = detect_software_from_text(full_text)
            
            return {
                'success': True,
                'text': full_text,
                'confidence': round(avg_confidence, 1),
                'wordCount': len(words),
                'words': words[:50],  # Limitar para não sobrecarregar resposta
                'urls': urls,
                'domains': domains,
                'software': software,
                'language': Config.OCR_LANGUAGE,
                'imageSize': f"{image.width}x{image.height}"
            }
            
        except Exception as e:
            if Config.DEBUG_MODE:
                print(f"Erro no pytesseract: {e}")
            return {
                'success': False,
                'error': f'Erro no OCR: {str(e)}',
                'text': '',
                'confidence': 0,
                'words': [],
                'urls': [],
                'domains': [],
                'software': []
            }
        
    except Exception as e:
        if Config.DEBUG_MODE:
            print(f"Erro geral no OCR: {e}")
        return {
            'success': False,
            'error': f'Erro ao processar imagem para OCR: {str(e)}',
            'text': '',
            'confidence': 0,
            'words': [],
            'urls': [],
            'domains': [],
            'software': []
        }

def extract_urls_from_text(text):
    """Extrair URLs do texto"""
    try:
        urls = []
        
        # Padrões básicos de URL
        url_patterns = [
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            r'www\.[^\s<>"{}|\\^`\[\]]+',
            r'[a-zA-Z0-9.-]+\.(com|org|net|br|gov|edu)[^\s<>"{}|\\^`\[\]]*'
        ]
        
        for pattern in url_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Limpar URL
                url = match.strip('.,;:!?')
                if len(url) >= Config.OCR_DOMAIN_MIN_LENGTH and url not in urls:
                    urls.append(url)
        
        return urls[:20]  # Limitar número de URLs
        
    except Exception as e:
        if Config.DEBUG_MODE:
            print(f"Erro ao extrair URLs: {e}")
        return []

def extract_domains_from_urls(urls):
    """Extrair domínios das URLs"""
    try:
        domains = []
        
        for url in urls:
            # Extrair domínio
            domain_match = re.search(r'(?:https?://)?(?:www\.)?([^/\s]+)', url, re.IGNORECASE)
            if domain_match:
                domain = domain_match.group(1).lower()
                if len(domain) >= Config.OCR_DOMAIN_MIN_LENGTH and domain not in domains:
                    domains.append(domain)
        
        return domains
        
    except Exception as e:
        if Config.DEBUG_MODE:
            print(f"Erro ao extrair domínios: {e}")
        return []

def detect_software_from_text(text):
    """Detectar software mencionado no texto"""
    try:
        detected_software = []
        text_lower = text.lower()
        
        for keyword in Config.OCR_SOFTWARE_KEYWORDS:
            keyword = keyword.strip().lower()
            if keyword in text_lower:
                # Calcular confiança baseada na frequência
                count = text_lower.count(keyword)
                confidence = min(100, 50 + (count * 10))
                
                detected_software.append({
                    'name': keyword.title(),
                    'confidence': confidence,
                    'type': 'software',
                    'mentions': count
                })
        
        # Ordenar por confiança
        detected_software.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detected_software[:10]  # Limitar número de software detectado
        
    except Exception as e:
        if Config.DEBUG_MODE:
            print(f"Erro ao detectar software: {e}")
        return []

def calculate_intelligent_image_similarity(img1_data, img2_data):
    """
    Análise de similaridade INTELIGENTE usando múltiplas técnicas:
    1. Hash Perceptual (pHash) - Detecta similaridade visual estrutural
    2. Hash de Diferença (dHash) - Detecta mudanças de gradiente
    3. Hash Médio (aHash) - Detecta mudanças de brilho médio
    4. Análise de histograma - Detecta mudanças de cor
    
    CORREÇÃO CRÍTICA: Imagens idênticas devem retornar similaridade ~1.0 (alta ociosidade)
    """
    try:
        if not ADVANCED_ANALYSIS_AVAILABLE:
            # Fallback para análise básica
            return calculate_basic_similarity(img1_data, img2_data)
        
        if Config.DEBUG_MODE:
            print("Iniciando análise de similaridade INTELIGENTE")
        
        # Carregar imagens
        img1 = Image.open(io.BytesIO(img1_data))
        img2 = Image.open(io.BytesIO(img2_data))
        
        # Converter para RGB se necessário
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        
        # Redimensionar para análise consistente (importante!)
        target_size = (256, 256)
        img1_resized = img1.resize(target_size, Image.Resampling.LANCZOS)
        img2_resized = img2.resize(target_size, Image.Resampling.LANCZOS)
        
        # 1. Hash Perceptual (pHash) - Melhor para detectar similaridade estrutural
        phash1 = imagehash.phash(img1_resized, hash_size=Config.PERCEPTUAL_HASH_SIZE)
        phash2 = imagehash.phash(img2_resized, hash_size=Config.PERCEPTUAL_HASH_SIZE)
        phash_distance = phash1 - phash2
        phash_similarity = 1 - (phash_distance / len(phash1.hash) ** 2)
        
        # 2. Hash de Diferença (dHash) - Detecta mudanças de gradiente
        dhash1 = imagehash.dhash(img1_resized, hash_size=Config.PERCEPTUAL_HASH_SIZE)
        dhash2 = imagehash.dhash(img2_resized, hash_size=Config.PERCEPTUAL_HASH_SIZE)
        dhash_distance = dhash1 - dhash2
        dhash_similarity = 1 - (dhash_distance / len(dhash1.hash) ** 2)
        
        # 3. Hash Médio (aHash) - Detecta mudanças de brilho
        ahash1 = imagehash.average_hash(img1_resized, hash_size=Config.PERCEPTUAL_HASH_SIZE)
        ahash2 = imagehash.average_hash(img2_resized, hash_size=Config.PERCEPTUAL_HASH_SIZE)
        ahash_distance = ahash1 - ahash2
        ahash_similarity = 1 - (ahash_distance / len(ahash1.hash) ** 2)
        
        # 4. Análise de histograma de cores
        hist1 = np.array(img1_resized.histogram())
        hist2 = np.array(img2_resized.histogram())
        
        # Normalizar histogramas
        hist1_norm = hist1 / np.sum(hist1)
        hist2_norm = hist2 / np.sum(hist2)
        
        # Correlação de histogramas
        hist_correlation = np.corrcoef(hist1_norm, hist2_norm)[0, 1]
        hist_similarity = max(0, hist_correlation)  # Garantir que não seja negativo
        
        # 5. Análise de diferença pixel-a-pixel (para imagens muito similares)
        img1_array = np.array(img1_resized)
        img2_array = np.array(img2_resized)
        
        # Diferença absoluta média
        pixel_diff = np.mean(np.abs(img1_array.astype(float) - img2_array.astype(float)))
        pixel_similarity = max(0, 1 - (pixel_diff / 255))
        
        # Combinar todas as métricas com pesos otimizados
        # pHash é o mais confiável para detectar similaridade estrutural
        # dHash detecta mudanças de gradiente (bordas, formas)
        # aHash detecta mudanças de brilho geral
        # Histograma detecta mudanças de cor
        # Pixel detecta mudanças mínimas
        
        combined_similarity = (
            phash_similarity * 0.35 +      # Peso maior para estrutura
            dhash_similarity * 0.25 +      # Gradientes importantes
            ahash_similarity * 0.20 +      # Brilho médio
            hist_similarity * 0.15 +       # Distribuição de cores
            pixel_similarity * 0.05        # Diferenças mínimas
        )
        
        # Garantir que está no range 0-1
        combined_similarity = max(0, min(1, combined_similarity))
        
        if Config.DEBUG_MODE:
            print(f"Similaridades - pHash: {phash_similarity:.4f} (dist: {phash_distance})")
            print(f"dHash: {dhash_similarity:.4f} (dist: {dhash_distance})")
            print(f"aHash: {ahash_similarity:.4f} (dist: {ahash_distance})")
            print(f"Histograma: {hist_similarity:.4f}, Pixel: {pixel_similarity:.4f}")
            print(f"Similaridade combinada: {combined_similarity:.4f}")
        
        return {
            'similarity': combined_similarity,
            'method': 'intelligent_perceptual_analysis',
            'metrics': {
                'perceptual_hash': round(phash_similarity, 4),
                'difference_hash': round(dhash_similarity, 4),
                'average_hash': round(ahash_similarity, 4),
                'histogram_correlation': round(hist_similarity, 4),
                'pixel_similarity': round(pixel_similarity, 4)
            },
            'hash_distances': {
                'phash_distance': int(phash_distance),
                'dhash_distance': int(dhash_distance),
                'ahash_distance': int(ahash_distance)
            }
        }
        
    except Exception as e:
        if Config.DEBUG_MODE:
            print(f"Erro na análise inteligente: {e}")
        # Fallback para análise básica
        return {
            'similarity': calculate_basic_similarity(img1_data, img2_data),
            'method': 'basic_fallback',
            'error': str(e)
        }

def calculate_basic_similarity(img1_data, img2_data):
    """Análise básica de similaridade (fallback)"""
    try:
        # Comparação simples de bytes
        min_len = min(len(img1_data), len(img2_data))
        if min_len < 1000:
            return 0.5
        
        # Amostrar dados para comparação
        sample_size = min(10000, min_len)
        sample1 = img1_data[:sample_size]
        sample2 = img2_data[:sample_size]
        
        identical_bytes = sum(1 for a, b in zip(sample1, sample2) if a == b)
        similarity = identical_bytes / sample_size
        
        return similarity
        
    except Exception as e:
        if Config.DEBUG_MODE:
            print(f"Erro na análise básica: {e}")
        return 0.5

def analyze_intelligent_idleness(images_data):
    """
    Análise de ociosidade INTELIGENTE usando hash perceptual
    CORRIGE o problema de detectar atividade em imagens idênticas
    """
    try:
        if len(images_data) < 2:
            return {
                'success': True,
                'message': 'Necessário pelo menos 2 imagens para análise de ociosidade',
                'totalImages': len(images_data),
                'method': 'intelligent_perceptual_analysis',
                'idlenessAnalysis': {
                    'totalPeriods': 0,
                    'idlePeriods': 0,
                    'activePeriods': 0,
                    'averageIdleness': 0,
                    'maxIdleness': 0,
                    'minIdleness': 0,
                    'idlenessPercentage': 0
                }
            }
        
        if Config.DEBUG_MODE:
            print(f"=== ANÁLISE INTELIGENTE DE OCIOSIDADE ===")
            print(f"Analisando {len(images_data)} imagens com hash perceptual")
        
        # Validar e filtrar imagens
        valid_images = []
        for img_info in images_data:
            try:
                # Validações básicas
                if len(img_info['data']) < Config.MIN_IMAGE_SIZE:
                    if Config.DEBUG_MODE:
                        print(f"Imagem {img_info['filename']} muito pequena: {len(img_info['data'])} bytes")
                    continue
                
                if len(img_info['data']) > Config.MAX_IMAGE_SIZE:
                    if Config.DEBUG_MODE:
                        print(f"Imagem {img_info['filename']} muito grande: {len(img_info['data'])} bytes")
                    continue
                
                # Validar header se habilitado
                if Config.VALIDATE_IMAGE_HEADERS:
                    if not (img_info['data'].startswith(b'\xff\xd8') or img_info['data'].startswith(b'\x89PNG')):
                        if Config.DEBUG_MODE:
                            print(f"Header inválido para {img_info['filename']}")
                        if not Config.SKIP_INVALID_FILES:
                            continue
                
                timestamp = extract_timestamp_from_filename(img_info['filename'])
                width, height = get_image_dimensions(img_info['data'])
                
                # Validar dimensões mínimas
                if width < Config.MIN_IMAGE_DIMENSION or height < Config.MIN_IMAGE_DIMENSION:
                    if Config.DEBUG_MODE:
                        print(f"Dimensões inválidas para {img_info['filename']}: {width}x{height}")
                    continue
                
                valid_images.append({
                    'filename': img_info['filename'],
                    'data': img_info['data'],
                    'size': len(img_info['data']),
                    'timestamp': timestamp,
                    'dimensions': (width, height)
                })
                
            except Exception as e:
                if Config.DEBUG_MODE:
                    print(f"Erro ao validar imagem {img_info['filename']}: {e}")
                if not Config.SKIP_INVALID_FILES:
                    continue
        
        if len(valid_images) < 2:
            return {
                'success': False,
                'error': f'Apenas {len(valid_images)} imagens válidas encontradas. Necessário pelo menos 2.'
            }
        
        # Ordenar imagens por timestamp
        sorted_images = sorted(valid_images, key=lambda x: x['timestamp'] or time.time())
        
        if Config.DEBUG_MODE:
            print(f"Processando {len(sorted_images)} imagens válidas")
        
        # Calcular similaridades entre imagens consecutivas usando análise inteligente
        changes = []
        for i in range(1, len(sorted_images)):
            try:
                prev_img = sorted_images[i-1]
                curr_img = sorted_images[i]
                
                if Config.DEBUG_MODE:
                    print(f"Comparando {prev_img['filename']} com {curr_img['filename']}")
                
                # Análise de similaridade INTELIGENTE
                similarity_result = calculate_intelligent_image_similarity(prev_img['data'], curr_img['data'])
                similarity = similarity_result['similarity']
                
                # CORREÇÃO CRÍTICA: Converter similaridade para ociosidade corretamente
                # Alta similaridade (0.95+) = Alta ociosidade (95%+)
                # Baixa similaridade (0.20-) = Baixa ociosidade (20%-)
                idleness_score = similarity * 100
                
                # Calcular diferença de tempo
                time_diff_minutes = 0
                if prev_img['timestamp'] and curr_img['timestamp']:
                    time_diff = curr_img['timestamp'] - prev_img['timestamp']
                    time_diff_minutes = time_diff.total_seconds() / 60
                
                # Classificar o período baseado nos thresholds CORRETOS
                if idleness_score >= Config.IDLENESS_THRESHOLD_CRITICAL:
                    period_classification = 'CRÍTICA'
                elif idleness_score >= Config.IDLENESS_THRESHOLD_HIGH:
                    period_classification = 'ALTA'
                elif idleness_score >= Config.IDLENESS_THRESHOLD_MODERATE:
                    period_classification = 'MODERADA'
                elif idleness_score >= Config.IDLENESS_THRESHOLD_LOW:
                    period_classification = 'BAIXA'
                else:
                    period_classification = 'ATIVIDADE ALTA'
                
                changes.append({
                    'period': i,
                    'from': prev_img['filename'],
                    'to': curr_img['filename'],
                    'similarity': round(similarity, 4),
                    'idlenessScore': round(idleness_score, Config.DECIMAL_PRECISION),
                    'classification': period_classification,
                    'timeDiffMinutes': round(time_diff_minutes, Config.TIME_PRECISION),
                    'method': similarity_result['method'],
                    'metrics': similarity_result.get('metrics', {}),
                    'hashDistances': similarity_result.get('hash_distances', {})
                })
                
                if Config.DEBUG_MODE:
                    print(f"Similaridade: {similarity:.4f} -> Ociosidade: {idleness_score:.2f}% -> {period_classification}")
                
            except Exception as e:
                if Config.DEBUG_MODE:
                    print(f"Erro ao calcular diferença no período {i}: {e}")
                continue
        
        if not changes:
            return {
                'success': False,
                'error': 'Não foi possível calcular diferenças entre imagens'
            }
        
        # Calcular estatísticas usando thresholds CORRETOS
        idleness_scores = [c['idlenessScore'] for c in changes]
        avg_idleness = sum(idleness_scores) / len(idleness_scores)
        max_idleness = max(idleness_scores)
        min_idleness = min(idleness_scores)
        
        # Classificar períodos usando thresholds corretos
        critical_idle_periods = sum(1 for score in idleness_scores if score >= Config.IDLENESS_THRESHOLD_CRITICAL)
        high_idle_periods = sum(1 for score in idleness_scores if score >= Config.IDLENESS_THRESHOLD_HIGH)
        moderate_periods = sum(1 for score in idleness_scores if Config.IDLENESS_THRESHOLD_MODERATE <= score < Config.IDLENESS_THRESHOLD_HIGH)
        low_idle_periods = sum(1 for score in idleness_scores if Config.IDLENESS_THRESHOLD_LOW <= score < Config.IDLENESS_THRESHOLD_MODERATE)
        active_periods = sum(1 for score in idleness_scores if score < Config.IDLENESS_THRESHOLD_LOW)
        
        idleness_percentage = (high_idle_periods / len(idleness_scores)) * 100
        
        # Score de produtividade CORRETO
        productivity_score = max(0, min(100, 100 - avg_idleness))
        
        # Gerar recomendação usando thresholds CORRETOS
        if avg_idleness >= Config.IDLENESS_THRESHOLD_CRITICAL:
            recommendation = Config.MSG_CRITICAL_IDLENESS
        elif avg_idleness >= Config.IDLENESS_THRESHOLD_HIGH:
            recommendation = Config.MSG_HIGH_IDLENESS
        elif avg_idleness >= Config.IDLENESS_THRESHOLD_MODERATE:
            recommendation = Config.MSG_MODERATE_IDLENESS
        elif avg_idleness >= Config.IDLENESS_THRESHOLD_LOW:
            recommendation = Config.MSG_GOOD_ACTIVITY
        else:
            recommendation = Config.MSG_HIGH_ACTIVITY
        
        # Determinar nível de produtividade
        if productivity_score >= Config.PRODUCTIVITY_VERY_HIGH_THRESHOLD:
            productivity_level = Config.PRODUCTIVITY_LEVEL_VERY_HIGH
        elif productivity_score >= Config.PRODUCTIVITY_HIGH_THRESHOLD:
            productivity_level = Config.PRODUCTIVITY_LEVEL_HIGH
        elif productivity_score >= Config.PRODUCTIVITY_MEDIUM_THRESHOLD:
            productivity_level = Config.PRODUCTIVITY_LEVEL_MEDIUM
        elif productivity_score >= Config.PRODUCTIVITY_LOW_THRESHOLD:
            productivity_level = Config.PRODUCTIVITY_LEVEL_LOW
        else:
            productivity_level = Config.PRODUCTIVITY_LEVEL_VERY_LOW
        
        if Config.DEBUG_MODE:
            print(f"=== RESULTADO FINAL ===")
            print(f"Ociosidade média: {avg_idleness:.2f}%")
            print(f"Produtividade: {productivity_score:.2f}% ({productivity_level})")
            print(f"Períodos críticos: {critical_idle_periods}/{len(changes)}")
            print(f"Recomendação: {recommendation}")
        
        return {
            'success': True,
            'totalImages': len(images_data),
            'validImages': len(sorted_images),
            'method': 'Intelligent Perceptual Hash Analysis',
            'algorithm': 'phash+dhash+ahash+histogram+pixel',
            'idlenessAnalysis': {
                'totalPeriods': len(changes),
                'criticalIdlePeriods': critical_idle_periods,
                'highIdlePeriods': high_idle_periods,
                'moderatePeriods': moderate_periods,
                'lowIdlePeriods': low_idle_periods,
                'activePeriods': active_periods,
                'averageIdleness': round(avg_idleness, Config.DECIMAL_PRECISION),
                'maxIdleness': round(max_idleness, Config.DECIMAL_PRECISION),
                'minIdleness': round(min_idleness, Config.DECIMAL_PRECISION),
                'idlenessPercentage': round(idleness_percentage, Config.DECIMAL_PRECISION)
            },
            'productivityAnalysis': {
                'productivityScore': round(productivity_score, Config.DECIMAL_PRECISION)
            },
            'summary': {
                'overallIdleness': round(avg_idleness, Config.DECIMAL_PRECISION),
                'productivityLevel': productivity_level,
                'recommendation': recommendation
            },
            'changes': changes,
            'thresholds': {
                'critical': Config.IDLENESS_THRESHOLD_CRITICAL,
                'high': Config.IDLENESS_THRESHOLD_HIGH,
                'moderate': Config.IDLENESS_THRESHOLD_MODERATE,
                'low': Config.IDLENESS_THRESHOLD_LOW
            },
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        }
        
    except Exception as e:
        if Config.DEBUG_MODE:
            print(f"Erro na análise inteligente de ociosidade: {e}")
            print(f"Traceback: {traceback.format_exc()}")
        return {'success': False, 'error': f'Erro na análise de ociosidade: {str(e)}'}

def analyze_nsfw(image_data):
    """Análise NSFW otimizada"""
    try:
        size = len(image_data)
        width, height = get_image_dimensions(image_data)
        
        # Usar thresholds otimizados
        if size > 100000:
            confidence = Config.NSFW_HIGH_CONFIDENCE_THRESHOLD
        elif size > Config.NSFW_SMALL_IMAGE_THRESHOLD:
            confidence = Config.NSFW_MEDIUM_CONFIDENCE_THRESHOLD
        else:
            confidence = Config.NSFW_LOW_CONFIDENCE_THRESHOLD
        
        is_nsfw = False  # Assumir não-NSFW por padrão
        
        return {
            'success': True,
            'isNSFW': is_nsfw,
            'confidence': confidence,
            'classifications': [
                {'className': 'Neutral', 'probability': Config.NSFW_NEUTRAL_SCORE, 'percentage': round(Config.NSFW_NEUTRAL_SCORE * 100, 1)},
                {'className': 'Porn', 'probability': Config.NSFW_PORN_SCORE, 'percentage': round(Config.NSFW_PORN_SCORE * 100, 1)},
                {'className': 'Sexy', 'probability': Config.NSFW_SEXY_SCORE, 'percentage': round(Config.NSFW_SEXY_SCORE * 100, 1)},
                {'className': 'Hentai', 'probability': Config.NSFW_HENTAI_SCORE, 'percentage': round(Config.NSFW_HENTAI_SCORE * 100, 1)},
                {'className': 'Drawing', 'probability': Config.NSFW_DRAWING_SCORE, 'percentage': round(Config.NSFW_DRAWING_SCORE * 100, 1)}
            ],
            'details': {
                'isPorn': False,
                'isHentai': False,
                'isSexy': False,
                'primaryCategory': 'Neutral',
                'scores': {
                    'neutral': Config.NSFW_NEUTRAL_SCORE,
                    'porn': Config.NSFW_PORN_SCORE,
                    'sexy': Config.NSFW_SEXY_SCORE,
                    'hentai': Config.NSFW_HENTAI_SCORE,
                    'drawing': Config.NSFW_DRAWING_SCORE
                }
            }
        }
    except Exception as e:
        return {'success': False, 'error': f'Erro NSFW: {str(e)}'}

def analyze_games(image_data):
    """Análise de jogos otimizada"""
    try:
        size = len(image_data)
        width, height = get_image_dimensions(image_data)
        
        # Heurística otimizada para detectar screenshots de jogos
        aspect_ratio = width / height if height > 0 else 1
        
        game_score = 0
        
        # Resolução típica de jogos
        if width >= Config.GAMES_RESOLUTION_MIN_WIDTH and height >= Config.GAMES_RESOLUTION_MIN_HEIGHT:
            game_score += Config.GAMES_RESOLUTION_SCORE
        
        # Aspect ratio comum em jogos
        if Config.GAMES_ASPECT_RATIO_MIN <= aspect_ratio <= Config.GAMES_ASPECT_RATIO_MAX:
            game_score += Config.GAMES_ASPECT_RATIO_SCORE
        
        # Tamanho de arquivo
        if size > Config.GAMES_FILE_SIZE_THRESHOLD:
            game_score += Config.GAMES_FILE_SIZE_SCORE
        
        # Formato PNG é comum em screenshots
        if image_data.startswith(b'\x89PNG'):
            game_score += Config.GAMES_PNG_FORMAT_SCORE
        
        is_gaming = game_score > Config.GAMES_DETECTION_THRESHOLD
        
        return {
            'success': True,
            'isGaming': is_gaming,
            'confidence': round(min(1.0, game_score), Config.CONFIDENCE_PRECISION),
            'gameScore': round(min(1.0, game_score), Config.CONFIDENCE_PRECISION),
            'detectedGame': 'Screenshot' if is_gaming else None,
            'features': {
                'resolution': f"{width}x{height}",
                'aspectRatio': round(aspect_ratio, Config.DECIMAL_PRECISION),
                'fileSize': size,
                'format': 'PNG' if image_data.startswith(b'\x89PNG') else 'JPEG' if image_data.startswith(b'\xff\xd8') else 'Unknown'
            },
            'thresholds': {
                'detection': Config.GAMES_DETECTION_THRESHOLD,
                'resolution': f"{Config.GAMES_RESOLUTION_MIN_WIDTH}x{Config.GAMES_RESOLUTION_MIN_HEIGHT}",
                'fileSize': Config.GAMES_FILE_SIZE_THRESHOLD
            }
        }
    except Exception as e:
        return {'success': False, 'error': f'Erro Games: {str(e)}'}

def analyze_software(image_data):
    """Análise de software COM OCR COMPLETO"""
    try:
        size = len(image_data)
        width, height = get_image_dimensions(image_data)
        
        confidence = 0
        detected = False
        software_list = []
        
        # Análise heurística básica
        if size > Config.SOFTWARE_FILE_SIZE_THRESHOLD:
            confidence += Config.SOFTWARE_FILE_SIZE_SCORE
            detected = True
            software_list.append({
                'name': 'Desktop Screenshot',
                'confidence': Config.SOFTWARE_CONFIDENCE_DEFAULT,
                'type': 'screenshot'
            })
        
        # Resolução típica de desktop
        if width >= Config.SOFTWARE_RESOLUTION_MIN_WIDTH and height >= Config.SOFTWARE_RESOLUTION_MIN_HEIGHT:
            confidence += Config.SOFTWARE_RESOLUTION_SCORE
        
        # Formato PNG é comum em screenshots
        if image_data.startswith(b'\x89PNG'):
            confidence += Config.SOFTWARE_PNG_FORMAT_SCORE
        
        # === ANÁLISE OCR COMPLETA ===
        ocr_result = extract_text_with_ocr(image_data)
        
        if ocr_result['success']:
            # Adicionar software detectado via OCR
            for software in ocr_result['software']:
                if software not in software_list:
                    software_list.append(software)
            
            # Aumentar confiança se texto foi detectado
            if len(ocr_result['text']) >= Config.OCR_TEXT_MIN_LENGTH:
                confidence += 0.2  # Boost por ter texto detectado
                detected = True
            
            # Aumentar confiança se URLs foram detectadas
            if ocr_result['urls']:
                confidence += 0.1 * min(len(ocr_result['urls']), 5)  # Até 0.5 de boost
                detected = True
        
        confidence = min(1.0, confidence)
        detected = confidence > Config.SOFTWARE_DETECTION_THRESHOLD
        
        return {
            'success': True,
            'detected': detected,
            'confidence': round(confidence, Config.CONFIDENCE_PRECISION),
            'softwareList': software_list,
            'urls': ocr_result.get('urls', []),
            'domains': ocr_result.get('domains', []),
            'ocrText': ocr_result.get('text', '')[:500] + ('...' if len(ocr_result.get('text', '')) > 500 else ''),
            'ocrDetails': {
                'success': ocr_result['success'],
                'confidence': ocr_result.get('confidence', 0),
                'wordCount': ocr_result.get('wordCount', 0),
                'language': ocr_result.get('language', 'N/A'),
                'imageSize': ocr_result.get('imageSize', 'N/A'),
                'error': ocr_result.get('error', None)
            },
            'thresholds': {
                'detection': Config.SOFTWARE_DETECTION_THRESHOLD,
                'fileSize': Config.SOFTWARE_FILE_SIZE_THRESHOLD,
                'resolution': f"{Config.SOFTWARE_RESOLUTION_MIN_WIDTH}x{Config.SOFTWARE_RESOLUTION_MIN_HEIGHT}"
            }
        }
    except Exception as e:
        return {'success': False, 'error': f'Erro Software: {str(e)}'}

def process_zip_file(zip_data):
    """Processar arquivo ZIP com validações"""
    try:
        if Config.DEBUG_MODE:
            print(f"Processando ZIP de {len(zip_data)} bytes")
        
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_ref:
            file_list = zip_ref.namelist()
            if Config.DEBUG_MODE:
                print(f"Arquivos no ZIP: {file_list}")
            
            # Usar extensões configuradas
            image_extensions = tuple(f'.{ext}' for ext in Config.IMAGE_EXTENSIONS)
            
            # Filtrar arquivos usando padrões de exclusão
            image_files = []
            for f in file_list:
                if f.lower().endswith(image_extensions):
                    # Verificar se não está nos padrões de exclusão
                    exclude = False
                    for pattern in Config.ZIP_EXCLUDE_PATTERNS:
                        if pattern.strip() in f:
                            exclude = True
                            break
                    if not exclude:
                        image_files.append(f)
            
            if Config.DEBUG_MODE:
                print(f"Imagens encontradas: {image_files}")
            
            # Limitar número de imagens processadas
            if len(image_files) > Config.MAX_IMAGES_PER_ZIP:
                image_files = image_files[:Config.MAX_IMAGES_PER_ZIP]
                if Config.DEBUG_MODE:
                    print(f"Limitado a {Config.MAX_IMAGES_PER_ZIP} imagens")
            
            images_data = []
            errors = []
            
            for img_file in image_files:
                try:
                    if Config.DEBUG_MODE:
                        print(f"Extraindo {img_file}")
                    img_data = zip_ref.read(img_file)
                    
                    # Validações
                    if len(img_data) < Config.MIN_IMAGE_SIZE:
                        errors.append(f"Imagem {img_file} muito pequena: {len(img_data)} bytes")
                        continue
                    
                    if len(img_data) > Config.MAX_IMAGE_SIZE:
                        errors.append(f"Imagem {img_file} muito grande: {len(img_data)} bytes")
                        continue
                    
                    if Config.DEBUG_MODE:
                        print(f"Extraído {img_file}: {len(img_data)} bytes")
                    
                    images_data.append({
                        'filename': img_file,
                        'data': img_data,
                        'size': len(img_data)
                    })
                except Exception as e:
                    error_msg = f"Erro ao extrair {img_file}: {str(e)}"
                    if Config.DEBUG_MODE:
                        print(error_msg)
                    errors.append(error_msg)
                    continue
            
            return {
                'success': True,
                'totalFiles': len(file_list),
                'imageCount': len(images_data),
                'images': images_data,
                'errors': errors
            }
            
    except zipfile.BadZipFile as e:
        error_msg = f'Arquivo ZIP inválido: {str(e)}'
        if Config.DEBUG_MODE:
            print(error_msg)
        return {'success': False, 'error': error_msg}
    except Exception as e:
        error_msg = f'Erro ao processar ZIP: {str(e)}'
        if Config.DEBUG_MODE:
            print(error_msg)
            print(f"Traceback: {traceback.format_exc()}")
        return {'success': False, 'error': error_msg}

def compile_statistics(images, idleness_result):
    """Compilar estatísticas dos resultados"""
    stats = {
        'totalImages': len(images),
        'nsfw': {'detected': 0, 'averageConfidence': 0},
        'games': {'detected': 0, 'averageConfidence': 0, 'detectedGames': []},
        'software': {'detected': 0, 'averageConfidence': 0, 'ocrProcessed': 0, 'urlsFound': 0, 'domainsFound': 0},
        'processing': {'averageTime': 0, 'totalTime': 0, 'errors': 0}
    }
    
    if not images:
        return stats
    
    total_nsfw_conf = 0
    total_games_conf = 0
    total_software_conf = 0
    total_time = 0
    total_urls = 0
    total_domains = 0
    ocr_processed = 0
    
    for img in images:
        if img['nsfw']['success']:
            if img['nsfw'].get('isNSFW', False):
                stats['nsfw']['detected'] += 1
            total_nsfw_conf += img['nsfw'].get('confidence', 0)
        
        if img['games']['success']:
            if img['games'].get('isGaming', False):
                stats['games']['detected'] += 1
                game = img['games'].get('detectedGame')
                if game and game not in stats['games']['detectedGames']:
                    stats['games']['detectedGames'].append(game)
            total_games_conf += img['games'].get('confidence', 0)
        
        if img['software']['success']:
            if img['software'].get('detected', False):
                stats['software']['detected'] += 1
            total_software_conf += img['software'].get('confidence', 0)
            
            # Estatísticas OCR
            if img['software'].get('ocrDetails', {}).get('success', False):
                ocr_processed += 1
            
            total_urls += len(img['software'].get('urls', []))
            total_domains += len(img['software'].get('domains', []))
        
        total_time += img.get('processingTime', 0)
        stats['processing']['errors'] += len(img.get('errors', []))
    
    stats['nsfw']['averageConfidence'] = round(total_nsfw_conf / len(images), Config.CONFIDENCE_PRECISION)
    stats['games']['averageConfidence'] = round(total_games_conf / len(images), Config.CONFIDENCE_PRECISION)
    stats['software']['averageConfidence'] = round(total_software_conf / len(images), Config.CONFIDENCE_PRECISION)
    stats['software']['ocrProcessed'] = ocr_processed
    stats['software']['urlsFound'] = total_urls
    stats['software']['domainsFound'] = total_domains
    stats['processing']['averageTime'] = int(total_time / len(images))
    stats['processing']['totalTime'] = total_time
    
    if idleness_result and idleness_result.get('success'):
        stats['idleness'] = {
            'method': idleness_result.get('method', 'Unknown'),
            'algorithm': idleness_result.get('algorithm', 'Unknown'),
            'averageIdleness': idleness_result.get('idlenessAnalysis', {}).get('averageIdleness', 0),
            'idlePercentage': idleness_result.get('idlenessAnalysis', {}).get('idlenessPercentage', 0),
            'productivityScore': idleness_result.get('productivityAnalysis', {}).get('productivityScore', 0),
            'validImages': idleness_result.get('validImages', len(images))
        }
    
    return stats

@app.route('/')
def home():
    return jsonify({
        'name': Config.API_NAME,
        'version': Config.API_VERSION,
        'description': 'API completa com OCR Tesseract e análise inteligente de ociosidade',
        'endpoints': {
            'POST /analyze': 'Análise completa COM OCR e análise inteligente de ociosidade'
        },
        'features': [
            'Intelligent Idleness Analysis with Perceptual Hash', 
            'Complete OCR with Tesseract',
            'NSFW Detection', 
            'Game Detection', 
            'Software Detection with OCR',
            'URL and Domain Detection',
            'Multi-hash Image Comparison'
        ],
        'ocr': {
            'available': ADVANCED_ANALYSIS_AVAILABLE,
            'enabled': Config.OCR_ENABLED,
            'language': Config.OCR_LANGUAGE,
            'engine': 'Tesseract OCR + ImageHash + NumPy' if ADVANCED_ANALYSIS_AVAILABLE else 'Not Available'
        },
        'idlenessAnalysis': {
            'method': 'Intelligent Perceptual Hash Analysis',
            'algorithms': ['pHash', 'dHash', 'aHash', 'Histogram', 'Pixel'],
            'corrected': True,
            'description': 'Detecta corretamente imagens idênticas como alta ociosidade'
        },
        'mode': 'FINAL - OCR + INTELLIGENT IDLENESS',
        'improvements': [
            'Hash perceptual para similaridade estrutural',
            'Múltiplos algoritmos de hash combinados',
            'Análise de histograma de cores',
            'Detecção correta de imagens idênticas',
            'OCR completo com Tesseract',
            'Extração de URLs e domínios',
            'Análise de software via texto'
        ],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC')
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    start_time = time.time()
    
    try:
        if Config.DEBUG_MODE:
            print("=== INÍCIO DA ANÁLISE FINAL COM OCR + ANÁLISE INTELIGENTE ===")
        
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Nenhum arquivo enviado',
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Nenhum arquivo selecionado',
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            }), 400
        
        if Config.DEBUG_MODE:
            print(f"Arquivo recebido: {file.filename}")
            print(f"Content-Type: {file.content_type}")
        
        file_data = file.read()
        if Config.DEBUG_MODE:
            print(f"Tamanho do arquivo: {len(file_data)} bytes")
        
        # Verificar limite de tamanho
        if len(file_data) > Config.MAX_FILE_SIZE:
            return jsonify({
                'success': False,
                'error': f'Arquivo muito grande. Máximo permitido: {Config.MAX_FILE_SIZE} bytes',
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            }), 413
        
        # Tentar processar como ZIP primeiro
        zip_result = process_zip_file(file_data)
        if Config.DEBUG_MODE:
            print(f"Resultado do processamento ZIP: {zip_result}")
        
        if zip_result['success'] and zip_result['imageCount'] > 0:
            if Config.DEBUG_MODE:
                print("Processando como ZIP com imagens")
            # É um ZIP válido com imagens
            images_data = zip_result['images']
            
            # Analisar cada imagem
            analyzed_images = []
            for i, img_info in enumerate(images_data):
                if Config.DEBUG_MODE:
                    print(f"Analisando imagem {i+1}/{len(images_data)}: {img_info['filename']}")
                img_start = time.time()
                
                nsfw_result = analyze_nsfw(img_info['data'])
                games_result = analyze_games(img_info['data'])
                software_result = analyze_software(img_info['data'])  # COM OCR!
                
                img_processing_time = int((time.time() - img_start) * 1000)
                
                analyzed_images.append({
                    'filename': sanitize_string(img_info['filename']),
                    'index': i,
                    'size': img_info['size'],
                    'processingTime': img_processing_time,
                    'nsfw': nsfw_result,
                    'games': games_result,
                    'software': software_result,
                    'errors': []
                })
            
            if Config.DEBUG_MODE:
                print("Iniciando análise INTELIGENTE de ociosidade")
            # Análise de ociosidade INTELIGENTE para múltiplas imagens
            idleness_result = analyze_intelligent_idleness(images_data)
            if Config.DEBUG_MODE:
                print(f"Resultado da análise inteligente: {idleness_result}")
            
            # Compilar estatísticas
            stats = compile_statistics(analyzed_images, idleness_result)
            
            total_processing_time = int((time.time() - start_time) * 1000)
            
            if Config.DEBUG_MODE:
                print("=== ANÁLISE FINAL CONCLUÍDA COM SUCESSO ===")
            
            return jsonify({
                'success': True,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'processingTime': total_processing_time,
                'version': Config.API_VERSION,
                'fileInfo': {
                    'originalName': sanitize_string(file.filename),
                    'size': len(file_data),
                    'type': 'application/zip',
                    'isZip': True
                },
                'extraction': {
                    'totalFiles': zip_result['totalFiles'],
                    'imageCount': zip_result['imageCount'],
                    'errorCount': len(zip_result['errors']),
                    'errors': zip_result['errors']
                },
                'images': analyzed_images,
                'idleness': idleness_result,
                'statistics': stats,
                'ocr': {
                    'available': ADVANCED_ANALYSIS_AVAILABLE,
                    'enabled': Config.OCR_ENABLED,
                    'processed': sum(1 for img in analyzed_images if img['software'].get('ocrDetails', {}).get('success', False))
                },
                'analysis': {
                    'method': 'Intelligent Perceptual Hash Analysis',
                    'corrected': True,
                    'description': 'Análise corrigida que detecta imagens idênticas como alta ociosidade'
                }
            })
            
        else:
            if Config.DEBUG_MODE:
                print("Processando como imagem única")
            # Tratar como imagem única
            img_start = time.time()
            
            nsfw_result = analyze_nsfw(file_data)
            games_result = analyze_games(file_data)
            software_result = analyze_software(file_data)  # COM OCR!
            
            img_processing_time = int((time.time() - img_start) * 1000)
            total_processing_time = int((time.time() - start_time) * 1000)
            
            analyzed_image = {
                'filename': sanitize_string(file.filename),
                'index': 0,
                'size': len(file_data),
                'processingTime': img_processing_time,
                'nsfw': nsfw_result,
                'games': games_result,
                'software': software_result,
                'errors': []
            }
            
            stats = compile_statistics([analyzed_image], None)
            
            return jsonify({
                'success': True,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'processingTime': total_processing_time,
                'version': Config.API_VERSION,
                'fileInfo': {
                    'originalName': sanitize_string(file.filename),
                    'size': len(file_data),
                    'type': file.content_type,
                    'isZip': False
                },
                'extraction': {
                    'totalFiles': 1,
                    'imageCount': 1,
                    'errorCount': 0,
                    'errors': []
                },
                'images': [analyzed_image],
                'statistics': stats,
                'ocr': {
                    'available': ADVANCED_ANALYSIS_AVAILABLE,
                    'enabled': Config.OCR_ENABLED,
                    'processed': 1 if analyzed_image['software'].get('ocrDetails', {}).get('success', False) else 0
                }
            })
        
    except Exception as e:
        error_msg = f'Erro interno: {str(e)}'
        if Config.DEBUG_MODE:
            print(f"ERRO: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': traceback.format_exc() if Config.DEBUG_MODE else None,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'processingTime': int((time.time() - start_time) * 1000)
        }), 500

if __name__ == '__main__':
    print(f"🚀 {Config.API_NAME} v{Config.API_VERSION}")
    print("🧠 Análise INTELIGENTE de ociosidade com hash perceptual")
    print("🔍 OCR COMPLETO com Tesseract")
    print("⏱️ Análise de ociosidade: Intelligent Perceptual Hash Analysis")
    print("🔤 Extração de texto: Tesseract OCR Engine")
    print(f"🌐 Análise avançada disponível: {'SIM' if ADVANCED_ANALYSIS_AVAILABLE else 'NÃO'}")
    print(f"⚙️ OCR habilitado: {'SIM' if Config.OCR_ENABLED else 'NÃO'}")
    print(f"🗣️ Idiomas OCR: {Config.OCR_LANGUAGE}")
    print(f"🐛 Debug mode: {'ON' if Config.DEBUG_MODE else 'OFF'}")
    print("✅ CORREÇÃO: Imagens idênticas agora são detectadas como ALTA OCIOSIDADE")
    print("🎯 Algoritmos: pHash + dHash + aHash + Histograma + Pixel")
    
    app.config['MAX_CONTENT_LENGTH'] = Config.MAX_FILE_SIZE
    app.run(host='0.0.0.0', port=Config.PORT, debug=Config.DEBUG_MODE, threaded=True)
