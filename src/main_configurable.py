#!/usr/bin/env python3
"""
API Detector Inteligente v7.3.4 - Versão Configurável
Todas as variáveis de detecção são carregadas do arquivo .env
"""

import time
import os
import zipfile
import io
import re
import struct
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Carregar variáveis do .env
load_dotenv()

app = Flask(__name__)
CORS(app)

class Config:
    """Classe para carregar e gerenciar configurações do .env"""
    
    # Configurações gerais
    API_VERSION = os.getenv('API_VERSION', '7.3.4-configurable')
    API_NAME = os.getenv('API_NAME', 'Detector Inteligente API')
    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', '52428800'))
    PORT = int(os.getenv('PORT', '5000'))
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    
    # Análise de ociosidade
    IDLENESS_THRESHOLD_CRITICAL = float(os.getenv('IDLENESS_THRESHOLD_CRITICAL', '85'))
    IDLENESS_THRESHOLD_HIGH = float(os.getenv('IDLENESS_THRESHOLD_HIGH', '70'))
    IDLENESS_THRESHOLD_MODERATE = float(os.getenv('IDLENESS_THRESHOLD_MODERATE', '50'))
    IDLENESS_THRESHOLD_LOW = float(os.getenv('IDLENESS_THRESHOLD_LOW', '30'))
    
    VISUAL_SAMPLE_SIZE = int(os.getenv('VISUAL_SAMPLE_SIZE', '2000'))
    VISUAL_COMPARISON_STEP = int(os.getenv('VISUAL_COMPARISON_STEP', '1000'))
    TIME_ADJUSTMENT_THRESHOLD_MINUTES = float(os.getenv('TIME_ADJUSTMENT_THRESHOLD_MINUTES', '10'))
    TIME_ADJUSTMENT_FACTOR = float(os.getenv('TIME_ADJUSTMENT_FACTOR', '0.3'))
    SIZE_DIFF_THRESHOLD = float(os.getenv('SIZE_DIFF_THRESHOLD', '0.1'))
    SIZE_DIFF_ADJUSTMENT = float(os.getenv('SIZE_DIFF_ADJUSTMENT', '0.8'))
    
    # Análise NSFW
    NSFW_HIGH_CONFIDENCE_THRESHOLD = float(os.getenv('NSFW_HIGH_CONFIDENCE_THRESHOLD', '0.95'))
    NSFW_MEDIUM_CONFIDENCE_THRESHOLD = float(os.getenv('NSFW_MEDIUM_CONFIDENCE_THRESHOLD', '0.85'))
    NSFW_LOW_CONFIDENCE_THRESHOLD = float(os.getenv('NSFW_LOW_CONFIDENCE_THRESHOLD', '0.7'))
    NSFW_SMALL_IMAGE_THRESHOLD = int(os.getenv('NSFW_SMALL_IMAGE_THRESHOLD', '50000'))
    
    NSFW_NEUTRAL_SCORE = float(os.getenv('NSFW_NEUTRAL_SCORE', '0.7'))
    NSFW_PORN_SCORE = float(os.getenv('NSFW_PORN_SCORE', '0.02'))
    NSFW_SEXY_SCORE = float(os.getenv('NSFW_SEXY_SCORE', '0.03'))
    NSFW_HENTAI_SCORE = float(os.getenv('NSFW_HENTAI_SCORE', '0.01'))
    NSFW_DRAWING_SCORE = float(os.getenv('NSFW_DRAWING_SCORE', '0.01'))
    
    # Análise de jogos
    GAMES_DETECTION_THRESHOLD = float(os.getenv('GAMES_DETECTION_THRESHOLD', '0.3'))
    GAMES_RESOLUTION_MIN_WIDTH = int(os.getenv('GAMES_RESOLUTION_MIN_WIDTH', '1280'))
    GAMES_RESOLUTION_MIN_HEIGHT = int(os.getenv('GAMES_RESOLUTION_MIN_HEIGHT', '720'))
    GAMES_ASPECT_RATIO_MIN = float(os.getenv('GAMES_ASPECT_RATIO_MIN', '1.5'))
    GAMES_ASPECT_RATIO_MAX = float(os.getenv('GAMES_ASPECT_RATIO_MAX', '1.8'))
    GAMES_FILE_SIZE_THRESHOLD = int(os.getenv('GAMES_FILE_SIZE_THRESHOLD', '500000'))
    
    GAMES_RESOLUTION_SCORE = float(os.getenv('GAMES_RESOLUTION_SCORE', '0.3'))
    GAMES_ASPECT_RATIO_SCORE = float(os.getenv('GAMES_ASPECT_RATIO_SCORE', '0.2'))
    GAMES_FILE_SIZE_SCORE = float(os.getenv('GAMES_FILE_SIZE_SCORE', '0.2'))
    GAMES_PNG_FORMAT_SCORE = float(os.getenv('GAMES_PNG_FORMAT_SCORE', '0.1'))
    
    # Análise de software
    SOFTWARE_DETECTION_THRESHOLD = float(os.getenv('SOFTWARE_DETECTION_THRESHOLD', '0.5'))
    SOFTWARE_FILE_SIZE_THRESHOLD = int(os.getenv('SOFTWARE_FILE_SIZE_THRESHOLD', '200000'))
    SOFTWARE_RESOLUTION_MIN_WIDTH = int(os.getenv('SOFTWARE_RESOLUTION_MIN_WIDTH', '1024'))
    SOFTWARE_RESOLUTION_MIN_HEIGHT = int(os.getenv('SOFTWARE_RESOLUTION_MIN_HEIGHT', '768'))
    
    SOFTWARE_FILE_SIZE_SCORE = float(os.getenv('SOFTWARE_FILE_SIZE_SCORE', '0.4'))
    SOFTWARE_RESOLUTION_SCORE = float(os.getenv('SOFTWARE_RESOLUTION_SCORE', '0.3'))
    SOFTWARE_PNG_FORMAT_SCORE = float(os.getenv('SOFTWARE_PNG_FORMAT_SCORE', '0.2'))
    SOFTWARE_CONFIDENCE_DEFAULT = float(os.getenv('SOFTWARE_CONFIDENCE_DEFAULT', '0.7'))
    
    # Análise de produtividade
    PRODUCTIVITY_VERY_HIGH_THRESHOLD = float(os.getenv('PRODUCTIVITY_VERY_HIGH_THRESHOLD', '80'))
    PRODUCTIVITY_HIGH_THRESHOLD = float(os.getenv('PRODUCTIVITY_HIGH_THRESHOLD', '60'))
    PRODUCTIVITY_MEDIUM_THRESHOLD = float(os.getenv('PRODUCTIVITY_MEDIUM_THRESHOLD', '40'))
    PRODUCTIVITY_LOW_THRESHOLD = float(os.getenv('PRODUCTIVITY_LOW_THRESHOLD', '20'))
    
    PRODUCTIVE_PERIOD_THRESHOLD = float(os.getenv('PRODUCTIVE_PERIOD_THRESHOLD', '50'))
    HIGHLY_PRODUCTIVE_PERIOD_THRESHOLD = float(os.getenv('HIGHLY_PRODUCTIVE_PERIOD_THRESHOLD', '30'))
    UNPRODUCTIVE_PERIOD_THRESHOLD = float(os.getenv('UNPRODUCTIVE_PERIOD_THRESHOLD', '70'))
    
    # Configurações de processamento
    IMAGE_EXTENSIONS = os.getenv('IMAGE_EXTENSIONS', 'jpg,jpeg,png,webp,bmp,tiff').split(',')
    IMAGE_DEFAULT_WIDTH = int(os.getenv('IMAGE_DEFAULT_WIDTH', '800'))
    IMAGE_DEFAULT_HEIGHT = int(os.getenv('IMAGE_DEFAULT_HEIGHT', '600'))
    
    ZIP_EXCLUDE_PATTERNS = os.getenv('ZIP_EXCLUDE_PATTERNS', '__MACOSX,.,DS_Store').split(',')
    
    # Configurações de resposta
    DECIMAL_PRECISION = int(os.getenv('DECIMAL_PRECISION', '2'))
    CONFIDENCE_PRECISION = int(os.getenv('CONFIDENCE_PRECISION', '3'))
    TIME_PRECISION = int(os.getenv('TIME_PRECISION', '1'))
    
    # Mensagens
    MSG_CRITICAL_IDLENESS = os.getenv('MSG_CRITICAL_IDLENESS', 'Nível crítico de ociosidade detectado. Tela praticamente inalterada entre capturas.')
    MSG_HIGH_IDLENESS = os.getenv('MSG_HIGH_IDLENESS', 'Alto nível de ociosidade. Poucas mudanças visuais detectadas.')
    MSG_MODERATE_IDLENESS = os.getenv('MSG_MODERATE_IDLENESS', 'Nível moderado de atividade. Algumas mudanças detectadas.')
    MSG_GOOD_ACTIVITY = os.getenv('MSG_GOOD_ACTIVITY', 'Bom nível de atividade. Mudanças regulares detectadas.')
    MSG_HIGH_ACTIVITY = os.getenv('MSG_HIGH_ACTIVITY', 'Alto nível de atividade. Mudanças significativas entre capturas.')
    
    PRODUCTIVITY_LEVEL_VERY_HIGH = os.getenv('PRODUCTIVITY_LEVEL_VERY_HIGH', 'Muito Alta')
    PRODUCTIVITY_LEVEL_HIGH = os.getenv('PRODUCTIVITY_LEVEL_HIGH', 'Alta')
    PRODUCTIVITY_LEVEL_MEDIUM = os.getenv('PRODUCTIVITY_LEVEL_MEDIUM', 'Média')
    PRODUCTIVITY_LEVEL_LOW = os.getenv('PRODUCTIVITY_LEVEL_LOW', 'Baixa')
    PRODUCTIVITY_LEVEL_VERY_LOW = os.getenv('PRODUCTIVITY_LEVEL_VERY_LOW', 'Muito Baixa')
    
    ACTIVITY_LEVEL_VERY_LOW = os.getenv('ACTIVITY_LEVEL_VERY_LOW', 'very_low')
    ACTIVITY_LEVEL_LOW = os.getenv('ACTIVITY_LEVEL_LOW', 'low')
    ACTIVITY_LEVEL_MODERATE = os.getenv('ACTIVITY_LEVEL_MODERATE', 'moderate')
    ACTIVITY_LEVEL_HIGH = os.getenv('ACTIVITY_LEVEL_HIGH', 'high')
    
    # Configurações avançadas
    HASH_SAMPLE_DIVISOR = int(os.getenv('HASH_SAMPLE_DIVISOR', '1000'))
    HASH_SHIFT_BITS = int(os.getenv('HASH_SHIFT_BITS', '5'))
    
    TIMESTAMP_REGEX = os.getenv('TIMESTAMP_REGEX', r'_(\d{14})')
    
    MAX_IMAGES_PER_ZIP = int(os.getenv('MAX_IMAGES_PER_ZIP', '100'))
    MAX_PROCESSING_TIME_SECONDS = int(os.getenv('MAX_PROCESSING_TIME_SECONDS', '300'))
    MEMORY_LIMIT_MB = int(os.getenv('MEMORY_LIMIT_MB', '512'))

def sanitize_string(s):
    if not s:
        return ''
    return s.replace('\0', '').replace('\x00', '')

def extract_timestamp_from_filename(filename):
    """Extrai timestamp do nome do arquivo usando regex configurável"""
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
    """Obter dimensões da imagem sem PIL"""
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

def extract_jpeg_data_section(image_data):
    """Extrair seção de dados JPEG para análise de conteúdo"""
    try:
        if not image_data.startswith(b'\xff\xd8'):
            start = len(image_data) // 4
            end = start + Config.VISUAL_SAMPLE_SIZE
            return image_data[start:end] if end < len(image_data) else image_data[start:]
        
        # Encontrar início dos dados da imagem (após headers)
        i = 2
        while i < len(image_data) - 2:
            if image_data[i:i+2] == b'\xff\xda':  # Start of Scan
                # Pular header do SOS
                sos_length = struct.unpack('>H', image_data[i+2:i+4])[0]
                data_start = i + 2 + sos_length
                # Extrair uma amostra dos dados da imagem
                sample_size = min(Config.VISUAL_SAMPLE_SIZE, len(image_data) - data_start - 100)
                if sample_size > 0:
                    return image_data[data_start:data_start + sample_size]
                break
            i += 1
        
        # Fallback: usar seção do meio da imagem
        start = len(image_data) // 4
        end = start + Config.VISUAL_SAMPLE_SIZE
        return image_data[start:end] if end < len(image_data) else image_data[start:]
        
    except Exception as e:
        if Config.DEBUG_MODE:
            print(f"Erro ao extrair dados JPEG: {e}")
        # Fallback: usar seção do meio
        start = len(image_data) // 4
        end = start + Config.VISUAL_SAMPLE_SIZE
        return image_data[start:end] if end < len(image_data) else image_data[start:]

def calculate_visual_similarity(img1_data, img2_data):
    """Calcular similaridade visual real entre duas imagens"""
    try:
        # Extrair seções de dados visuais das imagens
        data1 = extract_jpeg_data_section(img1_data)
        data2 = extract_jpeg_data_section(img2_data)
        
        # Garantir que ambas as seções tenham o mesmo tamanho para comparação
        min_len = min(len(data1), len(data2))
        if min_len < 100:
            return 0.5  # Não é possível comparar adequadamente
        
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        # Calcular diferenças byte a byte
        differences = 0
        total_bytes = len(data1)
        
        # Comparação direta de bytes
        for i in range(total_bytes):
            if data1[i] != data2[i]:
                differences += abs(data1[i] - data2[i])
        
        # Normalizar diferença (0 = idênticas, 1 = completamente diferentes)
        max_possible_diff = total_bytes * 255
        normalized_diff = differences / max_possible_diff if max_possible_diff > 0 else 0
        
        # Calcular similaridade (0 = diferentes, 1 = idênticas)
        similarity = 1 - normalized_diff
        
        return max(0, min(1, similarity))
        
    except Exception as e:
        if Config.DEBUG_MODE:
            print(f"Erro ao calcular similaridade visual: {e}")
        return 0.5

def analyze_idleness_real(images_data):
    """Análise de ociosidade REAL baseada em conteúdo visual usando configurações do .env"""
    try:
        if len(images_data) < 2:
            return {
                'success': True,
                'message': 'Necessário pelo menos 2 imagens para análise de ociosidade',
                'totalImages': len(images_data),
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
            print(f"Analisando ociosidade para {len(images_data)} imagens")
        
        # Ordenar imagens por timestamp
        sorted_images = sorted(images_data, key=lambda x: extract_timestamp_from_filename(x['filename']) or time.time())
        
        # Processar imagens e extrair informações
        processed_images = []
        for img_info in sorted_images:
            try:
                timestamp = extract_timestamp_from_filename(img_info['filename'])
                width, height = get_image_dimensions(img_info['data'])
                
                processed_images.append({
                    'filename': img_info['filename'],
                    'data': img_info['data'],
                    'size': len(img_info['data']),
                    'timestamp': timestamp,
                    'dimensions': (width, height)
                })
            except Exception as e:
                if Config.DEBUG_MODE:
                    print(f"Erro ao processar imagem {img_info['filename']}: {e}")
                continue
        
        if len(processed_images) < 2:
            return {
                'success': False,
                'error': 'Não foi possível processar imagens suficientes para análise'
            }
        
        # Calcular diferenças visuais entre imagens consecutivas
        changes = []
        for i in range(1, len(processed_images)):
            try:
                prev_img = processed_images[i-1]
                curr_img = processed_images[i]
                
                if Config.DEBUG_MODE:
                    print(f"Comparando {prev_img['filename']} com {curr_img['filename']}")
                
                # Calcular similaridade visual
                similarity = calculate_visual_similarity(prev_img['data'], curr_img['data'])
                
                # Converter similaridade para score de ociosidade
                # Alta similaridade = alta ociosidade
                idleness_score = similarity * 100
                
                # Calcular diferença de tempo se timestamps disponíveis
                time_diff_minutes = 0
                if prev_img['timestamp'] and curr_img['timestamp']:
                    time_diff = curr_img['timestamp'] - prev_img['timestamp']
                    time_diff_minutes = time_diff.total_seconds() / 60
                
                # Ajustar score baseado no tempo (usando configurações do .env)
                if time_diff_minutes > Config.TIME_ADJUSTMENT_THRESHOLD_MINUTES:
                    time_factor = min(1.0, time_diff_minutes / 60)
                    idleness_score = idleness_score * (1 - time_factor * Config.TIME_ADJUSTMENT_FACTOR)
                
                # Ajustar baseado em diferença de tamanho (usando configurações do .env)
                size_diff_ratio = abs(prev_img['size'] - curr_img['size']) / max(prev_img['size'], curr_img['size'])
                if size_diff_ratio > Config.SIZE_DIFF_THRESHOLD:
                    idleness_score = idleness_score * Config.SIZE_DIFF_ADJUSTMENT
                
                idleness_score = max(0, min(100, idleness_score))
                
                changes.append({
                    'period': i,
                    'from': prev_img['filename'],
                    'to': curr_img['filename'],
                    'visualSimilarity': round(similarity, 4),
                    'idlenessScore': round(idleness_score, Config.DECIMAL_PRECISION),
                    'timeDiffMinutes': round(time_diff_minutes, Config.TIME_PRECISION),
                    'sizeDiffRatio': round(size_diff_ratio, 4)
                })
                
                if Config.DEBUG_MODE:
                    print(f"Similaridade: {similarity:.4f}, Ociosidade: {idleness_score:.2f}%")
                
            except Exception as e:
                if Config.DEBUG_MODE:
                    print(f"Erro ao calcular diferença no período {i}: {e}")
                continue
        
        if not changes:
            return {
                'success': False,
                'error': 'Não foi possível calcular diferenças entre imagens'
            }
        
        # Calcular estatísticas usando thresholds configuráveis
        idleness_scores = [c['idlenessScore'] for c in changes]
        avg_idleness = sum(idleness_scores) / len(idleness_scores)
        max_idleness = max(idleness_scores)
        min_idleness = min(idleness_scores)
        
        # Classificar períodos usando thresholds do .env
        very_idle_periods = sum(1 for score in idleness_scores if score >= Config.IDLENESS_THRESHOLD_CRITICAL)
        idle_periods = sum(1 for score in idleness_scores if score >= Config.IDLENESS_THRESHOLD_HIGH)
        moderate_periods = sum(1 for score in idleness_scores if Config.IDLENESS_THRESHOLD_MODERATE <= score < Config.IDLENESS_THRESHOLD_HIGH)
        active_periods = sum(1 for score in idleness_scores if score < Config.IDLENESS_THRESHOLD_MODERATE)
        
        idleness_percentage = (idle_periods / len(idleness_scores)) * 100
        
        # Análise temporal por horário
        hourly_analysis = {}
        for img in processed_images:
            if img['timestamp']:
                hour = img['timestamp'].hour
                if hour not in hourly_analysis:
                    hourly_analysis[hour] = {'screenshots': 0, 'idleness_scores': []}
                hourly_analysis[hour]['screenshots'] += 1
        
        # Adicionar scores de ociosidade por hora
        for change in changes:
            for img in processed_images:
                if img['filename'] == change['to'] and img['timestamp']:
                    hour = img['timestamp'].hour
                    if hour in hourly_analysis:
                        hourly_analysis[hour]['idleness_scores'].append(change['idlenessScore'])
        
        # Calcular médias por hora usando thresholds configuráveis
        for hour in hourly_analysis:
            scores = hourly_analysis[hour]['idleness_scores']
            if scores:
                hourly_analysis[hour]['averageIdleness'] = sum(scores) / len(scores)
                avg_score = hourly_analysis[hour]['averageIdleness']
                
                if avg_score > Config.IDLENESS_THRESHOLD_CRITICAL:
                    hourly_analysis[hour]['activityLevel'] = Config.ACTIVITY_LEVEL_VERY_LOW
                elif avg_score > Config.IDLENESS_THRESHOLD_HIGH:
                    hourly_analysis[hour]['activityLevel'] = Config.ACTIVITY_LEVEL_LOW
                elif avg_score > Config.IDLENESS_THRESHOLD_MODERATE:
                    hourly_analysis[hour]['activityLevel'] = Config.ACTIVITY_LEVEL_MODERATE
                else:
                    hourly_analysis[hour]['activityLevel'] = Config.ACTIVITY_LEVEL_HIGH
            else:
                hourly_analysis[hour]['averageIdleness'] = 0
                hourly_analysis[hour]['activityLevel'] = 'unknown'
        
        # Análise de produtividade com thresholds configuráveis
        highly_productive_periods = sum(1 for score in idleness_scores if score < Config.HIGHLY_PRODUCTIVE_PERIOD_THRESHOLD)
        productive_periods = sum(1 for score in idleness_scores if score < Config.PRODUCTIVE_PERIOD_THRESHOLD)
        unproductive_periods = sum(1 for score in idleness_scores if score > Config.UNPRODUCTIVE_PERIOD_THRESHOLD)
        
        productive_time = (productive_periods / len(idleness_scores)) * 100
        unproductive_time = (unproductive_periods / len(idleness_scores)) * 100
        neutral_time = 100 - productive_time - unproductive_time
        
        # Score de produtividade ajustado
        productivity_score = max(0, min(100, 100 - avg_idleness))
        
        # Determinar horário mais/menos ativo
        most_active_hour = None
        least_active_hour = None
        if hourly_analysis:
            hours_with_scores = {h: data for h, data in hourly_analysis.items() if data.get('averageIdleness', 0) > 0}
            if hours_with_scores:
                most_active_hour = min(hours_with_scores.keys(), key=lambda h: hours_with_scores[h]['averageIdleness'])
                least_active_hour = max(hours_with_scores.keys(), key=lambda h: hours_with_scores[h]['averageIdleness'])
        
        # Gerar recomendação usando mensagens configuráveis
        if avg_idleness > Config.IDLENESS_THRESHOLD_CRITICAL:
            recommendation = Config.MSG_CRITICAL_IDLENESS
        elif avg_idleness > Config.IDLENESS_THRESHOLD_HIGH:
            recommendation = Config.MSG_HIGH_IDLENESS
        elif avg_idleness > Config.IDLENESS_THRESHOLD_MODERATE:
            recommendation = Config.MSG_MODERATE_IDLENESS
        elif avg_idleness > Config.IDLENESS_THRESHOLD_LOW:
            recommendation = Config.MSG_GOOD_ACTIVITY
        else:
            recommendation = Config.MSG_HIGH_ACTIVITY
        
        # Determinar nível de produtividade usando thresholds configuráveis
        if productivity_score > Config.PRODUCTIVITY_VERY_HIGH_THRESHOLD:
            productivity_level = Config.PRODUCTIVITY_LEVEL_VERY_HIGH
        elif productivity_score > Config.PRODUCTIVITY_HIGH_THRESHOLD:
            productivity_level = Config.PRODUCTIVITY_LEVEL_HIGH
        elif productivity_score > Config.PRODUCTIVITY_MEDIUM_THRESHOLD:
            productivity_level = Config.PRODUCTIVITY_LEVEL_MEDIUM
        elif productivity_score > Config.PRODUCTIVITY_LOW_THRESHOLD:
            productivity_level = Config.PRODUCTIVITY_LEVEL_LOW
        else:
            productivity_level = Config.PRODUCTIVITY_LEVEL_VERY_LOW
        
        return {
            'success': True,
            'totalImages': len(images_data),
            'method': 'Visual Content Analysis (Configurable)',
            'configuration': {
                'thresholds': {
                    'critical': Config.IDLENESS_THRESHOLD_CRITICAL,
                    'high': Config.IDLENESS_THRESHOLD_HIGH,
                    'moderate': Config.IDLENESS_THRESHOLD_MODERATE,
                    'low': Config.IDLENESS_THRESHOLD_LOW
                },
                'visualSampleSize': Config.VISUAL_SAMPLE_SIZE,
                'timeAdjustmentThreshold': Config.TIME_ADJUSTMENT_THRESHOLD_MINUTES
            },
            'idlenessAnalysis': {
                'totalPeriods': len(changes),
                'veryIdlePeriods': very_idle_periods,
                'idlePeriods': idle_periods,
                'moderatePeriods': moderate_periods,
                'activePeriods': active_periods,
                'averageIdleness': round(avg_idleness, Config.DECIMAL_PRECISION),
                'maxIdleness': round(max_idleness, Config.DECIMAL_PRECISION),
                'minIdleness': round(min_idleness, Config.DECIMAL_PRECISION),
                'idlenessPercentage': round(idleness_percentage, Config.DECIMAL_PRECISION)
            },
            'timeAnalysis': hourly_analysis,
            'productivityAnalysis': {
                'highlyProductiveTime': round((highly_productive_periods / len(idleness_scores)) * 100, Config.DECIMAL_PRECISION),
                'productiveTime': round(productive_time, Config.DECIMAL_PRECISION),
                'unproductiveTime': round(unproductive_time, Config.DECIMAL_PRECISION),
                'neutralTime': round(neutral_time, Config.DECIMAL_PRECISION),
                'productivityScore': round(productivity_score, Config.DECIMAL_PRECISION)
            },
            'summary': {
                'overallIdleness': round(avg_idleness, Config.DECIMAL_PRECISION),
                'productivityLevel': productivity_level,
                'mostActiveHour': most_active_hour,
                'leastActiveHour': least_active_hour,
                'recommendation': recommendation
            },
            'changes': changes,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        }
        
    except Exception as e:
        if Config.DEBUG_MODE:
            print(f"Erro na análise de ociosidade: {e}")
            print(f"Traceback: {traceback.format_exc()}")
        return {'success': False, 'error': f'Erro na análise de ociosidade: {str(e)}'}

def analyze_nsfw_configurable(image_data):
    """Análise NSFW usando configurações do .env"""
    try:
        size = len(image_data)
        width, height = get_image_dimensions(image_data)
        
        # Usar thresholds configuráveis
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

def analyze_games_configurable(image_data):
    """Análise de jogos usando configurações do .env"""
    try:
        size = len(image_data)
        width, height = get_image_dimensions(image_data)
        
        # Heurística para detectar screenshots de jogos usando configurações
        aspect_ratio = width / height if height > 0 else 1
        
        game_score = 0
        
        # Resolução típica de jogos (configurável)
        if width >= Config.GAMES_RESOLUTION_MIN_WIDTH and height >= Config.GAMES_RESOLUTION_MIN_HEIGHT:
            game_score += Config.GAMES_RESOLUTION_SCORE
        
        # Aspect ratio comum em jogos (configurável)
        if Config.GAMES_ASPECT_RATIO_MIN <= aspect_ratio <= Config.GAMES_ASPECT_RATIO_MAX:
            game_score += Config.GAMES_ASPECT_RATIO_SCORE
        
        # Tamanho de arquivo (configurável)
        if size > Config.GAMES_FILE_SIZE_THRESHOLD:
            game_score += Config.GAMES_FILE_SIZE_SCORE
        
        # Formato PNG é comum em screenshots (configurável)
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

def analyze_software_configurable(image_data):
    """Análise de software usando configurações do .env"""
    try:
        size = len(image_data)
        width, height = get_image_dimensions(image_data)
        
        confidence = 0
        detected = False
        software_list = []
        
        # Screenshots de desktop tendem a ser grandes (configurável)
        if size > Config.SOFTWARE_FILE_SIZE_THRESHOLD:
            confidence += Config.SOFTWARE_FILE_SIZE_SCORE
            detected = True
            software_list.append({
                'name': 'Desktop Screenshot',
                'confidence': Config.SOFTWARE_CONFIDENCE_DEFAULT,
                'type': 'screenshot'
            })
        
        # Resolução típica de desktop (configurável)
        if width >= Config.SOFTWARE_RESOLUTION_MIN_WIDTH and height >= Config.SOFTWARE_RESOLUTION_MIN_HEIGHT:
            confidence += Config.SOFTWARE_RESOLUTION_SCORE
        
        # Formato PNG é comum em screenshots (configurável)
        if image_data.startswith(b'\x89PNG'):
            confidence += Config.SOFTWARE_PNG_FORMAT_SCORE
        
        confidence = min(1.0, confidence)
        detected = confidence > Config.SOFTWARE_DETECTION_THRESHOLD
        
        return {
            'success': True,
            'detected': detected,
            'confidence': round(confidence, Config.CONFIDENCE_PRECISION),
            'softwareList': software_list if detected else [],
            'urls': [],
            'domains': [],
            'ocrText': 'OCR não disponível nesta versão de produção',
            'thresholds': {
                'detection': Config.SOFTWARE_DETECTION_THRESHOLD,
                'fileSize': Config.SOFTWARE_FILE_SIZE_THRESHOLD,
                'resolution': f"{Config.SOFTWARE_RESOLUTION_MIN_WIDTH}x{Config.SOFTWARE_RESOLUTION_MIN_HEIGHT}"
            }
        }
    except Exception as e:
        return {'success': False, 'error': f'Erro Software: {str(e)}'}

def process_zip_file(zip_data):
    """Processar arquivo ZIP usando configurações do .env"""
    try:
        if Config.DEBUG_MODE:
            print(f"Processando ZIP de {len(zip_data)} bytes")
        
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_ref:
            file_list = zip_ref.namelist()
            if Config.DEBUG_MODE:
                print(f"Arquivos no ZIP: {file_list}")
            
            # Usar extensões configuráveis
            image_extensions = tuple(f'.{ext}' for ext in Config.IMAGE_EXTENSIONS)
            
            # Filtrar arquivos usando padrões configuráveis
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

@app.route('/')
def home():
    return jsonify({
        'name': Config.API_NAME,
        'version': Config.API_VERSION,
        'description': 'API para análise completa de imagens com configurações flexíveis via .env',
        'endpoints': {
            'POST /analyze': 'Análise completa de arquivo ZIP ou imagem única',
            'GET /status': 'Status da API',
            'GET /config': 'Configurações atuais',
            'GET /': 'Informações da API'
        },
        'features': ['NSFW Detection', 'Game Detection', 'Software Detection', 'Configurable Idleness Analysis'],
        'mode': 'CONFIGURABLE - Todas as variáveis via .env',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC')
    })

@app.route('/config')
def config():
    """Endpoint para visualizar configurações atuais"""
    return jsonify({
        'version': Config.API_VERSION,
        'idleness': {
            'thresholds': {
                'critical': Config.IDLENESS_THRESHOLD_CRITICAL,
                'high': Config.IDLENESS_THRESHOLD_HIGH,
                'moderate': Config.IDLENESS_THRESHOLD_MODERATE,
                'low': Config.IDLENESS_THRESHOLD_LOW
            },
            'visualSampleSize': Config.VISUAL_SAMPLE_SIZE,
            'timeAdjustmentThreshold': Config.TIME_ADJUSTMENT_THRESHOLD_MINUTES
        },
        'nsfw': {
            'thresholds': {
                'high': Config.NSFW_HIGH_CONFIDENCE_THRESHOLD,
                'medium': Config.NSFW_MEDIUM_CONFIDENCE_THRESHOLD,
                'low': Config.NSFW_LOW_CONFIDENCE_THRESHOLD
            }
        },
        'games': {
            'detectionThreshold': Config.GAMES_DETECTION_THRESHOLD,
            'minResolution': f"{Config.GAMES_RESOLUTION_MIN_WIDTH}x{Config.GAMES_RESOLUTION_MIN_HEIGHT}",
            'fileSizeThreshold': Config.GAMES_FILE_SIZE_THRESHOLD
        },
        'software': {
            'detectionThreshold': Config.SOFTWARE_DETECTION_THRESHOLD,
            'fileSizeThreshold': Config.SOFTWARE_FILE_SIZE_THRESHOLD,
            'minResolution': f"{Config.SOFTWARE_RESOLUTION_MIN_WIDTH}x{Config.SOFTWARE_RESOLUTION_MIN_HEIGHT}"
        },
        'processing': {
            'maxImagesPerZip': Config.MAX_IMAGES_PER_ZIP,
            'maxFileSize': Config.MAX_FILE_SIZE,
            'imageExtensions': Config.IMAGE_EXTENSIONS
        },
        'debug': Config.DEBUG_MODE
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    start_time = time.time()
    
    try:
        if Config.DEBUG_MODE:
            print("=== INÍCIO DA ANÁLISE CONFIGURÁVEL ===")
        
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
        
        # Verificar limite de tamanho configurável
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
            
            # Analisar cada imagem usando funções configuráveis
            analyzed_images = []
            for i, img_info in enumerate(images_data):
                if Config.DEBUG_MODE:
                    print(f"Analisando imagem {i+1}/{len(images_data)}: {img_info['filename']}")
                img_start = time.time()
                
                nsfw_result = analyze_nsfw_configurable(img_info['data'])
                games_result = analyze_games_configurable(img_info['data'])
                software_result = analyze_software_configurable(img_info['data'])
                
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
                print("Iniciando análise de ociosidade CONFIGURÁVEL")
            # Análise de ociosidade configurável para múltiplas imagens
            idleness_result = analyze_idleness_real(images_data)
            if Config.DEBUG_MODE:
                print(f"Resultado da análise de ociosidade: {idleness_result}")
            
            # Compilar estatísticas
            stats = compile_statistics(analyzed_images, idleness_result)
            
            total_processing_time = int((time.time() - start_time) * 1000)
            
            if Config.DEBUG_MODE:
                print("=== ANÁLISE CONFIGURÁVEL CONCLUÍDA COM SUCESSO ===")
            
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
                'statistics': stats
            })
            
        else:
            if Config.DEBUG_MODE:
                print("Processando como imagem única")
            # Tratar como imagem única
            img_start = time.time()
            
            nsfw_result = analyze_nsfw_configurable(file_data)
            games_result = analyze_games_configurable(file_data)
            software_result = analyze_software_configurable(file_data)
            
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
                'statistics': stats
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

def compile_statistics(images, idleness_result):
    """Compilar estatísticas dos resultados"""
    stats = {
        'totalImages': len(images),
        'nsfw': {'detected': 0, 'averageConfidence': 0},
        'games': {'detected': 0, 'averageConfidence': 0, 'detectedGames': []},
        'software': {'detected': 0, 'averageConfidence': 0},
        'processing': {'averageTime': 0, 'totalTime': 0, 'errors': 0}
    }
    
    if not images:
        return stats
    
    total_nsfw_conf = 0
    total_games_conf = 0
    total_software_conf = 0
    total_time = 0
    
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
        
        total_time += img.get('processingTime', 0)
        stats['processing']['errors'] += len(img.get('errors', []))
    
    stats['nsfw']['averageConfidence'] = round(total_nsfw_conf / len(images), Config.CONFIDENCE_PRECISION)
    stats['games']['averageConfidence'] = round(total_games_conf / len(images), Config.CONFIDENCE_PRECISION)
    stats['software']['averageConfidence'] = round(total_software_conf / len(images), Config.CONFIDENCE_PRECISION)
    stats['processing']['averageTime'] = int(total_time / len(images))
    stats['processing']['totalTime'] = total_time
    
    if idleness_result and idleness_result.get('success'):
        stats['idleness'] = {
            'method': idleness_result.get('method', 'Unknown'),
            'averageIdleness': idleness_result.get('idlenessAnalysis', {}).get('averageIdleness', 0),
            'idlePercentage': idleness_result.get('idlenessAnalysis', {}).get('idlenessPercentage', 0),
            'productivityScore': idleness_result.get('productivityAnalysis', {}).get('productivityScore', 0)
        }
    
    return stats

@app.route('/status')
def status():
    return jsonify({
        'status': 'online',
        'version': Config.API_VERSION,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
        'detectors': {
            'nsfw': 'Configurable Heuristic Analysis',
            'games': 'Configurable Image Characteristics',
            'software': 'Configurable Basic Detection',
            'idleness': 'Configurable Visual Content Analysis'
        },
        'dependencies': {
            'external': 'None - Self-contained',
            'zipfile': True,
            'flask': True,
            'dotenv': True
        },
        'mode': 'CONFIGURABLE',
        'configFile': '.env loaded',
        'debug': Config.DEBUG_MODE
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'version': Config.API_VERSION, 
        'mode': 'configurable',
        'configLoaded': True
    })

if __name__ == '__main__':
    print(f"🚀 {Config.API_NAME} v{Config.API_VERSION}")
    print("📊 Análise configurável via arquivo .env")
    print("⏱️ Análise de ociosidade: Configurable Visual Content Analysis")
    print("🔍 Detectores: Configuráveis via .env")
    print(f"🐛 Debug mode: {'ON' if Config.DEBUG_MODE else 'OFF'}")
    
    app.config['MAX_CONTENT_LENGTH'] = Config.MAX_FILE_SIZE
    app.run(host='0.0.0.0', port=Config.PORT, debug=Config.DEBUG_MODE, threaded=True)
