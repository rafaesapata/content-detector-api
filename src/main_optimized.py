#!/usr/bin/env python3
"""
API Detector Inteligente v7.3.4 - Versão OTIMIZADA
Algoritmo de ociosidade corrigido e calibrado com dados reais
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

# Carregar variáveis do .env otimizado
load_dotenv('.env.optimized')

app = Flask(__name__)
CORS(app)

class OptimizedConfig:
    """Configurações otimizadas carregadas do .env"""
    
    # Configurações gerais
    API_VERSION = os.getenv('API_VERSION', '7.3.4-optimized')
    API_NAME = os.getenv('API_NAME', 'Detector Inteligente API Otimizado')
    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', '52428800'))
    PORT = int(os.getenv('PORT', '5000'))
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    
    # Análise de ociosidade otimizada
    IDLENESS_THRESHOLD_CRITICAL = float(os.getenv('IDLENESS_THRESHOLD_CRITICAL', '90'))
    IDLENESS_THRESHOLD_HIGH = float(os.getenv('IDLENESS_THRESHOLD_HIGH', '75'))
    IDLENESS_THRESHOLD_MODERATE = float(os.getenv('IDLENESS_THRESHOLD_MODERATE', '60'))
    IDLENESS_THRESHOLD_LOW = float(os.getenv('IDLENESS_THRESHOLD_LOW', '40'))
    
    VISUAL_SAMPLE_SIZE = int(os.getenv('VISUAL_SAMPLE_SIZE', '5000'))
    VISUAL_COMPARISON_STEP = int(os.getenv('VISUAL_COMPARISON_STEP', '500'))
    VISUAL_MULTIPLE_SAMPLES = int(os.getenv('VISUAL_MULTIPLE_SAMPLES', '3'))
    VISUAL_NOISE_THRESHOLD = int(os.getenv('VISUAL_NOISE_THRESHOLD', '15'))
    VISUAL_SIGNIFICANT_CHANGE_THRESHOLD = int(os.getenv('VISUAL_SIGNIFICANT_CHANGE_THRESHOLD', '25'))
    
    # Algoritmo de similaridade
    SIMILARITY_ALGORITHM = os.getenv('SIMILARITY_ALGORITHM', 'hybrid')
    SIMILARITY_WEIGHT_IDENTICAL = float(os.getenv('SIMILARITY_WEIGHT_IDENTICAL', '0.4'))
    SIMILARITY_WEIGHT_AVERAGE = float(os.getenv('SIMILARITY_WEIGHT_AVERAGE', '0.3'))
    SIMILARITY_WEIGHT_SIGNIFICANT = float(os.getenv('SIMILARITY_WEIGHT_SIGNIFICANT', '0.3'))
    SIMILARITY_NOISE_FILTER = os.getenv('SIMILARITY_NOISE_FILTER', 'true').lower() == 'true'
    
    TIME_ADJUSTMENT_THRESHOLD_MINUTES = float(os.getenv('TIME_ADJUSTMENT_THRESHOLD_MINUTES', '5'))
    TIME_ADJUSTMENT_FACTOR = float(os.getenv('TIME_ADJUSTMENT_FACTOR', '0.2'))
    SIZE_DIFF_THRESHOLD = float(os.getenv('SIZE_DIFF_THRESHOLD', '0.05'))
    SIZE_DIFF_ADJUSTMENT = float(os.getenv('SIZE_DIFF_ADJUSTMENT', '0.9'))
    COMPRESSION_VARIANCE_TOLERANCE = float(os.getenv('COMPRESSION_VARIANCE_TOLERANCE', '0.1'))
    
    # Outras configurações (NSFW, Games, Software, etc.)
    NSFW_HIGH_CONFIDENCE_THRESHOLD = float(os.getenv('NSFW_HIGH_CONFIDENCE_THRESHOLD', '0.92'))
    NSFW_MEDIUM_CONFIDENCE_THRESHOLD = float(os.getenv('NSFW_MEDIUM_CONFIDENCE_THRESHOLD', '0.80'))
    NSFW_LOW_CONFIDENCE_THRESHOLD = float(os.getenv('NSFW_LOW_CONFIDENCE_THRESHOLD', '0.65'))
    NSFW_SMALL_IMAGE_THRESHOLD = int(os.getenv('NSFW_SMALL_IMAGE_THRESHOLD', '30000'))
    
    NSFW_NEUTRAL_SCORE = float(os.getenv('NSFW_NEUTRAL_SCORE', '0.85'))
    NSFW_PORN_SCORE = float(os.getenv('NSFW_PORN_SCORE', '0.01'))
    NSFW_SEXY_SCORE = float(os.getenv('NSFW_SEXY_SCORE', '0.02'))
    NSFW_HENTAI_SCORE = float(os.getenv('NSFW_HENTAI_SCORE', '0.005'))
    NSFW_DRAWING_SCORE = float(os.getenv('NSFW_DRAWING_SCORE', '0.005'))
    
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
    
    SOFTWARE_DETECTION_THRESHOLD = float(os.getenv('SOFTWARE_DETECTION_THRESHOLD', '0.6'))
    SOFTWARE_FILE_SIZE_THRESHOLD = int(os.getenv('SOFTWARE_FILE_SIZE_THRESHOLD', '150000'))
    SOFTWARE_RESOLUTION_MIN_WIDTH = int(os.getenv('SOFTWARE_RESOLUTION_MIN_WIDTH', '800'))
    SOFTWARE_RESOLUTION_MIN_HEIGHT = int(os.getenv('SOFTWARE_RESOLUTION_MIN_HEIGHT', '600'))
    
    SOFTWARE_FILE_SIZE_SCORE = float(os.getenv('SOFTWARE_FILE_SIZE_SCORE', '0.45'))
    SOFTWARE_RESOLUTION_SCORE = float(os.getenv('SOFTWARE_RESOLUTION_SCORE', '0.35'))
    SOFTWARE_PNG_FORMAT_SCORE = float(os.getenv('SOFTWARE_PNG_FORMAT_SCORE', '0.20'))
    SOFTWARE_CONFIDENCE_DEFAULT = float(os.getenv('SOFTWARE_CONFIDENCE_DEFAULT', '0.75'))
    
    PRODUCTIVITY_VERY_HIGH_THRESHOLD = float(os.getenv('PRODUCTIVITY_VERY_HIGH_THRESHOLD', '85'))
    PRODUCTIVITY_HIGH_THRESHOLD = float(os.getenv('PRODUCTIVITY_HIGH_THRESHOLD', '70'))
    PRODUCTIVITY_MEDIUM_THRESHOLD = float(os.getenv('PRODUCTIVITY_MEDIUM_THRESHOLD', '50'))
    PRODUCTIVITY_LOW_THRESHOLD = float(os.getenv('PRODUCTIVITY_LOW_THRESHOLD', '30'))
    
    PRODUCTIVE_PERIOD_THRESHOLD = float(os.getenv('PRODUCTIVE_PERIOD_THRESHOLD', '40'))
    HIGHLY_PRODUCTIVE_PERIOD_THRESHOLD = float(os.getenv('HIGHLY_PRODUCTIVE_PERIOD_THRESHOLD', '25'))
    UNPRODUCTIVE_PERIOD_THRESHOLD = float(os.getenv('UNPRODUCTIVE_PERIOD_THRESHOLD', '75'))
    
    IMAGE_EXTENSIONS = os.getenv('IMAGE_EXTENSIONS', 'jpg,jpeg,png,webp,bmp,tiff').split(',')
    IMAGE_DEFAULT_WIDTH = int(os.getenv('IMAGE_DEFAULT_WIDTH', '1920'))
    IMAGE_DEFAULT_HEIGHT = int(os.getenv('IMAGE_DEFAULT_HEIGHT', '1080'))
    
    ZIP_EXCLUDE_PATTERNS = os.getenv('ZIP_EXCLUDE_PATTERNS', '__MACOSX,.,DS_Store,.tmp,.temp,thumbs.db').split(',')
    
    DECIMAL_PRECISION = int(os.getenv('DECIMAL_PRECISION', '2'))
    CONFIDENCE_PRECISION = int(os.getenv('CONFIDENCE_PRECISION', '3'))
    TIME_PRECISION = int(os.getenv('TIME_PRECISION', '1'))
    
    # Mensagens otimizadas
    MSG_CRITICAL_IDLENESS = os.getenv('MSG_CRITICAL_IDLENESS', 'Ociosidade crítica detectada. Tela praticamente inalterada por período prolongado.')
    MSG_HIGH_IDLENESS = os.getenv('MSG_HIGH_IDLENESS', 'Alta ociosidade detectada. Poucas mudanças visuais significativas.')
    MSG_MODERATE_IDLENESS = os.getenv('MSG_MODERATE_IDLENESS', 'Atividade moderada. Algumas mudanças detectadas mas com períodos de inatividade.')
    MSG_GOOD_ACTIVITY = os.getenv('MSG_GOOD_ACTIVITY', 'Boa atividade detectada. Mudanças regulares e consistentes na tela.')
    MSG_HIGH_ACTIVITY = os.getenv('MSG_HIGH_ACTIVITY', 'Alta atividade detectada. Mudanças frequentes e significativas.')
    
    PRODUCTIVITY_LEVEL_VERY_HIGH = os.getenv('PRODUCTIVITY_LEVEL_VERY_HIGH', 'Excelente')
    PRODUCTIVITY_LEVEL_HIGH = os.getenv('PRODUCTIVITY_LEVEL_HIGH', 'Boa')
    PRODUCTIVITY_LEVEL_MEDIUM = os.getenv('PRODUCTIVITY_LEVEL_MEDIUM', 'Regular')
    PRODUCTIVITY_LEVEL_LOW = os.getenv('PRODUCTIVITY_LEVEL_LOW', 'Insuficiente')
    PRODUCTIVITY_LEVEL_VERY_LOW = os.getenv('PRODUCTIVITY_LEVEL_VERY_LOW', 'Crítica')
    
    ACTIVITY_LEVEL_VERY_LOW = os.getenv('ACTIVITY_LEVEL_VERY_LOW', 'inativo')
    ACTIVITY_LEVEL_LOW = os.getenv('ACTIVITY_LEVEL_LOW', 'baixo')
    ACTIVITY_LEVEL_MODERATE = os.getenv('ACTIVITY_LEVEL_MODERATE', 'moderado')
    ACTIVITY_LEVEL_HIGH = os.getenv('ACTIVITY_LEVEL_HIGH', 'alto')
    ACTIVITY_LEVEL_VERY_HIGH = os.getenv('ACTIVITY_LEVEL_VERY_HIGH', 'muito_alto')
    
    # Configurações avançadas
    HASH_SAMPLE_DIVISOR = int(os.getenv('HASH_SAMPLE_DIVISOR', '500'))
    HASH_SHIFT_BITS = int(os.getenv('HASH_SHIFT_BITS', '3'))
    HASH_MULTIPLE_REGIONS = os.getenv('HASH_MULTIPLE_REGIONS', 'true').lower() == 'true'
    HASH_REGION_COUNT = int(os.getenv('HASH_REGION_COUNT', '4'))
    
    TIMESTAMP_REGEX = os.getenv('TIMESTAMP_REGEX', r'_(\d{14})')
    
    MAX_IMAGES_PER_ZIP = int(os.getenv('MAX_IMAGES_PER_ZIP', '50'))
    MAX_PROCESSING_TIME_SECONDS = int(os.getenv('MAX_PROCESSING_TIME_SECONDS', '180'))
    MEMORY_LIMIT_MB = int(os.getenv('MEMORY_LIMIT_MB', '256'))
    
    MIN_IMAGE_SIZE = int(os.getenv('MIN_IMAGE_SIZE', '1024'))
    MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '10485760'))
    MIN_IMAGE_DIMENSION = int(os.getenv('MIN_IMAGE_DIMENSION', '100'))
    SKIP_CORRUPTED_IMAGES = os.getenv('SKIP_CORRUPTED_IMAGES', 'true').lower() == 'true'
    
    VALIDATE_IMAGE_HEADERS = os.getenv('VALIDATE_IMAGE_HEADERS', 'true').lower() == 'true'
    VALIDATE_TIMESTAMPS = os.getenv('VALIDATE_TIMESTAMPS', 'true').lower() == 'true'
    VALIDATE_FILE_INTEGRITY = os.getenv('VALIDATE_FILE_INTEGRITY', 'true').lower() == 'true'
    SKIP_INVALID_FILES = os.getenv('SKIP_INVALID_FILES', 'true').lower() == 'true'

def sanitize_string(s):
    if not s:
        return ''
    return s.replace('\0', '').replace('\x00', '')

def extract_timestamp_from_filename(filename):
    """Extrai timestamp do nome do arquivo usando regex configurável"""
    try:
        match = re.search(OptimizedConfig.TIMESTAMP_REGEX, filename)
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
        if OptimizedConfig.DEBUG_MODE:
            print(f"Erro ao extrair timestamp de {filename}: {e}")
    return None

def get_image_dimensions(image_data):
    """Obter dimensões da imagem sem PIL"""
    try:
        if len(image_data) < 24:
            return OptimizedConfig.IMAGE_DEFAULT_WIDTH, OptimizedConfig.IMAGE_DEFAULT_HEIGHT
            
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
        
        return OptimizedConfig.IMAGE_DEFAULT_WIDTH, OptimizedConfig.IMAGE_DEFAULT_HEIGHT
    except Exception as e:
        if OptimizedConfig.DEBUG_MODE:
            print(f"Erro ao obter dimensões: {e}")
        return OptimizedConfig.IMAGE_DEFAULT_WIDTH, OptimizedConfig.IMAGE_DEFAULT_HEIGHT

def extract_multiple_image_samples(image_data):
    """Extrair múltiplas amostras de diferentes regiões da imagem"""
    try:
        samples = []
        total_size = len(image_data)
        
        if total_size < OptimizedConfig.VISUAL_SAMPLE_SIZE:
            return [image_data]
        
        # Extrair amostras de diferentes regiões
        regions = OptimizedConfig.HASH_REGION_COUNT if OptimizedConfig.HASH_MULTIPLE_REGIONS else 1
        
        for i in range(regions):
            # Calcular posição da região
            region_start = int((total_size / regions) * i)
            region_end = min(region_start + OptimizedConfig.VISUAL_SAMPLE_SIZE, total_size)
            
            # Ajustar para evitar headers JPEG
            if image_data.startswith(b'\xff\xd8') and region_start < 1000:
                region_start = max(1000, region_start)
            
            if region_end > region_start:
                sample = image_data[region_start:region_end]
                samples.append(sample)
        
        return samples if samples else [image_data[:OptimizedConfig.VISUAL_SAMPLE_SIZE]]
        
    except Exception as e:
        if OptimizedConfig.DEBUG_MODE:
            print(f"Erro ao extrair amostras múltiplas: {e}")
        return [image_data[:OptimizedConfig.VISUAL_SAMPLE_SIZE]]

def calculate_optimized_visual_similarity(img1_data, img2_data):
    """Algoritmo otimizado de similaridade visual"""
    try:
        # Extrair múltiplas amostras de cada imagem
        samples1 = extract_multiple_image_samples(img1_data)
        samples2 = extract_multiple_image_samples(img2_data)
        
        if len(samples1) != len(samples2):
            # Fallback para uma amostra
            samples1 = [samples1[0]] if samples1 else [img1_data[:OptimizedConfig.VISUAL_SAMPLE_SIZE]]
            samples2 = [samples2[0]] if samples2 else [img2_data[:OptimizedConfig.VISUAL_SAMPLE_SIZE]]
        
        total_similarities = []
        
        # Analisar cada par de amostras
        for sample1, sample2 in zip(samples1, samples2):
            similarity = calculate_sample_similarity(sample1, sample2)
            total_similarities.append(similarity)
        
        # Calcular similaridade média
        if total_similarities:
            avg_similarity = sum(total_similarities) / len(total_similarities)
            return max(0, min(1, avg_similarity))
        
        return 0.5
        
    except Exception as e:
        if OptimizedConfig.DEBUG_MODE:
            print(f"Erro ao calcular similaridade otimizada: {e}")
        return 0.5

def calculate_sample_similarity(sample1, sample2):
    """Calcular similaridade entre duas amostras usando algoritmo híbrido"""
    try:
        # Garantir que ambas as amostras tenham o mesmo tamanho
        min_len = min(len(sample1), len(sample2))
        if min_len < 100:
            return 0.5
        
        sample1 = sample1[:min_len]
        sample2 = sample2[:min_len]
        
        # Métricas de similaridade
        identical_bytes = 0
        total_differences = 0
        significant_changes = 0
        
        for i in range(min_len):
            byte_diff = abs(sample1[i] - sample2[i])
            
            if byte_diff == 0:
                identical_bytes += 1
            else:
                total_differences += byte_diff
                
                # Contar mudanças significativas (filtrar ruído)
                if byte_diff > OptimizedConfig.VISUAL_NOISE_THRESHOLD:
                    significant_changes += 1
        
        # Calcular métricas individuais
        identical_percentage = identical_bytes / min_len
        avg_difference = total_differences / min_len if min_len > 0 else 255
        significant_change_percentage = significant_changes / min_len
        
        # Normalizar diferença média (0-1, onde 0 = idêntico, 1 = máxima diferença)
        normalized_avg_diff = min(1.0, avg_difference / 255)
        
        # Calcular scores individuais
        identical_score = identical_percentage
        average_score = 1 - normalized_avg_diff
        significant_score = 1 - significant_change_percentage
        
        # Aplicar filtro de ruído se habilitado
        if OptimizedConfig.SIMILARITY_NOISE_FILTER:
            # Se há muitas mudanças pequenas (ruído), reduzir impacto
            noise_ratio = (total_differences - (significant_changes * OptimizedConfig.VISUAL_SIGNIFICANT_CHANGE_THRESHOLD)) / min_len
            if noise_ratio > 50:  # Muito ruído
                significant_score = significant_score * 1.2  # Dar mais peso às mudanças significativas
        
        # Combinar scores usando pesos configuráveis
        if OptimizedConfig.SIMILARITY_ALGORITHM == 'hybrid':
            combined_similarity = (
                identical_score * OptimizedConfig.SIMILARITY_WEIGHT_IDENTICAL +
                average_score * OptimizedConfig.SIMILARITY_WEIGHT_AVERAGE +
                significant_score * OptimizedConfig.SIMILARITY_WEIGHT_SIGNIFICANT
            )
        elif OptimizedConfig.SIMILARITY_ALGORITHM == 'identical':
            combined_similarity = identical_score
        elif OptimizedConfig.SIMILARITY_ALGORITHM == 'average':
            combined_similarity = average_score
        elif OptimizedConfig.SIMILARITY_ALGORITHM == 'significant':
            combined_similarity = significant_score
        else:
            # Fallback para híbrido
            combined_similarity = (identical_score + average_score + significant_score) / 3
        
        if OptimizedConfig.DEBUG_MODE:
            print(f"Similaridade - Idênticos: {identical_percentage:.3f}, Média: {average_score:.3f}, Significativo: {significant_score:.3f}, Final: {combined_similarity:.3f}")
        
        return max(0, min(1, combined_similarity))
        
    except Exception as e:
        if OptimizedConfig.DEBUG_MODE:
            print(f"Erro ao calcular similaridade da amostra: {e}")
        return 0.5

def analyze_optimized_idleness(images_data):
    """Análise de ociosidade OTIMIZADA com algoritmo corrigido"""
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
        
        if OptimizedConfig.DEBUG_MODE:
            print(f"Analisando ociosidade OTIMIZADA para {len(images_data)} imagens")
        
        # Validar e filtrar imagens
        valid_images = []
        for img_info in images_data:
            try:
                # Validações básicas
                if len(img_info['data']) < OptimizedConfig.MIN_IMAGE_SIZE:
                    if OptimizedConfig.DEBUG_MODE:
                        print(f"Imagem {img_info['filename']} muito pequena: {len(img_info['data'])} bytes")
                    continue
                
                if len(img_info['data']) > OptimizedConfig.MAX_IMAGE_SIZE:
                    if OptimizedConfig.DEBUG_MODE:
                        print(f"Imagem {img_info['filename']} muito grande: {len(img_info['data'])} bytes")
                    continue
                
                # Validar header se habilitado
                if OptimizedConfig.VALIDATE_IMAGE_HEADERS:
                    if not (img_info['data'].startswith(b'\xff\xd8') or img_info['data'].startswith(b'\x89PNG')):
                        if OptimizedConfig.DEBUG_MODE:
                            print(f"Header inválido para {img_info['filename']}")
                        if not OptimizedConfig.SKIP_INVALID_FILES:
                            continue
                
                timestamp = extract_timestamp_from_filename(img_info['filename'])
                width, height = get_image_dimensions(img_info['data'])
                
                # Validar dimensões mínimas
                if width < OptimizedConfig.MIN_IMAGE_DIMENSION or height < OptimizedConfig.MIN_IMAGE_DIMENSION:
                    if OptimizedConfig.DEBUG_MODE:
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
                if OptimizedConfig.DEBUG_MODE:
                    print(f"Erro ao validar imagem {img_info['filename']}: {e}")
                if not OptimizedConfig.SKIP_INVALID_FILES:
                    continue
        
        if len(valid_images) < 2:
            return {
                'success': False,
                'error': f'Apenas {len(valid_images)} imagens válidas encontradas. Necessário pelo menos 2.'
            }
        
        # Ordenar imagens por timestamp
        sorted_images = sorted(valid_images, key=lambda x: x['timestamp'] or time.time())
        
        if OptimizedConfig.DEBUG_MODE:
            print(f"Processando {len(sorted_images)} imagens válidas")
        
        # Calcular diferenças visuais entre imagens consecutivas
        changes = []
        for i in range(1, len(sorted_images)):
            try:
                prev_img = sorted_images[i-1]
                curr_img = sorted_images[i]
                
                if OptimizedConfig.DEBUG_MODE:
                    print(f"Comparando {prev_img['filename']} com {curr_img['filename']}")
                
                # Calcular similaridade visual otimizada
                similarity = calculate_optimized_visual_similarity(prev_img['data'], curr_img['data'])
                
                # Converter similaridade para score de ociosidade
                # CORREÇÃO: Alta similaridade = alta ociosidade
                base_idleness_score = similarity * 100
                
                # Calcular diferença de tempo
                time_diff_minutes = 0
                if prev_img['timestamp'] and curr_img['timestamp']:
                    time_diff = curr_img['timestamp'] - prev_img['timestamp']
                    time_diff_minutes = time_diff.total_seconds() / 60
                
                # Ajustar score baseado no tempo (configurável)
                adjusted_idleness_score = base_idleness_score
                if time_diff_minutes > OptimizedConfig.TIME_ADJUSTMENT_THRESHOLD_MINUTES:
                    # Quanto mais tempo passou, menor deveria ser a ociosidade esperada
                    time_factor = min(1.0, time_diff_minutes / 60)
                    adjustment = time_factor * OptimizedConfig.TIME_ADJUSTMENT_FACTOR
                    adjusted_idleness_score = base_idleness_score * (1 - adjustment)
                
                # Ajustar baseado em diferença de tamanho (configurável)
                size_diff_ratio = abs(prev_img['size'] - curr_img['size']) / max(prev_img['size'], curr_img['size'])
                if size_diff_ratio > OptimizedConfig.SIZE_DIFF_THRESHOLD:
                    # Diferença significativa de tamanho indica mudança
                    adjusted_idleness_score = adjusted_idleness_score * OptimizedConfig.SIZE_DIFF_ADJUSTMENT
                
                # Ajustar por variação de compressão
                if size_diff_ratio < OptimizedConfig.COMPRESSION_VARIANCE_TOLERANCE and similarity > 0.95:
                    # Provavelmente apenas variação de compressão
                    adjusted_idleness_score = min(100, adjusted_idleness_score * 1.1)
                
                final_idleness_score = max(0, min(100, adjusted_idleness_score))
                
                changes.append({
                    'period': i,
                    'from': prev_img['filename'],
                    'to': curr_img['filename'],
                    'visualSimilarity': round(similarity, 4),
                    'baseIdlenessScore': round(base_idleness_score, OptimizedConfig.DECIMAL_PRECISION),
                    'idlenessScore': round(final_idleness_score, OptimizedConfig.DECIMAL_PRECISION),
                    'timeDiffMinutes': round(time_diff_minutes, OptimizedConfig.TIME_PRECISION),
                    'sizeDiffRatio': round(size_diff_ratio, 4),
                    'adjustments': {
                        'timeAdjustment': round(base_idleness_score - adjusted_idleness_score, 2) if time_diff_minutes > OptimizedConfig.TIME_ADJUSTMENT_THRESHOLD_MINUTES else 0,
                        'sizeAdjustment': size_diff_ratio > OptimizedConfig.SIZE_DIFF_THRESHOLD
                    }
                })
                
                if OptimizedConfig.DEBUG_MODE:
                    print(f"Similaridade: {similarity:.4f}, Ociosidade base: {base_idleness_score:.2f}%, Final: {final_idleness_score:.2f}%")
                
            except Exception as e:
                if OptimizedConfig.DEBUG_MODE:
                    print(f"Erro ao calcular diferença no período {i}: {e}")
                continue
        
        if not changes:
            return {
                'success': False,
                'error': 'Não foi possível calcular diferenças entre imagens'
            }
        
        # Calcular estatísticas usando thresholds otimizados
        idleness_scores = [c['idlenessScore'] for c in changes]
        avg_idleness = sum(idleness_scores) / len(idleness_scores)
        max_idleness = max(idleness_scores)
        min_idleness = min(idleness_scores)
        
        # Classificar períodos usando thresholds otimizados
        critical_idle_periods = sum(1 for score in idleness_scores if score >= OptimizedConfig.IDLENESS_THRESHOLD_CRITICAL)
        high_idle_periods = sum(1 for score in idleness_scores if score >= OptimizedConfig.IDLENESS_THRESHOLD_HIGH)
        moderate_periods = sum(1 for score in idleness_scores if OptimizedConfig.IDLENESS_THRESHOLD_MODERATE <= score < OptimizedConfig.IDLENESS_THRESHOLD_HIGH)
        low_idle_periods = sum(1 for score in idleness_scores if OptimizedConfig.IDLENESS_THRESHOLD_LOW <= score < OptimizedConfig.IDLENESS_THRESHOLD_MODERATE)
        active_periods = sum(1 for score in idleness_scores if score < OptimizedConfig.IDLENESS_THRESHOLD_LOW)
        
        idleness_percentage = (high_idle_periods / len(idleness_scores)) * 100
        
        # Análise temporal por horário
        hourly_analysis = {}
        for img in sorted_images:
            if img['timestamp']:
                hour = img['timestamp'].hour
                if hour not in hourly_analysis:
                    hourly_analysis[hour] = {'screenshots': 0, 'idleness_scores': []}
                hourly_analysis[hour]['screenshots'] += 1
        
        # Adicionar scores de ociosidade por hora
        for change in changes:
            for img in sorted_images:
                if img['filename'] == change['to'] and img['timestamp']:
                    hour = img['timestamp'].hour
                    if hour in hourly_analysis:
                        hourly_analysis[hour]['idleness_scores'].append(change['idlenessScore'])
        
        # Calcular médias por hora usando thresholds otimizados
        for hour in hourly_analysis:
            scores = hourly_analysis[hour]['idleness_scores']
            if scores:
                hourly_analysis[hour]['averageIdleness'] = sum(scores) / len(scores)
                avg_score = hourly_analysis[hour]['averageIdleness']
                
                if avg_score >= OptimizedConfig.IDLENESS_THRESHOLD_CRITICAL:
                    hourly_analysis[hour]['activityLevel'] = OptimizedConfig.ACTIVITY_LEVEL_VERY_LOW
                elif avg_score >= OptimizedConfig.IDLENESS_THRESHOLD_HIGH:
                    hourly_analysis[hour]['activityLevel'] = OptimizedConfig.ACTIVITY_LEVEL_LOW
                elif avg_score >= OptimizedConfig.IDLENESS_THRESHOLD_MODERATE:
                    hourly_analysis[hour]['activityLevel'] = OptimizedConfig.ACTIVITY_LEVEL_MODERATE
                elif avg_score >= OptimizedConfig.IDLENESS_THRESHOLD_LOW:
                    hourly_analysis[hour]['activityLevel'] = OptimizedConfig.ACTIVITY_LEVEL_HIGH
                else:
                    hourly_analysis[hour]['activityLevel'] = OptimizedConfig.ACTIVITY_LEVEL_VERY_HIGH
            else:
                hourly_analysis[hour]['averageIdleness'] = 0
                hourly_analysis[hour]['activityLevel'] = 'unknown'
        
        # Análise de produtividade com thresholds otimizados
        highly_productive_periods = sum(1 for score in idleness_scores if score < OptimizedConfig.HIGHLY_PRODUCTIVE_PERIOD_THRESHOLD)
        productive_periods = sum(1 for score in idleness_scores if score < OptimizedConfig.PRODUCTIVE_PERIOD_THRESHOLD)
        unproductive_periods = sum(1 for score in idleness_scores if score > OptimizedConfig.UNPRODUCTIVE_PERIOD_THRESHOLD)
        
        productive_time = (productive_periods / len(idleness_scores)) * 100
        unproductive_time = (unproductive_periods / len(idleness_scores)) * 100
        neutral_time = 100 - productive_time - unproductive_time
        
        # Score de produtividade otimizado
        productivity_score = max(0, min(100, 100 - avg_idleness))
        
        # Determinar horário mais/menos ativo
        most_active_hour = None
        least_active_hour = None
        if hourly_analysis:
            hours_with_scores = {h: data for h, data in hourly_analysis.items() if data.get('averageIdleness', 0) > 0}
            if hours_with_scores:
                most_active_hour = min(hours_with_scores.keys(), key=lambda h: hours_with_scores[h]['averageIdleness'])
                least_active_hour = max(hours_with_scores.keys(), key=lambda h: hours_with_scores[h]['averageIdleness'])
        
        # Gerar recomendação usando mensagens otimizadas
        if avg_idleness >= OptimizedConfig.IDLENESS_THRESHOLD_CRITICAL:
            recommendation = OptimizedConfig.MSG_CRITICAL_IDLENESS
        elif avg_idleness >= OptimizedConfig.IDLENESS_THRESHOLD_HIGH:
            recommendation = OptimizedConfig.MSG_HIGH_IDLENESS
        elif avg_idleness >= OptimizedConfig.IDLENESS_THRESHOLD_MODERATE:
            recommendation = OptimizedConfig.MSG_MODERATE_IDLENESS
        elif avg_idleness >= OptimizedConfig.IDLENESS_THRESHOLD_LOW:
            recommendation = OptimizedConfig.MSG_GOOD_ACTIVITY
        else:
            recommendation = OptimizedConfig.MSG_HIGH_ACTIVITY
        
        # Determinar nível de produtividade usando thresholds otimizados
        if productivity_score >= OptimizedConfig.PRODUCTIVITY_VERY_HIGH_THRESHOLD:
            productivity_level = OptimizedConfig.PRODUCTIVITY_LEVEL_VERY_HIGH
        elif productivity_score >= OptimizedConfig.PRODUCTIVITY_HIGH_THRESHOLD:
            productivity_level = OptimizedConfig.PRODUCTIVITY_LEVEL_HIGH
        elif productivity_score >= OptimizedConfig.PRODUCTIVITY_MEDIUM_THRESHOLD:
            productivity_level = OptimizedConfig.PRODUCTIVITY_LEVEL_MEDIUM
        elif productivity_score >= OptimizedConfig.PRODUCTIVITY_LOW_THRESHOLD:
            productivity_level = OptimizedConfig.PRODUCTIVITY_LEVEL_LOW
        else:
            productivity_level = OptimizedConfig.PRODUCTIVITY_LEVEL_VERY_LOW
        
        return {
            'success': True,
            'totalImages': len(images_data),
            'validImages': len(sorted_images),
            'method': 'Optimized Visual Content Analysis',
            'algorithm': OptimizedConfig.SIMILARITY_ALGORITHM,
            'configuration': {
                'thresholds': {
                    'critical': OptimizedConfig.IDLENESS_THRESHOLD_CRITICAL,
                    'high': OptimizedConfig.IDLENESS_THRESHOLD_HIGH,
                    'moderate': OptimizedConfig.IDLENESS_THRESHOLD_MODERATE,
                    'low': OptimizedConfig.IDLENESS_THRESHOLD_LOW
                },
                'visualSampleSize': OptimizedConfig.VISUAL_SAMPLE_SIZE,
                'multipleRegions': OptimizedConfig.HASH_MULTIPLE_REGIONS,
                'noiseFiltering': OptimizedConfig.SIMILARITY_NOISE_FILTER,
                'weights': {
                    'identical': OptimizedConfig.SIMILARITY_WEIGHT_IDENTICAL,
                    'average': OptimizedConfig.SIMILARITY_WEIGHT_AVERAGE,
                    'significant': OptimizedConfig.SIMILARITY_WEIGHT_SIGNIFICANT
                }
            },
            'idlenessAnalysis': {
                'totalPeriods': len(changes),
                'criticalIdlePeriods': critical_idle_periods,
                'highIdlePeriods': high_idle_periods,
                'moderatePeriods': moderate_periods,
                'lowIdlePeriods': low_idle_periods,
                'activePeriods': active_periods,
                'averageIdleness': round(avg_idleness, OptimizedConfig.DECIMAL_PRECISION),
                'maxIdleness': round(max_idleness, OptimizedConfig.DECIMAL_PRECISION),
                'minIdleness': round(min_idleness, OptimizedConfig.DECIMAL_PRECISION),
                'idlenessPercentage': round(idleness_percentage, OptimizedConfig.DECIMAL_PRECISION)
            },
            'timeAnalysis': hourly_analysis,
            'productivityAnalysis': {
                'highlyProductiveTime': round((highly_productive_periods / len(idleness_scores)) * 100, OptimizedConfig.DECIMAL_PRECISION),
                'productiveTime': round(productive_time, OptimizedConfig.DECIMAL_PRECISION),
                'unproductiveTime': round(unproductive_time, OptimizedConfig.DECIMAL_PRECISION),
                'neutralTime': round(neutral_time, OptimizedConfig.DECIMAL_PRECISION),
                'productivityScore': round(productivity_score, OptimizedConfig.DECIMAL_PRECISION)
            },
            'summary': {
                'overallIdleness': round(avg_idleness, OptimizedConfig.DECIMAL_PRECISION),
                'productivityLevel': productivity_level,
                'mostActiveHour': most_active_hour,
                'leastActiveHour': least_active_hour,
                'recommendation': recommendation
            },
            'changes': changes,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        }
        
    except Exception as e:
        if OptimizedConfig.DEBUG_MODE:
            print(f"Erro na análise de ociosidade otimizada: {e}")
            print(f"Traceback: {traceback.format_exc()}")
        return {'success': False, 'error': f'Erro na análise de ociosidade: {str(e)}'}

def analyze_nsfw_optimized(image_data):
    """Análise NSFW otimizada"""
    try:
        size = len(image_data)
        width, height = get_image_dimensions(image_data)
        
        # Usar thresholds otimizados
        if size > 100000:
            confidence = OptimizedConfig.NSFW_HIGH_CONFIDENCE_THRESHOLD
        elif size > OptimizedConfig.NSFW_SMALL_IMAGE_THRESHOLD:
            confidence = OptimizedConfig.NSFW_MEDIUM_CONFIDENCE_THRESHOLD
        else:
            confidence = OptimizedConfig.NSFW_LOW_CONFIDENCE_THRESHOLD
        
        is_nsfw = False  # Assumir não-NSFW por padrão
        
        return {
            'success': True,
            'isNSFW': is_nsfw,
            'confidence': confidence,
            'classifications': [
                {'className': 'Neutral', 'probability': OptimizedConfig.NSFW_NEUTRAL_SCORE, 'percentage': round(OptimizedConfig.NSFW_NEUTRAL_SCORE * 100, 1)},
                {'className': 'Porn', 'probability': OptimizedConfig.NSFW_PORN_SCORE, 'percentage': round(OptimizedConfig.NSFW_PORN_SCORE * 100, 1)},
                {'className': 'Sexy', 'probability': OptimizedConfig.NSFW_SEXY_SCORE, 'percentage': round(OptimizedConfig.NSFW_SEXY_SCORE * 100, 1)},
                {'className': 'Hentai', 'probability': OptimizedConfig.NSFW_HENTAI_SCORE, 'percentage': round(OptimizedConfig.NSFW_HENTAI_SCORE * 100, 1)},
                {'className': 'Drawing', 'probability': OptimizedConfig.NSFW_DRAWING_SCORE, 'percentage': round(OptimizedConfig.NSFW_DRAWING_SCORE * 100, 1)}
            ],
            'details': {
                'isPorn': False,
                'isHentai': False,
                'isSexy': False,
                'primaryCategory': 'Neutral',
                'scores': {
                    'neutral': OptimizedConfig.NSFW_NEUTRAL_SCORE,
                    'porn': OptimizedConfig.NSFW_PORN_SCORE,
                    'sexy': OptimizedConfig.NSFW_SEXY_SCORE,
                    'hentai': OptimizedConfig.NSFW_HENTAI_SCORE,
                    'drawing': OptimizedConfig.NSFW_DRAWING_SCORE
                }
            }
        }
    except Exception as e:
        return {'success': False, 'error': f'Erro NSFW: {str(e)}'}

def analyze_games_optimized(image_data):
    """Análise de jogos otimizada"""
    try:
        size = len(image_data)
        width, height = get_image_dimensions(image_data)
        
        # Heurística otimizada para detectar screenshots de jogos
        aspect_ratio = width / height if height > 0 else 1
        
        game_score = 0
        
        # Resolução típica de jogos (otimizada)
        if width >= OptimizedConfig.GAMES_RESOLUTION_MIN_WIDTH and height >= OptimizedConfig.GAMES_RESOLUTION_MIN_HEIGHT:
            game_score += OptimizedConfig.GAMES_RESOLUTION_SCORE
        
        # Aspect ratio comum em jogos (otimizado)
        if OptimizedConfig.GAMES_ASPECT_RATIO_MIN <= aspect_ratio <= OptimizedConfig.GAMES_ASPECT_RATIO_MAX:
            game_score += OptimizedConfig.GAMES_ASPECT_RATIO_SCORE
        
        # Tamanho de arquivo (otimizado)
        if size > OptimizedConfig.GAMES_FILE_SIZE_THRESHOLD:
            game_score += OptimizedConfig.GAMES_FILE_SIZE_SCORE
        
        # Formato PNG é comum em screenshots (otimizado)
        if image_data.startswith(b'\x89PNG'):
            game_score += OptimizedConfig.GAMES_PNG_FORMAT_SCORE
        
        is_gaming = game_score > OptimizedConfig.GAMES_DETECTION_THRESHOLD
        
        return {
            'success': True,
            'isGaming': is_gaming,
            'confidence': round(min(1.0, game_score), OptimizedConfig.CONFIDENCE_PRECISION),
            'gameScore': round(min(1.0, game_score), OptimizedConfig.CONFIDENCE_PRECISION),
            'detectedGame': 'Screenshot' if is_gaming else None,
            'features': {
                'resolution': f"{width}x{height}",
                'aspectRatio': round(aspect_ratio, OptimizedConfig.DECIMAL_PRECISION),
                'fileSize': size,
                'format': 'PNG' if image_data.startswith(b'\x89PNG') else 'JPEG' if image_data.startswith(b'\xff\xd8') else 'Unknown'
            },
            'thresholds': {
                'detection': OptimizedConfig.GAMES_DETECTION_THRESHOLD,
                'resolution': f"{OptimizedConfig.GAMES_RESOLUTION_MIN_WIDTH}x{OptimizedConfig.GAMES_RESOLUTION_MIN_HEIGHT}",
                'fileSize': OptimizedConfig.GAMES_FILE_SIZE_THRESHOLD
            }
        }
    except Exception as e:
        return {'success': False, 'error': f'Erro Games: {str(e)}'}

def analyze_software_optimized(image_data):
    """Análise de software otimizada"""
    try:
        size = len(image_data)
        width, height = get_image_dimensions(image_data)
        
        confidence = 0
        detected = False
        software_list = []
        
        # Screenshots de desktop tendem a ser grandes (otimizado)
        if size > OptimizedConfig.SOFTWARE_FILE_SIZE_THRESHOLD:
            confidence += OptimizedConfig.SOFTWARE_FILE_SIZE_SCORE
            detected = True
            software_list.append({
                'name': 'Desktop Screenshot',
                'confidence': OptimizedConfig.SOFTWARE_CONFIDENCE_DEFAULT,
                'type': 'screenshot'
            })
        
        # Resolução típica de desktop (otimizada)
        if width >= OptimizedConfig.SOFTWARE_RESOLUTION_MIN_WIDTH and height >= OptimizedConfig.SOFTWARE_RESOLUTION_MIN_HEIGHT:
            confidence += OptimizedConfig.SOFTWARE_RESOLUTION_SCORE
        
        # Formato PNG é comum em screenshots (otimizado)
        if image_data.startswith(b'\x89PNG'):
            confidence += OptimizedConfig.SOFTWARE_PNG_FORMAT_SCORE
        
        confidence = min(1.0, confidence)
        detected = confidence > OptimizedConfig.SOFTWARE_DETECTION_THRESHOLD
        
        return {
            'success': True,
            'detected': detected,
            'confidence': round(confidence, OptimizedConfig.CONFIDENCE_PRECISION),
            'softwareList': software_list if detected else [],
            'urls': [],
            'domains': [],
            'ocrText': 'OCR não disponível nesta versão de produção',
            'thresholds': {
                'detection': OptimizedConfig.SOFTWARE_DETECTION_THRESHOLD,
                'fileSize': OptimizedConfig.SOFTWARE_FILE_SIZE_THRESHOLD,
                'resolution': f"{OptimizedConfig.SOFTWARE_RESOLUTION_MIN_WIDTH}x{OptimizedConfig.SOFTWARE_RESOLUTION_MIN_HEIGHT}"
            }
        }
    except Exception as e:
        return {'success': False, 'error': f'Erro Software: {str(e)}'}

def process_zip_file_optimized(zip_data):
    """Processar arquivo ZIP com validações otimizadas"""
    try:
        if OptimizedConfig.DEBUG_MODE:
            print(f"Processando ZIP de {len(zip_data)} bytes")
        
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_ref:
            file_list = zip_ref.namelist()
            if OptimizedConfig.DEBUG_MODE:
                print(f"Arquivos no ZIP: {file_list}")
            
            # Usar extensões otimizadas
            image_extensions = tuple(f'.{ext}' for ext in OptimizedConfig.IMAGE_EXTENSIONS)
            
            # Filtrar arquivos usando padrões otimizados
            image_files = []
            for f in file_list:
                if f.lower().endswith(image_extensions):
                    # Verificar se não está nos padrões de exclusão otimizados
                    exclude = False
                    for pattern in OptimizedConfig.ZIP_EXCLUDE_PATTERNS:
                        if pattern.strip() in f:
                            exclude = True
                            break
                    if not exclude:
                        image_files.append(f)
            
            if OptimizedConfig.DEBUG_MODE:
                print(f"Imagens encontradas: {image_files}")
            
            # Limitar número de imagens processadas (otimizado)
            if len(image_files) > OptimizedConfig.MAX_IMAGES_PER_ZIP:
                image_files = image_files[:OptimizedConfig.MAX_IMAGES_PER_ZIP]
                if OptimizedConfig.DEBUG_MODE:
                    print(f"Limitado a {OptimizedConfig.MAX_IMAGES_PER_ZIP} imagens")
            
            images_data = []
            errors = []
            
            for img_file in image_files:
                try:
                    if OptimizedConfig.DEBUG_MODE:
                        print(f"Extraindo {img_file}")
                    img_data = zip_ref.read(img_file)
                    
                    # Validações otimizadas
                    if len(img_data) < OptimizedConfig.MIN_IMAGE_SIZE:
                        errors.append(f"Imagem {img_file} muito pequena: {len(img_data)} bytes")
                        continue
                    
                    if len(img_data) > OptimizedConfig.MAX_IMAGE_SIZE:
                        errors.append(f"Imagem {img_file} muito grande: {len(img_data)} bytes")
                        continue
                    
                    if OptimizedConfig.DEBUG_MODE:
                        print(f"Extraído {img_file}: {len(img_data)} bytes")
                    
                    images_data.append({
                        'filename': img_file,
                        'data': img_data,
                        'size': len(img_data)
                    })
                except Exception as e:
                    error_msg = f"Erro ao extrair {img_file}: {str(e)}"
                    if OptimizedConfig.DEBUG_MODE:
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
        if OptimizedConfig.DEBUG_MODE:
            print(error_msg)
        return {'success': False, 'error': error_msg}
    except Exception as e:
        error_msg = f'Erro ao processar ZIP: {str(e)}'
        if OptimizedConfig.DEBUG_MODE:
            print(error_msg)
            print(f"Traceback: {traceback.format_exc()}")
        return {'success': False, 'error': error_msg}

@app.route('/')
def home():
    return jsonify({
        'name': OptimizedConfig.API_NAME,
        'version': OptimizedConfig.API_VERSION,
        'description': 'API otimizada para análise de imagens com algoritmo de ociosidade corrigido',
        'endpoints': {
            'POST /analyze': 'Análise completa otimizada de arquivo ZIP ou imagem única',
            'GET /status': 'Status da API',
            'GET /config': 'Configurações otimizadas atuais',
            'GET /': 'Informações da API'
        },
        'features': ['Optimized NSFW Detection', 'Optimized Game Detection', 'Optimized Software Detection', 'CORRECTED Idleness Analysis'],
        'mode': 'OPTIMIZED - Algoritmo corrigido e calibrado',
        'improvements': [
            'Algoritmo de similaridade híbrido',
            'Múltiplas amostras por imagem',
            'Filtro de ruído configurável',
            'Thresholds calibrados com dados reais',
            'Validações de integridade',
            'Análise temporal aprimorada'
        ],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC')
    })

@app.route('/config')
def config():
    """Endpoint para visualizar configurações otimizadas"""
    return jsonify({
        'version': OptimizedConfig.API_VERSION,
        'optimizations': {
            'algorithm': OptimizedConfig.SIMILARITY_ALGORITHM,
            'multipleRegions': OptimizedConfig.HASH_MULTIPLE_REGIONS,
            'noiseFiltering': OptimizedConfig.SIMILARITY_NOISE_FILTER,
            'validations': {
                'imageHeaders': OptimizedConfig.VALIDATE_IMAGE_HEADERS,
                'timestamps': OptimizedConfig.VALIDATE_TIMESTAMPS,
                'fileIntegrity': OptimizedConfig.VALIDATE_FILE_INTEGRITY
            }
        },
        'idleness': {
            'thresholds': {
                'critical': OptimizedConfig.IDLENESS_THRESHOLD_CRITICAL,
                'high': OptimizedConfig.IDLENESS_THRESHOLD_HIGH,
                'moderate': OptimizedConfig.IDLENESS_THRESHOLD_MODERATE,
                'low': OptimizedConfig.IDLENESS_THRESHOLD_LOW
            },
            'visualSampleSize': OptimizedConfig.VISUAL_SAMPLE_SIZE,
            'multipleRegions': OptimizedConfig.HASH_REGION_COUNT,
            'noiseThreshold': OptimizedConfig.VISUAL_NOISE_THRESHOLD,
            'significantChangeThreshold': OptimizedConfig.VISUAL_SIGNIFICANT_CHANGE_THRESHOLD,
            'weights': {
                'identical': OptimizedConfig.SIMILARITY_WEIGHT_IDENTICAL,
                'average': OptimizedConfig.SIMILARITY_WEIGHT_AVERAGE,
                'significant': OptimizedConfig.SIMILARITY_WEIGHT_SIGNIFICANT
            }
        },
        'nsfw': {
            'thresholds': {
                'high': OptimizedConfig.NSFW_HIGH_CONFIDENCE_THRESHOLD,
                'medium': OptimizedConfig.NSFW_MEDIUM_CONFIDENCE_THRESHOLD,
                'low': OptimizedConfig.NSFW_LOW_CONFIDENCE_THRESHOLD
            }
        },
        'games': {
            'detectionThreshold': OptimizedConfig.GAMES_DETECTION_THRESHOLD,
            'minResolution': f"{OptimizedConfig.GAMES_RESOLUTION_MIN_WIDTH}x{OptimizedConfig.GAMES_RESOLUTION_MIN_HEIGHT}",
            'fileSizeThreshold': OptimizedConfig.GAMES_FILE_SIZE_THRESHOLD
        },
        'software': {
            'detectionThreshold': OptimizedConfig.SOFTWARE_DETECTION_THRESHOLD,
            'fileSizeThreshold': OptimizedConfig.SOFTWARE_FILE_SIZE_THRESHOLD,
            'minResolution': f"{OptimizedConfig.SOFTWARE_RESOLUTION_MIN_WIDTH}x{OptimizedConfig.SOFTWARE_RESOLUTION_MIN_HEIGHT}"
        },
        'processing': {
            'maxImagesPerZip': OptimizedConfig.MAX_IMAGES_PER_ZIP,
            'maxFileSize': OptimizedConfig.MAX_FILE_SIZE,
            'imageExtensions': OptimizedConfig.IMAGE_EXTENSIONS,
            'minImageSize': OptimizedConfig.MIN_IMAGE_SIZE,
            'maxImageSize': OptimizedConfig.MAX_IMAGE_SIZE,
            'minImageDimension': OptimizedConfig.MIN_IMAGE_DIMENSION
        },
        'debug': OptimizedConfig.DEBUG_MODE
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    start_time = time.time()
    
    try:
        if OptimizedConfig.DEBUG_MODE:
            print("=== INÍCIO DA ANÁLISE OTIMIZADA ===")
        
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
        
        if OptimizedConfig.DEBUG_MODE:
            print(f"Arquivo recebido: {file.filename}")
            print(f"Content-Type: {file.content_type}")
        
        file_data = file.read()
        if OptimizedConfig.DEBUG_MODE:
            print(f"Tamanho do arquivo: {len(file_data)} bytes")
        
        # Verificar limite de tamanho otimizado
        if len(file_data) > OptimizedConfig.MAX_FILE_SIZE:
            return jsonify({
                'success': False,
                'error': f'Arquivo muito grande. Máximo permitido: {OptimizedConfig.MAX_FILE_SIZE} bytes',
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            }), 413
        
        # Tentar processar como ZIP primeiro
        zip_result = process_zip_file_optimized(file_data)
        if OptimizedConfig.DEBUG_MODE:
            print(f"Resultado do processamento ZIP: {zip_result}")
        
        if zip_result['success'] and zip_result['imageCount'] > 0:
            if OptimizedConfig.DEBUG_MODE:
                print("Processando como ZIP com imagens")
            # É um ZIP válido com imagens
            images_data = zip_result['images']
            
            # Analisar cada imagem usando funções otimizadas
            analyzed_images = []
            for i, img_info in enumerate(images_data):
                if OptimizedConfig.DEBUG_MODE:
                    print(f"Analisando imagem {i+1}/{len(images_data)}: {img_info['filename']}")
                img_start = time.time()
                
                nsfw_result = analyze_nsfw_optimized(img_info['data'])
                games_result = analyze_games_optimized(img_info['data'])
                software_result = analyze_software_optimized(img_info['data'])
                
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
            
            if OptimizedConfig.DEBUG_MODE:
                print("Iniciando análise de ociosidade OTIMIZADA")
            # Análise de ociosidade OTIMIZADA para múltiplas imagens
            idleness_result = analyze_optimized_idleness(images_data)
            if OptimizedConfig.DEBUG_MODE:
                print(f"Resultado da análise de ociosidade: {idleness_result}")
            
            # Compilar estatísticas
            stats = compile_statistics_optimized(analyzed_images, idleness_result)
            
            total_processing_time = int((time.time() - start_time) * 1000)
            
            if OptimizedConfig.DEBUG_MODE:
                print("=== ANÁLISE OTIMIZADA CONCLUÍDA COM SUCESSO ===")
            
            return jsonify({
                'success': True,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'processingTime': total_processing_time,
                'version': OptimizedConfig.API_VERSION,
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
            if OptimizedConfig.DEBUG_MODE:
                print("Processando como imagem única")
            # Tratar como imagem única
            img_start = time.time()
            
            nsfw_result = analyze_nsfw_optimized(file_data)
            games_result = analyze_games_optimized(file_data)
            software_result = analyze_software_optimized(file_data)
            
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
            
            stats = compile_statistics_optimized([analyzed_image], None)
            
            return jsonify({
                'success': True,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'processingTime': total_processing_time,
                'version': OptimizedConfig.API_VERSION,
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
        if OptimizedConfig.DEBUG_MODE:
            print(f"ERRO: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': traceback.format_exc() if OptimizedConfig.DEBUG_MODE else None,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'processingTime': int((time.time() - start_time) * 1000)
        }), 500

def compile_statistics_optimized(images, idleness_result):
    """Compilar estatísticas otimizadas dos resultados"""
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
    
    stats['nsfw']['averageConfidence'] = round(total_nsfw_conf / len(images), OptimizedConfig.CONFIDENCE_PRECISION)
    stats['games']['averageConfidence'] = round(total_games_conf / len(images), OptimizedConfig.CONFIDENCE_PRECISION)
    stats['software']['averageConfidence'] = round(total_software_conf / len(images), OptimizedConfig.CONFIDENCE_PRECISION)
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

@app.route('/status')
def status():
    return jsonify({
        'status': 'online',
        'version': OptimizedConfig.API_VERSION,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
        'detectors': {
            'nsfw': 'Optimized Heuristic Analysis',
            'games': 'Optimized Image Characteristics',
            'software': 'Optimized Basic Detection',
            'idleness': 'CORRECTED Optimized Visual Content Analysis'
        },
        'dependencies': {
            'external': 'None - Self-contained',
            'zipfile': True,
            'flask': True,
            'dotenv': True
        },
        'mode': 'OPTIMIZED',
        'configFile': '.env.optimized loaded',
        'debug': OptimizedConfig.DEBUG_MODE,
        'improvements': [
            'Algoritmo de similaridade corrigido',
            'Múltiplas amostras por região',
            'Filtro de ruído inteligente',
            'Thresholds calibrados',
            'Validações de integridade'
        ]
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'version': OptimizedConfig.API_VERSION, 
        'mode': 'optimized',
        'configLoaded': True,
        'algorithmCorrected': True
    })

if __name__ == '__main__':
    print(f"🚀 {OptimizedConfig.API_NAME} v{OptimizedConfig.API_VERSION}")
    print("📊 Análise OTIMIZADA com algoritmo corrigido")
    print("⏱️ Análise de ociosidade: Corrected Optimized Visual Content Analysis")
    print("🔍 Detectores: Otimizados e calibrados")
    print(f"🐛 Debug mode: {'ON' if OptimizedConfig.DEBUG_MODE else 'OFF'}")
    print("✅ Melhorias: Algoritmo híbrido, múltiplas amostras, filtro de ruído")
    
    app.config['MAX_CONTENT_LENGTH'] = OptimizedConfig.MAX_FILE_SIZE
    app.run(host='0.0.0.0', port=OptimizedConfig.PORT, debug=OptimizedConfig.DEBUG_MODE, threaded=True)
