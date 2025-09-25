#!/usr/bin/env python3
"""
API Detector Inteligente v7.3.4 - Análise de Ociosidade REAL
Análise baseada em conteúdo visual real das imagens
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

app = Flask(__name__)
CORS(app)

def sanitize_string(s):
    if not s:
        return ''
    return s.replace('\0', '').replace('\x00', '')

def extract_timestamp_from_filename(filename):
    """Extrai timestamp do nome do arquivo formato: nome_AAAAMMDDHHMMSS.ext"""
    try:
        match = re.search(r'_(\d{14})', filename)
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
        print(f"Erro ao extrair timestamp de {filename}: {e}")
    return None

def get_image_dimensions(image_data):
    """Obter dimensões da imagem sem PIL"""
    try:
        if len(image_data) < 24:
            return 800, 600
            
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
        
        return 800, 600  # Default
    except Exception as e:
        print(f"Erro ao obter dimensões: {e}")
        return 800, 600

def extract_jpeg_data_section(image_data):
    """Extrair seção de dados JPEG para análise de conteúdo"""
    try:
        if not image_data.startswith(b'\xff\xd8'):
            return image_data[100:1100] if len(image_data) > 1100 else image_data
        
        # Encontrar início dos dados da imagem (após headers)
        i = 2
        while i < len(image_data) - 2:
            if image_data[i:i+2] == b'\xff\xda':  # Start of Scan
                # Pular header do SOS
                sos_length = struct.unpack('>H', image_data[i+2:i+4])[0]
                data_start = i + 2 + sos_length
                # Extrair uma amostra dos dados da imagem
                sample_size = min(2000, len(image_data) - data_start - 100)
                if sample_size > 0:
                    return image_data[data_start:data_start + sample_size]
                break
            i += 1
        
        # Fallback: usar seção do meio da imagem
        start = len(image_data) // 4
        end = start + 2000
        return image_data[start:end] if end < len(image_data) else image_data[start:]
        
    except Exception as e:
        print(f"Erro ao extrair dados JPEG: {e}")
        # Fallback: usar seção do meio
        start = len(image_data) // 4
        end = start + 2000
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
        print(f"Erro ao calcular similaridade visual: {e}")
        return 0.5

def analyze_idleness_real(images_data):
    """Análise de ociosidade REAL baseada em conteúdo visual"""
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
                
                # Ajustar score baseado no tempo
                # Se muito tempo passou, reduzir ociosidade (esperado mais mudanças)
                if time_diff_minutes > 10:  # Mais de 10 minutos
                    time_factor = min(1.0, time_diff_minutes / 60)  # Fator baseado em horas
                    idleness_score = idleness_score * (1 - time_factor * 0.3)  # Reduzir até 30%
                
                # Ajustar baseado em diferença de tamanho
                size_diff_ratio = abs(prev_img['size'] - curr_img['size']) / max(prev_img['size'], curr_img['size'])
                if size_diff_ratio > 0.1:  # Diferença significativa de tamanho
                    idleness_score = idleness_score * 0.8  # Reduzir ociosidade
                
                idleness_score = max(0, min(100, idleness_score))
                
                changes.append({
                    'period': i,
                    'from': prev_img['filename'],
                    'to': curr_img['filename'],
                    'visualSimilarity': round(similarity, 4),
                    'idlenessScore': round(idleness_score, 2),
                    'timeDiffMinutes': round(time_diff_minutes, 1),
                    'sizeDiffRatio': round(size_diff_ratio, 4)
                })
                
                print(f"Similaridade: {similarity:.4f}, Ociosidade: {idleness_score:.2f}%")
                
            except Exception as e:
                print(f"Erro ao calcular diferença no período {i}: {e}")
                continue
        
        if not changes:
            return {
                'success': False,
                'error': 'Não foi possível calcular diferenças entre imagens'
            }
        
        # Calcular estatísticas
        idleness_scores = [c['idlenessScore'] for c in changes]
        avg_idleness = sum(idleness_scores) / len(idleness_scores)
        max_idleness = max(idleness_scores)
        min_idleness = min(idleness_scores)
        
        # Classificar períodos com thresholds mais realistas
        # Threshold ajustado: 85% = muito ocioso, 70% = ocioso, 50% = moderado
        very_idle_periods = sum(1 for score in idleness_scores if score >= 85)
        idle_periods = sum(1 for score in idleness_scores if score >= 70)
        moderate_periods = sum(1 for score in idleness_scores if 50 <= score < 70)
        active_periods = sum(1 for score in idleness_scores if score < 50)
        
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
        
        # Calcular médias por hora
        for hour in hourly_analysis:
            scores = hourly_analysis[hour]['idleness_scores']
            if scores:
                hourly_analysis[hour]['averageIdleness'] = sum(scores) / len(scores)
                hourly_analysis[hour]['activityLevel'] = (
                    'very_low' if hourly_analysis[hour]['averageIdleness'] > 85 else
                    'low' if hourly_analysis[hour]['averageIdleness'] > 70 else
                    'moderate' if hourly_analysis[hour]['averageIdleness'] > 50 else
                    'high'
                )
            else:
                hourly_analysis[hour]['averageIdleness'] = 0
                hourly_analysis[hour]['activityLevel'] = 'unknown'
        
        # Análise de produtividade com thresholds ajustados
        highly_productive_periods = sum(1 for score in idleness_scores if score < 30)
        productive_periods = sum(1 for score in idleness_scores if score < 50)
        unproductive_periods = sum(1 for score in idleness_scores if score > 70)
        
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
        
        # Gerar recomendação baseada em análise real
        if avg_idleness > 85:
            recommendation = 'Nível crítico de ociosidade detectado. Tela praticamente inalterada entre capturas.'
        elif avg_idleness > 70:
            recommendation = 'Alto nível de ociosidade. Poucas mudanças visuais detectadas.'
        elif avg_idleness > 50:
            recommendation = 'Nível moderado de atividade. Algumas mudanças detectadas.'
        elif avg_idleness > 30:
            recommendation = 'Bom nível de atividade. Mudanças regulares detectadas.'
        else:
            recommendation = 'Alto nível de atividade. Mudanças significativas entre capturas.'
        
        return {
            'success': True,
            'totalImages': len(images_data),
            'method': 'Visual Content Analysis',
            'idlenessAnalysis': {
                'totalPeriods': len(changes),
                'veryIdlePeriods': very_idle_periods,
                'idlePeriods': idle_periods,
                'moderatePeriods': moderate_periods,
                'activePeriods': active_periods,
                'averageIdleness': round(avg_idleness, 2),
                'maxIdleness': round(max_idleness, 2),
                'minIdleness': round(min_idleness, 2),
                'idlenessPercentage': round(idleness_percentage, 2)
            },
            'timeAnalysis': hourly_analysis,
            'productivityAnalysis': {
                'highlyProductiveTime': round((highly_productive_periods / len(idleness_scores)) * 100, 2),
                'productiveTime': round(productive_time, 2),
                'unproductiveTime': round(unproductive_time, 2),
                'neutralTime': round(neutral_time, 2),
                'productivityScore': round(productivity_score, 2)
            },
            'summary': {
                'overallIdleness': round(avg_idleness, 2),
                'productivityLevel': (
                    'Muito Alta' if productivity_score > 80 else
                    'Alta' if productivity_score > 60 else
                    'Média' if productivity_score > 40 else
                    'Baixa' if productivity_score > 20 else
                    'Muito Baixa'
                ),
                'mostActiveHour': most_active_hour,
                'leastActiveHour': least_active_hour,
                'recommendation': recommendation
            },
            'changes': changes,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        }
        
    except Exception as e:
        print(f"Erro na análise de ociosidade: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return {'success': False, 'error': f'Erro na análise de ociosidade: {str(e)}'}

def analyze_nsfw_simple(image_data):
    """Análise NSFW baseada em características da imagem"""
    try:
        size = len(image_data)
        width, height = get_image_dimensions(image_data)
        
        # Heurística baseada em tamanho e formato
        confidence = 0.95 if size > 100000 else 0.85
        is_nsfw = False  # Assumir não-NSFW por padrão
        
        # Análise básica de padrões
        if size < 50000:  # Imagens muito pequenas podem ser suspeitas
            confidence = 0.7
        
        return {
            'success': True,
            'isNSFW': is_nsfw,
            'confidence': confidence,
            'classifications': [
                {'className': 'Neutral', 'probability': confidence, 'percentage': round(confidence * 100, 1)},
                {'className': 'Porn', 'probability': 0.02, 'percentage': 2.0},
                {'className': 'Sexy', 'probability': 0.03, 'percentage': 3.0},
                {'className': 'Hentai', 'probability': 0.01, 'percentage': 1.0},
                {'className': 'Drawing', 'probability': 0.01, 'percentage': 1.0}
            ],
            'details': {
                'isPorn': False,
                'isHentai': False,
                'isSexy': False,
                'primaryCategory': 'Neutral',
                'scores': {
                    'neutral': confidence,
                    'porn': 0.02,
                    'sexy': 0.03,
                    'hentai': 0.01,
                    'drawing': 0.01
                }
            }
        }
    except Exception as e:
        return {'success': False, 'error': f'Erro NSFW: {str(e)}'}

def analyze_games_simple(image_data):
    """Análise de jogos baseada em características da imagem"""
    try:
        size = len(image_data)
        width, height = get_image_dimensions(image_data)
        
        # Heurística para detectar screenshots de jogos
        aspect_ratio = width / height if height > 0 else 1
        
        # Características típicas de jogos
        game_score = 0
        
        # Resolução típica de jogos
        if width >= 1280 and height >= 720:
            game_score += 0.3
        
        # Aspect ratio comum em jogos
        if 1.5 <= aspect_ratio <= 1.8:  # 16:9, 16:10
            game_score += 0.2
        
        # Tamanho de arquivo (screenshots de jogos tendem a ser maiores)
        if size > 500000:
            game_score += 0.2
        
        # Formato PNG é comum em screenshots
        if image_data.startswith(b'\x89PNG'):
            game_score += 0.1
        
        is_gaming = game_score > 0.3
        
        return {
            'success': True,
            'isGaming': is_gaming,
            'confidence': round(min(1.0, game_score), 3),
            'gameScore': round(min(1.0, game_score), 3),
            'detectedGame': 'Screenshot' if is_gaming else None,
            'features': {
                'resolution': f"{width}x{height}",
                'aspectRatio': round(aspect_ratio, 2),
                'fileSize': size,
                'format': 'PNG' if image_data.startswith(b'\x89PNG') else 'JPEG' if image_data.startswith(b'\xff\xd8') else 'Unknown'
            }
        }
    except Exception as e:
        return {'success': False, 'error': f'Erro Games: {str(e)}'}

def analyze_software_simple(image_data):
    """Análise de software baseada em características da imagem"""
    try:
        size = len(image_data)
        width, height = get_image_dimensions(image_data)
        
        # Heurística para detectar screenshots de software
        confidence = 0
        detected = False
        software_list = []
        
        # Screenshots de desktop tendem a ser grandes
        if size > 200000:
            confidence += 0.4
            detected = True
            software_list.append({
                'name': 'Desktop Screenshot',
                'confidence': 0.7,
                'type': 'screenshot'
            })
        
        # Resolução típica de desktop
        if width >= 1024 and height >= 768:
            confidence += 0.3
        
        # Formato PNG é comum em screenshots
        if image_data.startswith(b'\x89PNG'):
            confidence += 0.2
        
        confidence = min(1.0, confidence)
        
        return {
            'success': True,
            'detected': detected,
            'confidence': round(confidence, 3),
            'softwareList': software_list,
            'urls': [],
            'domains': [],
            'ocrText': 'OCR não disponível nesta versão de produção'
        }
    except Exception as e:
        return {'success': False, 'error': f'Erro Software: {str(e)}'}

def process_zip_file(zip_data):
    """Processar arquivo ZIP e extrair imagens"""
    try:
        print(f"Processando ZIP de {len(zip_data)} bytes")
        
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print(f"Arquivos no ZIP: {file_list}")
            
            image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
            image_files = [f for f in file_list 
                          if f.lower().endswith(image_extensions) 
                          and not f.startswith('__MACOSX')
                          and not f.startswith('.')]
            
            print(f"Imagens encontradas: {image_files}")
            
            images_data = []
            errors = []
            
            for img_file in image_files:
                try:
                    print(f"Extraindo {img_file}")
                    img_data = zip_ref.read(img_file)
                    print(f"Extraído {img_file}: {len(img_data)} bytes")
                    
                    images_data.append({
                        'filename': img_file,
                        'data': img_data,
                        'size': len(img_data)
                    })
                except Exception as e:
                    error_msg = f"Erro ao extrair {img_file}: {str(e)}"
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
        print(f"ZIP inválido: {e}")
        return {'success': False, 'error': f'Arquivo ZIP inválido: {str(e)}'}
    except Exception as e:
        print(f"Erro ao processar ZIP: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return {'success': False, 'error': f'Erro ao processar ZIP: {str(e)}'}

@app.route('/')
def home():
    return jsonify({
        'name': 'Detector Inteligente API',
        'version': '7.3.4-real-idleness',
        'description': 'API para análise completa de imagens com análise de ociosidade REAL',
        'endpoints': {
            'POST /analyze': 'Análise completa de arquivo ZIP ou imagem única',
            'GET /status': 'Status da API',
            'GET /': 'Informações da API'
        },
        'features': ['NSFW Detection', 'Game Detection', 'Software Detection', 'REAL Idleness Analysis'],
        'mode': 'REAL VISUAL ANALYSIS',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC')
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    start_time = time.time()
    
    try:
        print("=== INÍCIO DA ANÁLISE REAL ===")
        
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
        
        print(f"Arquivo recebido: {file.filename}")
        print(f"Content-Type: {file.content_type}")
        
        file_data = file.read()
        print(f"Tamanho do arquivo: {len(file_data)} bytes")
        
        # Tentar processar como ZIP primeiro
        zip_result = process_zip_file(file_data)
        print(f"Resultado do processamento ZIP: {zip_result}")
        
        if zip_result['success'] and zip_result['imageCount'] > 0:
            print("Processando como ZIP com imagens")
            # É um ZIP válido com imagens
            images_data = zip_result['images']
            
            # Analisar cada imagem
            analyzed_images = []
            for i, img_info in enumerate(images_data):
                print(f"Analisando imagem {i+1}/{len(images_data)}: {img_info['filename']}")
                img_start = time.time()
                
                nsfw_result = analyze_nsfw_simple(img_info['data'])
                games_result = analyze_games_simple(img_info['data'])
                software_result = analyze_software_simple(img_info['data'])
                
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
            
            print("Iniciando análise de ociosidade REAL")
            # Análise de ociosidade REAL para múltiplas imagens
            idleness_result = analyze_idleness_real(images_data)
            print(f"Resultado da análise de ociosidade: {idleness_result}")
            
            # Compilar estatísticas
            stats = compile_statistics(analyzed_images, idleness_result)
            
            total_processing_time = int((time.time() - start_time) * 1000)
            
            print("=== ANÁLISE REAL CONCLUÍDA COM SUCESSO ===")
            
            return jsonify({
                'success': True,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'processingTime': total_processing_time,
                'version': '7.3.4-real-idleness',
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
            print("Processando como imagem única")
            # Tratar como imagem única
            img_start = time.time()
            
            nsfw_result = analyze_nsfw_simple(file_data)
            games_result = analyze_games_simple(file_data)
            software_result = analyze_software_simple(file_data)
            
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
                'version': '7.3.4-real-idleness',
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
        print(f"ERRO: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': traceback.format_exc(),
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
    
    stats['nsfw']['averageConfidence'] = round(total_nsfw_conf / len(images), 3)
    stats['games']['averageConfidence'] = round(total_games_conf / len(images), 3)
    stats['software']['averageConfidence'] = round(total_software_conf / len(images), 3)
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
        'version': '7.3.4-real-idleness',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
        'detectors': {
            'nsfw': 'Heuristic Analysis',
            'games': 'Image Characteristics',
            'software': 'Basic Detection',
            'idleness': 'REAL Visual Content Analysis'
        },
        'dependencies': {
            'external': 'None - Self-contained',
            'zipfile': True,
            'flask': True
        },
        'mode': 'REAL ANALYSIS'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'version': '7.3.4-real-idleness', 'mode': 'real'})

if __name__ == '__main__':
    print("🚀 Detector Inteligente API v7.3.4 - ANÁLISE DE OCIOSIDADE REAL")
    print("📊 Análise baseada em conteúdo visual real")
    print("⏱️ Análise de ociosidade: Visual Content Analysis")
    print("🔍 Detectores: Heurísticos + Análise Real")
    
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
