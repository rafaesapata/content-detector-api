import subprocess
import json
import tempfile
import os
from PIL import Image
import zipfile
import io

def analyze_nsfw_real(image_data):
    """Análise NSFW real usando nsfwjs via Node.js"""
    try:
        # Salvar imagem temporariamente
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            if isinstance(image_data, bytes):
                # Converter bytes para imagem
                img = Image.open(io.BytesIO(image_data))
                img = img.convert('RGB')
                img.save(tmp.name, 'JPEG')
            else:
                image_data.save(tmp.name)
            
            # Script Node.js para análise NSFW
            node_script = f"""
const tf = require('@tensorflow/tfjs-node');
const nsfw = require('nsfwjs');
const fs = require('fs');

async function analyze() {{
    try {{
        const model = await nsfw.load();
        const pic = fs.readFileSync('{tmp.name}');
        const image = await tf.node.decodeImage(pic, 3);
        const predictions = await model.classify(image);
        image.dispose();
        
        const result = {{}};
        predictions.forEach(p => {{
            result[p.className.toLowerCase()] = p.probability;
        }});
        
        console.log(JSON.stringify(result));
    }} catch(e) {{
        console.log(JSON.stringify({{"error": e.message}}));
    }}
}}

analyze();
"""
            
            # Executar análise
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as script_file:
                script_file.write(node_script)
                script_file.flush()
                
                result = subprocess.run(
                    ['node', script_file.name],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd='/home/ubuntu/detector-api'
                )
                
                os.unlink(script_file.name)
            
            os.unlink(tmp.name)
            
            if result.returncode == 0:
                data = json.loads(result.stdout.strip())
                if 'error' in data:
                    return {'success': False, 'error': data['error']}
                
                # Processar resultados
                porn_score = data.get('porn', 0)
                hentai_score = data.get('hentai', 0) 
                sexy_score = data.get('sexy', 0)
                neutral_score = data.get('neutral', 0)
                drawing_score = data.get('drawing', 0)
                
                is_nsfw = (porn_score > 0.5 or hentai_score > 0.5 or sexy_score > 0.7)
                
                return {
                    'success': True,
                    'isNSFW': is_nsfw,
                    'confidence': max(porn_score, hentai_score, sexy_score, neutral_score),
                    'classifications': [
                        {'className': 'Porn', 'probability': porn_score, 'percentage': round(porn_score * 100, 1)},
                        {'className': 'Hentai', 'probability': hentai_score, 'percentage': round(hentai_score * 100, 1)},
                        {'className': 'Sexy', 'probability': sexy_score, 'percentage': round(sexy_score * 100, 1)},
                        {'className': 'Neutral', 'probability': neutral_score, 'percentage': round(neutral_score * 100, 1)},
                        {'className': 'Drawing', 'probability': drawing_score, 'percentage': round(drawing_score * 100, 1)}
                    ],
                    'details': {
                        'isPorn': porn_score > 0.5,
                        'isHentai': hentai_score > 0.5,
                        'isSexy': sexy_score > 0.7,
                        'primaryCategory': max(data.items(), key=lambda x: x[1])[0].title(),
                        'scores': data
                    }
                }
            else:
                return {'success': False, 'error': f'Node.js error: {result.stderr}'}
                
    except Exception as e:
        return {'success': False, 'error': str(e)}

def analyze_games_real(image_data):
    """Análise de jogos real usando características visuais"""
    try:
        # Converter para PIL Image
        if isinstance(image_data, bytes):
            img = Image.open(io.BytesIO(image_data))
        else:
            img = image_data
            
        img = img.convert('RGB')
        
        # Análise de características
        width, height = img.size
        pixels = list(img.getdata())
        
        # Calcular saturação média
        total_saturation = 0
        for r, g, b in pixels[::100]:  # Amostragem
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            if max_val > 0:
                saturation = (max_val - min_val) / max_val
                total_saturation += saturation
        
        avg_saturation = total_saturation / (len(pixels) // 100)
        
        # Detectar elementos de UI (bordas e contrastes)
        edge_count = 0
        for i in range(0, len(pixels) - width, width):
            for j in range(width - 1):
                if i + j + 1 < len(pixels):
                    r1, g1, b1 = pixels[i + j]
                    r2, g2, b2 = pixels[i + j + 1]
                    diff = abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
                    if diff > 100:
                        edge_count += 1
        
        edge_ratio = edge_count / (width * height) if width * height > 0 else 0
        
        # Detectar cores vibrantes típicas de jogos
        vibrant_colors = 0
        for r, g, b in pixels[::50]:
            if max(r, g, b) > 200 and min(r, g, b) < 100:
                vibrant_colors += 1
        
        vibrant_ratio = vibrant_colors / (len(pixels) // 50)
        
        # Score de jogo baseado em características
        game_score = (avg_saturation * 0.3 + edge_ratio * 0.4 + vibrant_ratio * 0.3)
        game_score = min(1.0, game_score * 2)  # Normalizar
        
        is_gaming = game_score > 0.3
        
        # Detectar jogos específicos por padrões
        detected_game = "Unknown"
        if avg_saturation > 0.6 and vibrant_ratio > 0.3:
            detected_game = "Action Game"
        elif edge_ratio > 0.15:
            detected_game = "Strategy Game"
        
        return {
            'success': True,
            'isGaming': is_gaming,
            'confidence': round(game_score, 3),
            'gameScore': round(game_score, 3),
            'detectedGame': detected_game if is_gaming else None,
            'features': {
                'saturation': round(avg_saturation, 3),
                'edgeRatio': round(edge_ratio, 3),
                'vibrantColors': round(vibrant_ratio, 3),
                'resolution': f"{width}x{height}"
            }
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def analyze_software_real(image_data):
    """Análise de software real usando OCR"""
    try:
        # Salvar imagem temporariamente
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            if isinstance(image_data, bytes):
                img = Image.open(io.BytesIO(image_data))
                img = img.convert('RGB')
                img.save(tmp.name, 'JPEG')
            else:
                image_data.save(tmp.name)
            
            # OCR usando tesseract
            result = subprocess.run(
                ['tesseract', tmp.name, 'stdout', '-l', 'eng+por'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            os.unlink(tmp.name)
            
            if result.returncode == 0:
                ocr_text = result.stdout.strip()
                
                # Lista de software conhecidos
                software_patterns = [
                    'Chrome', 'Firefox', 'Safari', 'Edge', 'Opera',
                    'Visual Studio', 'VSCode', 'Sublime', 'Atom', 'Notepad++',
                    'Photoshop', 'Illustrator', 'GIMP', 'Figma', 'Sketch',
                    'Excel', 'Word', 'PowerPoint', 'Outlook', 'Teams',
                    'Slack', 'Discord', 'Telegram', 'WhatsApp', 'Zoom',
                    'Steam', 'Epic Games', 'Origin', 'Battle.net', 'GOG',
                    'Windows', 'macOS', 'Linux', 'Ubuntu', 'Terminal',
                    'Docker', 'Git', 'GitHub', 'GitLab', 'Bitbucket'
                ]
                
                # Detectar software
                detected_software = []
                for software in software_patterns:
                    if software.lower() in ocr_text.lower():
                        detected_software.append({
                            'name': software,
                            'confidence': 0.8,
                            'type': 'detected_by_ocr'
                        })
                
                # Detectar URLs
                import re
                url_pattern = r'https?://[^\s]+'
                urls = re.findall(url_pattern, ocr_text)
                
                # Detectar domínios
                domain_pattern = r'[a-zA-Z0-9-]+\.[a-zA-Z]{2,}'
                domains = list(set(re.findall(domain_pattern, ocr_text)))
                
                confidence = len(detected_software) * 0.2 + len(urls) * 0.1 + len(domains) * 0.05
                confidence = min(1.0, confidence)
                
                return {
                    'success': True,
                    'detected': len(detected_software) > 0 or len(urls) > 0,
                    'confidence': round(confidence, 3),
                    'softwareList': detected_software[:5],
                    'urls': urls[:10],
                    'domains': domains[:10],
                    'ocrText': ocr_text[:500] if ocr_text else ""
                }
            else:
                return {'success': False, 'error': f'OCR failed: {result.stderr}'}
                
    except Exception as e:
        return {'success': False, 'error': str(e)}

def analyze_idleness_real(images_data):
    """Análise de ociosidade real comparando imagens"""
    try:
        if len(images_data) < 2:
            return {
                'success': True,
                'message': 'Necessário pelo menos 2 imagens para análise de ociosidade',
                'idlenessAnalysis': {'averageIdleness': 0}
            }
        
        # Converter imagens e calcular diferenças
        prev_img = None
        changes = []
        
        for i, img_data in enumerate(images_data):
            if isinstance(img_data, bytes):
                img = Image.open(io.BytesIO(img_data))
            else:
                img = img_data
                
            img = img.convert('RGB').resize((100, 100))  # Reduzir para comparação
            pixels = list(img.getdata())
            
            if prev_img:
                # Calcular diferença pixel a pixel
                diff_sum = 0
                for j in range(len(pixels)):
                    r1, g1, b1 = prev_img[j]
                    r2, g2, b2 = pixels[j]
                    diff = abs(r1-r2) + abs(g1-g2) + abs(b1-b2)
                    diff_sum += diff
                
                avg_diff = diff_sum / len(pixels)
                change_score = min(1.0, avg_diff / 300)  # Normalizar
                idleness_score = (1 - change_score) * 100
                changes.append(idleness_score)
            
            prev_img = pixels
        
        # Estatísticas
        avg_idleness = sum(changes) / len(changes) if changes else 0
        idle_periods = sum(1 for x in changes if x > 70)
        active_periods = len(changes) - idle_periods
        
        return {
            'success': True,
            'totalImages': len(images_data),
            'idlenessAnalysis': {
                'totalPeriods': len(changes),
                'idlePeriods': idle_periods,
                'activePeriods': active_periods,
                'averageIdleness': round(avg_idleness, 2),
                'maxIdleness': round(max(changes), 2) if changes else 0,
                'minIdleness': round(min(changes), 2) if changes else 0,
                'idlenessPercentage': round((idle_periods / len(changes)) * 100, 2) if changes else 0
            },
            'summary': {
                'overallIdleness': round(avg_idleness, 2),
                'recommendation': 'Alto nível de ociosidade' if avg_idleness > 60 else 'Nível de atividade adequado'
            }
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def process_zip_real(zip_data):
    """Processar ZIP real extraindo imagens"""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_ref:
            file_list = zip_ref.namelist()
            image_files = [f for f in file_list if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')) and not f.startswith('__MACOSX')]
            
            images_data = []
            for img_file in image_files:
                try:
                    img_data = zip_ref.read(img_file)
                    images_data.append({
                        'filename': img_file,
                        'data': img_data,
                        'size': len(img_data)
                    })
                except:
                    continue
            
            return {
                'success': True,
                'totalFiles': len(file_list),
                'imageCount': len(images_data),
                'images': images_data,
                'errors': []
            }
            
    except Exception as e:
        return {'success': False, 'error': str(e)}
