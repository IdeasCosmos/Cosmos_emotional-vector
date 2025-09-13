import requests
import json

def print_korean_result(response_json):
    emotion_map = ['기쁨', '슬픔', '분노', '두려움', '혐오', '놀람', '중립']
    music_map = {
        'tempo_bpm': '템포 (BPM)',
        'key': '키',
        'mode': '모드',
        'dynamics': '다이내믹스',
        'chord_progression': '코드 진행',
        'complexity': '복잡성'
    }

    print("\n--- 분석 결과 ---")
    if 'emotion_vector' in response_json:
        emotion_vector = response_json['emotion_vector']
        print(f"감정 벡터: {emotion_vector}")
        dominant_emotion_index = emotion_vector.index(max(emotion_vector))
        print(f"주요 감정: {emotion_map[dominant_emotion_index]}")
    if 'confidence' in response_json:
        print(f"신뢰도: {response_json['confidence']:.2f}")
    if 'music_parameters' in response_json:
        print("음악 파라미터:")
        for key, value in response_json['music_parameters'].items():
            print(f"  {music_map.get(key, key)}: {value}")
    if 'processing_time_ms' in response_json:
        print(f"처리 시간: {response_json['processing_time_ms']:.2f}ms")
    if 'eeg_quality' in response_json and response_json['eeg_quality'] is not None:
        print(f"EEG 품질: {response_json['eeg_quality']}")
    if 'snr_db' in response_json and response_json['snr_db'] is not None:
        print(f"SNR (dB): {response_json['snr_db']:.2f}")

# 텍스트 감정 분석
print("텍스트 감정 분석 요청...")
response_text = requests.post("http://localhost:8000/analyze", json={
    "text": "오늘 정말 기쁘네요!"
})
print_korean_result(response_text.json())

# EEG 밴드 파워로 분석
print("\nEEG 밴드 파워 분석 요청...")
response_eeg = requests.post("http://localhost:8000/analyze", json={
    "eeg_bands": {
        "delta": -0.1, "theta": -0.05, 
        "alpha": 0.35, "beta": 0.25, "gamma": 0.15
    }
})
print_korean_result(response_eeg.json())