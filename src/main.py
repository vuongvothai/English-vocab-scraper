import pandas as pd
import requests
import urllib.parse
import time
import re
import sys
import os

input_file = "E:/vocabulary - Vocabulary.csv"
output_file = "E:/vocabulary_completed.csv"

# Ưu tiên đọc file completed hiện tại để tiếp tục điền bổ sung / làm sạch
active_file = output_file if os.path.exists(output_file) else input_file
print(f"Đang đọc dữ liệu từ: {active_file}...")
df = pd.read_csv(active_file)
total_rows = len(df)

try:
    import eng_to_ipa as ipa
except ImportError:
    print("Vui lòng cài đặt thư viện: pip install pandas requests eng_to_ipa")
    sys.exit(1)

def is_invalid(val):
    if pd.isna(val): return True
    s = str(val).strip()
    return s == '' or s == '\n' or s == 'nan' or s == 'None' or 'A term referring to' in s

def extract_headword(term):
    if is_invalid(term): return ""
    s = str(term).strip()
    s = re.sub(r'\(.*?\)', '', s)
    if '/' in s: s = s.split('/')[0]
    s = re.sub(r'\b[A-Z]\b(\s+(to|from|with|for|doing|into|on|about))?', '', s)
    s = re.sub(r'\b(doing|something|someone)\b', '', s, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', s).strip()

# 1. HÀM PHÂN LOẠI TRÌNH ĐỘ CHUẨN (A1 - C2)
def predict_level(word, pos=""):
    clean_w = extract_headword(word)
    if not clean_w: clean_w = str(word).strip()
    words = clean_w.split()
    length = len(clean_w)
    pos_lower = str(pos).lower()
    
    # Cụm từ rất dài (>= 4 từ) hoặc Thành ngữ phức tạp -> C2
    if len(words) >= 4 or 'idiom' in pos_lower:
        return 'C2'
    
    # Cụm 3 từ -> C1
    if len(words) == 3:
        return 'C1'
    
    # Cụm 2 từ hoặc Từ đơn dài (>= 10 ký tự) -> B2
    if len(words) == 2 or length >= 10:
        return 'B2'
    
    # Từ đơn khá dài (7-9 ký tự) hoặc chứa hậu tố chuyên ngành -> B1
    if length >= 7 or any(clean_w.lower().endswith(ext) for ext in ['tion', 'ment', 'ance', 'ence', 'itive', 'ology', 'ability']):
        return 'B1'
    
    # Từ đơn ngắn thông thường (4-6 ký tự) -> A2
    if length >= 4:
        return 'A2'
        
    # Từ siêu ngắn / rất cơ bản (< 4 ký tự) -> A1
    return 'A1'

def get_safe_ipa(phrase):
    head = extract_headword(phrase)
    if not head: return ""
    words = head.replace('-', ' ').split()
    ipa_words = []
    for w in words:
        try:
            converted = ipa.convert(w)
            ipa_words.append(w if '*' in converted else converted)
        except Exception:
            ipa_words.append(w)
    return f"/{' '.join(ipa_words)}/"

def fetch_dict_info(word):
    head = extract_headword(word)
    if not head: return "", "", "", ""
    encoded_word = urllib.parse.quote(head.lower())
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{encoded_word}"
    
    pos_str, eng_def, ex1, ex2 = "", "", "", ""
    try:
        res = requests.get(url, timeout=2)
        if res.status_code == 200:
            data = res.json()
            meanings = data[0].get("meanings", [])
            pos_list = list(set(m.get("partOfSpeech", "") for m in meanings if m.get("partOfSpeech")))
            pos_str = " / ".join(pos_list).title()
            if meanings and meanings[0].get("definitions"):
                eng_def = meanings[0]["definitions"][0].get("definition", "")
            examples = []
            for m in meanings:
                for d in m.get("definitions", []):
                    if "example" in d and d["example"]:
                        examples.append(d["example"])
            if len(examples) > 0: ex1 = examples[0]
            if len(examples) > 1: ex2 = examples[1]
    except Exception:
        pass
    return pos_str, eng_def, ex1, ex2

def predict_pos(word):
    w = str(word).lower().strip()
    if any(p in w for p in ['be ', 'take ', 'make ', 'run ', 'get ', 'set ', 'put ']) or w.startswith('to ') or w.endswith('ize') or w.endswith('ate'):
        return 'Verb'
    if w.endswith('tion') or w.endswith('ment') or w.endswith('ance') or w.endswith('ity') or w.endswith('er') or w.endswith('or'):
        return 'Noun'
    if w.endswith('able') or w.endswith('ible') or w.endswith('ive') or w.endswith('ous') or w.endswith('al') or w.endswith('ic'):
        return 'Adjective'
    if ' ' in w:
        return 'Phrase'
    return 'Noun / Verb'

def direct_google_translate(text, src='en', tgt='vi'):
    if is_invalid(text): return ""
    clean_text = str(text).strip()
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={src}&tl={tgt}&dt=t&q={urllib.parse.quote(clean_text)}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=4)
        if res.status_code == 200:
            data = res.json()
            translated = "".join([item[0] for item in data[0] if item and item[0]])
            if translated:
                return translated.strip()
    except Exception:
        pass
    return ""

def generate_smart_eng_def(word, meaning_vi, pos):
    clean_w = extract_headword(word)
    pos_lower = str(pos).lower()
    
    if not is_invalid(meaning_vi):
        if 'verb' in pos_lower:
            text_to_translate = f"Hành động {meaning_vi}"
        elif 'adj' in pos_lower:
            text_to_translate = f"Có tính chất {meaning_vi}"
        elif 'noun' in pos_lower:
            text_to_translate = f"Khái niệm hoặc vật dùng để {meaning_vi}"
        else:
            text_to_translate = f"Chỉ việc {meaning_vi}"
        
        translated = direct_google_translate(text_to_translate, src='vi', tgt='en')
        if translated:
            return translated.capitalize()

    return f"The definition or action related to {clean_w}."

def generate_varied_examples(word, pos, idx):
    w = extract_headword(word)
    if not w: w = str(word).strip()
    pos_lower = str(pos).lower()
    
    verb_templates = [
        (f"It is essential to {w} before making a final decision.", f"Our team will {w} as soon as possible."),
        (f"The manager asked us to {w} during the project review.", f"They plan to {w} by the end of this week."),
        (f"Please make sure to {w} according to the new guidelines.", f"We need to {w} to achieve better performance.")
    ]
    
    noun_templates = [
        (f"The new {w} has received positive feedback from clients.", f"You can find detailed information about the {w} in the report."),
        (f"A well-planned {w} can significantly improve overall efficiency.", f"They discussed the main features of the {w} during the meeting."),
        (f"Please provide all necessary documents regarding the {w}.", f"The company recently updated its policies on {w}.")
    ]
    
    adj_templates = [
        (f"This solution appears to be very {w} for our current situation.", f"The management is looking for a more {w} approach."),
        (f"It is {w} to ensure that all safety standards are met.", f"They presented a {w} strategy to increase quarterly sales."),
        (f"The overall results were surprisingly {w} despite the tight schedule.", f"She delivered a {w} presentation to the board members.")
    ]

    phrase_templates = [
        (f"Employees are encouraged to {w} when dealing with customers.", f"It is important to {w} to maintain high standards."),
        (f"The supervisor reminded everyone to {w} during daily operations.", f"They decided to {w} in order to resolve the issue."),
        (f"Please remember to {w} before submitting your final report.", f"Our primary goal is to {w} effectively this quarter.")
    ]

    t_idx = idx % 3
    if 'verb' in pos_lower:
        return verb_templates[t_idx]
    elif 'adj' in pos_lower:
        return adj_templates[t_idx]
    elif 'phrase' in pos_lower:
        return phrase_templates[t_idx]
    else:
        return noun_templates[t_idx]

print("Bắt đầu chuẩn hóa & Hoàn thiện toàn bộ dữ liệu...")

processed_count = 0

for idx in range(total_rows):
    word = df.at[idx, 'Từ vựng']
    if is_invalid(word): continue
    
    meaning_vi = df.at[idx, 'Meaning'] if 'Meaning' in df.columns else ""
    
    # 1. CẬP NHẬT CỘT TRÌNH ĐỘ CHUẨN A1 - C2
    if 'Trình độ' in df.columns:
        curr_lvl = str(df.at[idx, 'Trình độ'])
        # Điền nếu trống hoặc nếu đang chứa chuỗi cũ có dấu gạch/tiếng Việt
        if is_invalid(curr_lvl) or '-' in curr_lvl or 'Cơ bản' in curr_lvl or 'Nâng cao' in curr_lvl:
            curr_pos = df.at[idx, 'Tu loai'] if 'Tu loai' in df.columns and not is_invalid(df.at[idx, 'Tu loai']) else ""
            df.at[idx, 'Trình độ'] = predict_level(word, curr_pos)
    
    # 2. Điền IPA
    if is_invalid(df.at[idx, 'Ipa']):
        df.at[idx, 'Ipa'] = get_safe_ipa(word)
        
    # 3. Điền Loại từ
    if is_invalid(df.at[idx, 'Tu loai']):
        pos, eng_def, ex1, ex2 = fetch_dict_info(word)
        df.at[idx, 'Tu loai'] = pos if pos else predict_pos(word)
    else:
        pos, eng_def, ex1, ex2 = "", "", "", ""
    
    curr_pos = df.at[idx, 'Tu loai']
        
    # 4. Điền Nghĩa Tiếng Anh chuẩn
    if is_invalid(df.at[idx, 'Nghĩa Tiếng Anh']):
        if eng_def:
            df.at[idx, 'Nghĩa Tiếng Anh'] = eng_def
        else:
            df.at[idx, 'Nghĩa Tiếng Anh'] = generate_smart_eng_def(word, meaning_vi, curr_pos)
        
    # 5. Tạo Ví dụ Tiếng Anh
    fb1, fb2 = generate_varied_examples(word, curr_pos, idx)
    
    if is_invalid(df.at[idx, 'Ví dụ']):
        df.at[idx, 'Ví dụ'] = ex1 if ex1 else fb1
        
    if 'Ví dụ.1' in df.columns and is_invalid(df.at[idx, 'Ví dụ.1']):
        df.at[idx, 'Ví dụ.1'] = ex2 if ex2 else fb2

    # 6. Dịch Nghĩa Ví dụ sang Tiếng Việt
    need_tr1 = is_invalid(df.at[idx, 'Nghĩa ví dụ']) and not is_invalid(df.at[idx, 'Ví dụ'])
    need_tr2 = 'Ví dụ.1' in df.columns and 'Nghĩa ví dụ.1' in df.columns and is_invalid(df.at[idx, 'Nghĩa ví dụ.1']) and not is_invalid(df.at[idx, 'Ví dụ.1'])

    if need_tr1 or need_tr2:
        print(f"[{idx+1}/{total_rows}] Đang xử lý dịch: '{word}'...", flush=True)
        if need_tr1:
            df.at[idx, 'Nghĩa ví dụ'] = direct_google_translate(df.at[idx, 'Ví dụ'], src='en', tgt='vi')
        if need_tr2:
            df.at[idx, 'Nghĩa ví dụ.1'] = direct_google_translate(df.at[idx, 'Ví dụ.1'], src='en', tgt='vi')
        processed_count += 1

    # Lưu checkpoint định kỳ mỗi 20 dòng
    if (idx + 1) % 20 == 0:
        df.to_csv(output_file, index=False, encoding='utf-8-sig')

    time.sleep(0.01)

# Lưu kết quả hoàn thiện 100%
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n HOÀN TẤT! Toàn bộ cột đã được hoàn thiện tại: {output_file}")