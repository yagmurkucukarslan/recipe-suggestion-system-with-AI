import os
import asyncio
import speech_recognition as sr
import joblib
import numpy as np
import pandas as pd
from gtts import gTTS
from core import Core
import time

# Model ve veritabanını yükleme
model_data = joblib.load("model.joblib")
model_title = model_data["model_title"]  
malzeme_sutunlari = model_data["malzemeler"]  
titles = model_data["titles"] 
tarif_veritabani = pd.read_csv("guncellenmis_dosyatarifdetayicin.csv")

from sklearn.preprocessing import LabelEncoder
le_title = LabelEncoder()
le_title.classes_ = np.array(titles)


async def play_sound_from_text(core, text: str):
    """Metni sese dönüştür ve ses dosyasını çal."""
    audio_dir = "audio_files"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    audio_path = os.path.join(audio_dir, f"temp_message_{time.time()}.mp3")
    try:
        # TTS işlemi ve dosyayı kaydetme
        tts = gTTS(text, lang='tr')
        tts.save(audio_path)

        # Ses dosyasını çalma
        await core.play_sound(audio_path)

        # Çalma bitmeden önce dosya silinmemesi için kısa bir bekleme


        asyncio.create_task(delete_file(audio_path))

    except Exception as e:
        print(f"Hata oluştu: {e}")

async def delete_file(file_path):
    try:
        os.remove(file_path)
    except OSError as e:
        print(f"Dosya silinemedi: {e}")

# Kullanıcıdan kategori seçimi alma
async def kategori_secme(core):
    await core.set_state("Kategori seçimi yapılıyor...")
    await play_sound_from_text(core, "Lütfen bir kategori seçiniz: ana yemek, çorba, salata, tatlı, hamur işi, kahvaltı, reçel, zayıflama.")
    r = sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            await core.set_state("Kullanıcıdan sesli kategori seçimi bekleniyor...")
            await play_sound_from_text(core, "Kategori seçmek için sesli komut veriniz.")
            try:
                ses = r.listen(source)
                metin = r.recognize_google(ses, language='tr-TR')
                secilen_kategori = metin.lower()
                if secilen_kategori:
                    await core.set_state(f"Seçilen kategori: {secilen_kategori}")
                    await play_sound_from_text(core, f"Seçilen kategori: {secilen_kategori}")
                    return secilen_kategori
                else:
                    await core.set_state("Geçersiz kategori seçimi yapıldı.")
                    await play_sound_from_text(core, "Geçersiz kategori. Lütfen tekrar deneyin.")
            except sr.UnknownValueError:
                await core.set_state("Ses anlaşılamadı.")
                await play_sound_from_text(core, "Ses anlaşılamadı, lütfen tekrar deneyin.")
            except sr.RequestError as e:
                await core.set_state(f"Servis hatası: {e}")
                await play_sound_from_text(core, f"Servis hatası: {e}")
                break

async def malzeme_toplama(core):
    await core.set_state("Malzeme toplama işlemi başladı...")
    r = sr.Recognizer()
    malzemeler = []
    await play_sound_from_text(core, "Lütfen malzemelerinizi söyleyin. Malzemeleri bitirdiğinizde 'bitti' diyebilirsiniz.")
    while True:
        with sr.Microphone() as source:
            await core.set_state("Kullanıcıdan malzeme bilgisi bekleniyor...")
            await play_sound_from_text(core, "Malzemelerinizi söyleyin:")
            try:
                ses = r.listen(source)
                metin = r.recognize_google(ses, language='tr-TR')
                if 'bitti' in metin.lower():
                    await core.set_state("Malzeme toplama işlemi sona erdi.")
                    break
                else:
                    temiz_metin = metin.lower().strip()
                    for malzeme in malzeme_sutunlari:
                        if malzeme in temiz_metin:
                            if malzeme not in malzemeler:
                                malzemeler.append(malzeme)
                                await core.set_state(f"Eklenen malzeme: {malzeme}")
                                await play_sound_from_text(core, f"Eklenen malzeme: {malzeme}")
            except sr.UnknownValueError:
                await core.set_state("Ses anlaşılamadı.")
                await play_sound_from_text(core, "Ses anlaşılamadı, lütfen tekrar deneyin.")
            except sr.RequestError as e:
                await core.set_state(f"Servis hatası: {e}")
                await play_sound_from_text(core, f"Servis hatası: {e}")
                break
    return malzemeler 

async def yemek_onerisi(core, kategori, malzemeler, onceki_yemek=None):
    await core.set_state("Yemek önerisi yapılıyor...")
    data = pd.DataFrame(
        [[int(malzeme in malzemeler) for malzeme in malzeme_sutunlari]],
        columns=malzeme_sutunlari
    )
    probas = model_title.predict_proba(data)
    tahmin_sirasi = probas[0].argsort()[::-1]
    
    for tahmin_idx in tahmin_sirasi:
        yemek_tahmini = le_title.inverse_transform([tahmin_idx])[0]
        if yemek_tahmini != onceki_yemek:
            await core.set_state(f"Önerilen yemek: {yemek_tahmini}")
            await play_sound_from_text(core, f"Önerilen tarif: {yemek_tahmini}")
            return yemek_tahmini
    await core.set_state("Başka yemek bulunamadı.")
    await play_sound_from_text(core, "Başka yemek bulunamadı.")
    return onceki_yemek

async def tarif_detaylari(core, yemek_adi, kategori, malzemeler):
    await core.set_state(f"{yemek_adi} tarifi için kullanıcıdan yanıt bekleniyor...")
    r = sr.Recognizer()
    await play_sound_from_text(core, f"{yemek_adi} yemeğinin tarifini ister misiniz? Evet veya hayır olarak yanıtlayın.")
    while True:
        with sr.Microphone() as source:
            try:
                ses = r.listen(source)
                metin = r.recognize_google(ses, language='tr-TR').lower()

                if "evet" in metin:
                    # Veritabanından tarif çek
                    tarif = tarif_veritabani.loc[tarif_veritabani['Title'] == yemek_adi, 'How-to-do'].values
                    if len(tarif) > 0:
                        await core.set_state(f"{yemek_adi} tarifi sunuluyor...")
                        await play_sound_from_text(core, f"{yemek_adi} tarifi:\n{tarif[0]}")
                    else:
                        await core.set_state(f"{yemek_adi} tarifi bulunamadı.")
                        await play_sound_from_text(core, f"{yemek_adi} için tarif bulunamadı.")
                    return "tamam"  # İşlem tamam
                elif "hayır" in metin:
                    await core.set_state("Kullanıcı tarif istemedi, yeni öneri yapılıyor.")
                    # Yeni yemek önerisi için devam sinyali döndür
                    return "devam"
                else:
                    await core.set_state("Kullanıcıdan geçersiz yanıt alındı.")
                    await play_sound_from_text(core, "Lütfen sadece evet veya hayır deyin.")
            except sr.UnknownValueError:
                await core.set_state("Ses anlaşılamadı.")
                await play_sound_from_text(core, "Ses anlaşılamadı, lütfen tekrar deneyin.")
            except sr.RequestError as e:
                await core.set_state(f"Servis hatası: {e}")
                await play_sound_from_text(core, f"Servis hatası: {e}")
                break


async def main(core: Core):
    while True:
        kategori = await kategori_secme(core)
        onceki_yemek = None
        
        while True:
            malzemeler = await malzeme_toplama(core)
            onceki_yemek = await yemek_onerisi(core, kategori, malzemeler, onceki_yemek)

            # Detayları sormak için tarif_detaylari fonksiyonunu çağırıyoruz
            detay_sonucu = await tarif_detaylari(core, onceki_yemek, kategori, malzemeler)
            
            if detay_sonucu == "tamam":
                break
            
            # Kullanıcı "Hayır" yanıtını verdiyse farklı bir yemek önerisi yapılır
            while detay_sonucu == "devam":
                onceki_yemek = await yemek_onerisi(core, kategori, malzemeler, onceki_yemek)
                detay_sonucu = await tarif_detaylari(core, onceki_yemek, kategori, malzemeler)


if __name__ == "__main__":
    core = Core()  
    asyncio.run(main(core))