import os
import subprocess
import re
import requests
from bs4 import BeautifulSoup
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from opencc import OpenCC

#  頻道播放清單頁面
channel_playlists_url = "https://www.youtube.com/@icare%E6%84%9B%E5%81%A5%E5%BA%B7/playlists"

#  資料夾設定
base_folder = "D:/rag"
videos_folder = os.path.join(base_folder, "videos")
output_folder = os.path.join(base_folder, "output")
os.makedirs(videos_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

#  繁體中文轉換器
cc = OpenCC("s2t")

#  取得所有播放清單連結
def get_all_playlist_links(channel_url):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(channel_url, headers=headers)
    matches = re.findall(r"/playlist\?list=[\w-]+", res.text)
    unique_playlists = sorted(set(matches))
    return ["https://www.youtube.com" + m for m in unique_playlists]

#  初始化模型
print("📦 載入語音模型中...")
model = AutoModel(
    model="FunAudioLLM/SenseVoiceSmall",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu",
    hub="hf",
)

#  處理播放清單
#  手動指定播放清單網址列表
playlist_links = [
    "https://www.youtube.com/watch?v=xwFqHXxQYJw&list=PLn4BRs6TsL8OCjHiJttO_YMawiQQjWCgH",
    "https://www.youtube.com/watch?v=r9ku8IWjro4&list=PLn4BRs6TsL8MIuxmswCG7RUcZhRd97kBz",
    "https://www.youtube.com/watch?v=aNx_RgNvacM&list=PLn4BRs6TsL8P6uiJI1RDv4AJ4NfcVNWHN",
    "https://www.youtube.com/watch?v=M47ulUpFr5c&list=PLn4BRs6TsL8M8Qk7au_MFcBBtr9kvl8gg",
    "https://www.youtube.com/watch?v=lym6Wfbhi0E&list=PLn4BRs6TsL8Ns7JuhCMweFn1Q0Zc6-T1s&pp=0gcJCWMEOCosWNin",
    "https://www.youtube.com/watch?v=E6fJjYCXUX4&list=PLn4BRs6TsL8M8Bk_Z2mxksBVmAgEqdL6Y",
    "https://www.youtube.com/watch?v=w4NPy71S6iM&list=PLn4BRs6TsL8O2WlnDiuW4obEtc3jV8NOD",
    "https://www.youtube.com/watch?v=PTBsVtDV6mE&list=PLn4BRs6TsL8NXQOvLJCoGvuqeP9W-Utay",
    "https://www.youtube.com/watch?v=8zYjdWycv30&list=PLn4BRs6TsL8MsQfgu2qskJxqAoN6ABDzO&pp=0gcJCWMEOCosWNin",
    "https://www.youtube.com/watch?v=MC0ouDIKmT0&list=PLn4BRs6TsL8NuOn3dWvZOQLELXM_mTT8u",
    "https://www.youtube.com/watch?v=ZgR_lYsWSTo&list=PLn4BRs6TsL8MpPWra4pN5ovZnPUDpcHjN",
    "https://www.youtube.com/watch?v=qONJBt_6uy4&list=PLn4BRs6TsL8P7Aa9mF5_lpLGZra-_R1Si",
    "https://www.youtube.com/watch?v=enLwCDVvu4w&list=PLn4BRs6TsL8PWjuxuAy0a7O8JSaNTpAyZ",
    "https://www.youtube.com/watch?v=UfYl-wGiM7o&list=PLn4BRs6TsL8Ni-nmPnFPNxi9aCDq8aVRr&pp=0gcJCWMEOCosWNin",
    "https://www.youtube.com/watch?v=_XjfnIGDxPI&list=PLn4BRs6TsL8PNL8vdggbJkdUejZMoM4SQ",
    "https://www.youtube.com/watch?v=A1O6Pp3Tv00&list=PLn4BRs6TsL8P66OUIeVSMtbo6V_5Eswh4"
]
print(f"🔍 共手動設定 {len(playlist_links)} 個播放清單，準備處理...")


total_transcribed = 0
total_skipped = 0

for idx, playlist_url in enumerate(playlist_links):
    print(f"\n📋 [{idx+1}/{len(playlist_links)}] 處理播放清單：{playlist_url}")

    # 🔍 下載前記錄現有檔案
    existing_files = set(os.listdir(videos_folder))

    #  下載音訊
    download_cmd = [
        "yt-dlp",
        "--yes-playlist",
        "-x", "--audio-format", "wav",
        "--no-post-overwrites",
        "-o", os.path.join(videos_folder, f"%(playlist_index)s_%(title)s.%(ext)s"),
        playlist_url,
    ]
    subprocess.run(download_cmd, check=True)

    #  抓取新下載的檔案
    all_files = set(os.listdir(videos_folder))
    new_audio_files = sorted([f for f in (all_files - existing_files) if f.endswith(".wav")])

    print(f"🎧 本播放清單共下載 {len(new_audio_files)} 支影片，開始語音辨識...")

    for i, audio_file in enumerate(new_audio_files):
        print(f"📝 處理第 {i+1}/{len(new_audio_files)} 支影片：{audio_file}")
        audio_path = os.path.join(videos_folder, audio_file)
        base_name = os.path.splitext(audio_file)[0]
        output_txt_path = os.path.join(output_folder, base_name + ".txt")

        if os.path.exists(output_txt_path):
            print(f"⚠️ 已存在文字檔，略過：{output_txt_path}")
            total_skipped += 1
            continue

        try:
            res = model.generate(
                input=audio_path,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            text = rich_transcription_postprocess(res[0]["text"])
            text = cc.convert(text)

            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"✅ 已轉錄完成：{audio_file}")
            total_transcribed += 1
        except Exception as e:
            print(f"❌ 發生錯誤於 {audio_file}：{e}")

#  統計回報
print("\n🎉 所有播放清單處理完畢！")
print(f"📑 成功轉錄：{total_transcribed} 支影片")
print(f"🟡 已略過（已有 txt）：{total_skipped} 支影片")