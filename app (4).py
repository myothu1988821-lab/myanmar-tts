# Myanmar TTS — long-text version (အပိုင်းလိုက် ခွဲထုတ်ပေးတယ်)
# app.py နေရာမှာ ဒီဖိုင်ကို အစားထိုးသုံးနိုင်တယ်။ requirements.txt က အတူတူပဲ။
# ပြောင်းလဲချက်: စာသားရှည် (ဥပမာ ဇာတ်လမ်းအကျဉ်း ၃၀၀၀ ကျော်) ကို တစ်ခါတည်း
# မထုတ်ဘဲ စာပိုဒ်လိုက် ခွဲပြီး တစ်ပိုဒ်ချင်း ထုတ်ကာ ffmpeg နဲ့ ပြန်ဆက်ပေးတယ်။
# → edge-tts ရဲ့ 60 စက္ကန့် timeout မိမှာ စိုးရိမ်စရာ မလိုတော့ဘူး
# → အပိုင်းတိုင်း progress ပြတယ်၊ ဘယ်အပိုင်းမှာ ရပ်သွားလဲ သိရတယ်

import streamlit as st, asyncio, subprocess, tempfile, os, re
from pathlib import Path

st.set_page_config(page_title="Myanmar TTS", page_icon="🎙️", layout="centered")

CSS="""<style>
.stApp{background:linear-gradient(135deg,#0f2027 0%,#203a43 55%,#2c5364 100%)}
.stApp h1{background:linear-gradient(90deg,#7fe7dc,#ffd76e);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800}
.stApp p,.stApp label,.stMarkdown{color:#eaf6f6!important}
div.stButton>button{background:linear-gradient(90deg,#00bfa5,#00acc1);color:#fff;border:none;border-radius:12px;padding:.6rem 2rem;font-size:1.1rem;font-weight:700;box-shadow:0 4px 14px rgba(0,191,165,.35);width:100%}
div.stButton>button:hover{filter:brightness(1.12)}
div[data-testid="stTextArea"] textarea{background-color:#16323c!important;color:#fff!important;border-radius:12px;border:1px solid rgba(255,255,255,.3)!important;font-size:1.05rem}
div[data-testid="stTextArea"] textarea::placeholder{color:rgba(255,255,255,.45)!important}
div[data-testid="stSelectbox"] div[data-baseweb="select"]>div{background-color:#16323c!important;border-radius:12px}
div[data-testid="stSelectbox"] span{color:#fff!important}
div[data-testid="stSlider"]{color:#fff}
header[data-testid="stHeader"]{background:rgba(0,0,0,0)}
footer{visibility:hidden}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)
st.title("🎙️ Myanmar TTS")
st.markdown("📝 <b>မြန်မာစာ</b> → 🔊 <b>အသံ MP3</b>", unsafe_allow_html=True)
st.write("")

VOICES={"Nilar (မိန်းကလေးအသံ)":"my-MM-NilarNeural",
        "Thiha (ယောက်ျားအသံ)":"my-MM-ThihaNeural",
        "MMS (offline အသံ)":"mms"}

def split_chunks(text, max_chars=600):
    """စာသားကို စာပိုဒ်လိုက် ခွဲတယ်။ စာပိုဒ်တစ်ခုက ရှည်လွန်းရင် ဝါကျအဆုံးမှာ ထပ်ခွဲတယ်။"""
    paras=[p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    chunks=[]
    for p in paras:
        if len(p)<=max_chars:
            chunks.append(p); continue
        cur=""
        for s in re.split(r"(?<=[။.!?])\s*", p):
            if not s: continue
            if len(cur)+len(s)<=max_chars:
                cur+=s
            else:
                if cur: chunks.append(cur)
                cur=s
        if cur: chunks.append(cur)
    return chunks or [text.strip()]

async def _edge(text, voice, out):
    import edge_tts
    await asyncio.wait_for(edge_tts.Communicate(text, voice).save(out), timeout=60)

def synth_edge(text, voice, out_mp3):
    asyncio.run(_edge(text, voice, out_mp3))

@st.cache_resource(show_spinner="MMS model ဒေါင်းနေတယ်...")
def get_tts():
    import sherpa_onnx
    d=Path.home()/".cache"/"mm-tts"; d.mkdir(parents=True,exist_ok=True)
    mo,tk=d/"model.onnx",d/"tokens.txt"
    if not mo.exists():
        import urllib.request
        B="https://huggingface.co/willwade/mms-tts-multilingual-models-onnx/resolve/main"
        urllib.request.urlretrieve(f"{B}/mya/model.onnx",mo)
        urllib.request.urlretrieve(f"{B}/mya/tokens.txt",tk)
    cfg=sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(mo),tokens=str(tk),lexicon="",data_dir="",dict_dir="")),
        max_num_sentences=1)
    return sherpa_onnx.OfflineTts(cfg)

def synth_mms_chunk(tts, text, speed=1.0):
    import numpy as np
    au=tts.generate(text,sid=0,speed=speed)
    return np.array(au.samples,dtype=np.float32), au.sample_rate

def srt_to_text(content):
    lines=[]
    for ln in content.splitlines():
        t=ln.strip()
        if not t: continue
        if re.match(r'^\d+$',t): continue
        if '-->' in t: continue
        lines.append(t)
    return "\n\n".join(lines)

up=st.file_uploader("📄 SRT file တင်မယ် (မတင်လည်း ရတယ်)",type=["srt"])
srt_text=""
if up:
    try:
        srt_text=srt_to_text(up.read().decode("utf-8-sig"))
        st.success(f"SRT ဖတ်ပြီးပြီ ✓ ({len(srt_text)} စာလုံး)")
    except Exception as e:
        st.error(f"SRT ဖတ်မရဘူး: {e}")

text=st.text_area("စာသား",value=srt_text,height=200,placeholder="အသံထွက်ချင်တဲ့ မြန်မာစာကို ဒီမှာ ရေးပါ... (သို့မဟုတ် အပေါ်က SRT file တင်ပါ)")
vname=st.selectbox("အသံ ရွေးပါ",list(VOICES.keys()))
speed=st.slider("အသံအမြန်နှုန်း",0.7,1.3,1.0,0.05)

if st.button("🔊 အသံထုတ်မယ်",type="primary"):
    if not text.strip():
        st.error("စာသား ထည့်ပေးပါ။");st.stop()
    chunks=split_chunks(text)
    if len(chunks)>1:
        st.caption(f"📄 စာသားရှည်လို့ အပိုင်း {len(chunks)} ပိုင်း ခွဲထုတ်ပေးမယ်")
    voice=VOICES[vname]
    tmp=tempfile.mkdtemp()
    out_mp3=os.path.join(tmp,"out.mp3")
    status=st.empty()
    bar=st.progress(0)
    try:
        if voice=="mms":
            import numpy as np, wave
            tts=get_tts()
            pcms,sr=[],None
            for i,ch in enumerate(chunks):
                status.info(f"🎙️ အပိုင်း {i+1}/{len(chunks)} ထုတ်နေတယ်...")
                pcm,s=synth_mms_chunk(tts,ch,speed=speed)
                pcms.append(pcm); sr=s
                bar.progress((i+1)/len(chunks),f"အပိုင်း {i+1}/{len(chunks)}")
            pcm=np.concatenate(pcms)
            pcm=pcm/np.max(np.abs(pcm))*0.9
            wav=os.path.join(tmp,"out.wav")
            w=wave.open(wav,"w");w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr)
            w.writeframes((pcm*32767).astype(np.int16).tobytes());w.close()
            status.info("🔗 အသံဆက်နေတယ်...")
            subprocess.run(["ffmpeg","-y","-v","error","-i",wav,
                "-filter:a",f"atempo={speed:.2f}","-c:a","libmp3lame","-q:a","4",out_mp3],
                check=True,stdin=subprocess.DEVNULL,capture_output=True)
        else:
            cps=[]
            for i,ch in enumerate(chunks):
                status.info(f"🌐 {vname} — အပိုင်း {i+1}/{len(chunks)} ထုတ်နေတယ်...")
                cp=os.path.join(tmp,f"c{i:02d}.mp3")
                synth_edge(ch,voice,cp)
                cps.append(cp)
                bar.progress((i+1)/len(chunks),f"အပိုင်း {i+1}/{len(chunks)}")
            status.info("🔗 အသံဆက်နေတယ်...")
            lst=os.path.join(tmp,"list.txt")
            with open(lst,"w",encoding="utf-8") as f:
                for cp in cps:
                    f.write("file '"+cp.replace("'","'\\''")+"'\n")
            subprocess.run(["ffmpeg","-y","-v","error","-f","concat","-safe","0","-i",lst,
                "-c:a","libmp3lame","-q:a","4",out_mp3],
                check=True,stdin=subprocess.DEVNULL,capture_output=True)
            if abs(speed-1.0)>0.01:
                tmp2=os.path.join(tmp,"out2.mp3")
                subprocess.run(["ffmpeg","-y","-v","error","-i",out_mp3,
                    "-filter:a",f"atempo={speed:.2f}","-c:a","libmp3lame","-q:a","4",tmp2],
                    check=True,stdin=subprocess.DEVNULL,capture_output=True)
                os.replace(tmp2,out_mp3)
        status.empty(); bar.empty()
        st.success("ပြီးပြီ! ✓")
        st.audio(out_mp3,format="audio/mp3")
        with open(out_mp3,"rb") as f:
            st.download_button("⬇️ MP3 ဒေါင်းမယ်",f,file_name="myanmar_tts.mp3",mime="audio/mp3")
    except Exception as e:
        status.empty(); bar.empty()
        st.error(f"Error: {str(e)[:300]}")
        if voice!="mms":
            st.info("အွန်လိုင်းအသံ မရလို့ MMS (offline) ကို ရွေးကြည့်ပါ။")
