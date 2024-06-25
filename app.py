from flask import Flask, render_template, request, redirect,url_for,session
from tacotron2.inference import (
    load_tacotron2,
    load_hifigan,
    load_waveglow,
    text_to_sequence,
)
from vits.inference import get_text, load_vits, get_hps
import numpy as np
import matplotlib.pyplot as plt
import torch
import soundfile as sf
import json
import matplotlib.pyplot as plt
import librosa
import threading


app = Flask(__name__)


def plot_image(audio):
    librosa.display.waveshow(audio, sr=22050, color="orange")
    plt.savefig('./static/images/audio.jpg',dpi=100)
    fbank = librosa.feature.melspectrogram(
        y=audio, sr=22050, n_fft=1024, hop_length=1024, win_length=256, n_mels=80
    )
    fbank_db = librosa.power_to_db(fbank, ref=np.max)
    plt.imshow(fbank_db, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar()
    plt.savefig('./static/images/spec.jpg',dpi=100)


def audio_generate(model, text, voice):
    if model == "vits":
        hps = get_hps("D:/图片/临时/config.json")
        vits = load_vits(hps)
        stn_tst = get_text(text, hps)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            if voice=='atri':
                num=23
            elif voice=='yoshino':
                num=20
            elif voice=='murasame':
                num=21
            elif voice=='mako':
                num=22
            sid = torch.LongTensor([num]).cuda()
            audio = (
                vits.infer(
                    x_tst,
                    x_tst_lengths,
                    sid=sid,
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1,
                )[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )
            t=threading.Thread(target=plot_image,args=[audio])
            t.start()
            sf.write("./static/audios/sukisuki.wav", audio, samplerate=22050)
    else:
        sequence = np.array(text_to_sequence(text, ["japanese_cleaners"]))[None, :]
        print(sequence)
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        tacotron2 = load_tacotron2()
        mel_outputs, mel_outputs_postnet, _, alignments = tacotron2.inference(sequence)
        if model == "taco-wave":
            waveglow, denoiser = load_waveglow()
            with torch.no_grad():
                audio = waveglow.infer(mel_outputs_postnet, sigma=0.9)
                audio_denoised = denoiser(audio, strength=0.01)[:, 0]
                audio = audio_denoised[0].data.cpu().numpy()
                t=threading.Thread(target=plot_image,args=[audio])
                t.start()
                sf.write("./static/audios/sukisuki.wav", audio, 22050)
        elif model == "taco-hifi":
            HiFi_GAN = load_hifigan()
            with torch.no_grad():
                y = HiFi_GAN.inference(mel_outputs_postnet)
                t=threading.Thread(target=plot_image,args=[audio])
                t.start()
                sf.write("./static/audios/sukisuki.wav", ｙ, 22050)
    return "success"


@app.route("/")
def index():
    models=request.args.get('models','')    
    print(models)
    if models:
        models = json.loads(models)
    else:
        pass
    text=request.args.get('text','')
    voice=request.args.get('voice','')
    message=request.args.get('message','')
    print(models)
    return render_template(
        "index.html", models=models, text=text, voice=voice, message=message
    )


@app.route("/synthesis", methods=["POST"])
def synthesis():
    form = request.form
    model = form.get("model")
    print(model)
    text = form.get("text")
    print(text)
    voice = form.get("voice")
    print(voice)
    message = audio_generate(model, text, voice)
    if model == "taco-wave":
        models = {
            "taco-wave": "Tacotron2+Waveglow",
            "taco-hifi": "Tacotron2+HiFi-GAN",
            "vits": "Vits",
        }
    elif model == "taco-hifi":
        models = {
            "taco-hifi": "Tacotron2+HiFi-GAN",
            "taco-wave": "Tacotron2+Waveglow",
            "vits": "Vits",
        }
    elif model=='vits':
        models = {
            "vits": "Vits",
            "taco-hifi": "Tacotron2+HiFi-GAN",
            "taco-wave": "Tacotron2+Waveglow",
        }
    return redirect(
        url_for("index", models=json.dumps(models), text=text, voice=voice, message=message)
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=152, threaded=True)
