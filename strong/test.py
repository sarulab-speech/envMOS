import torchaudio


wav1, sr1 = torchaudio.load("./data/wav/audiocaps/train/100012.mp3")
wav, sr = torchaudio.load("./data/wav/audioldm/train/92.wav")