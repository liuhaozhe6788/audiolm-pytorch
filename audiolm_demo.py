import os
import soundfile as sf

from audiolm_pytorch import AudioLM, HubertWithKmeans, EncodecWrapper, SemanticTransformer, CoarseTransformer, FineTransformer

encodec = EncodecWrapper()

# hubert checkpoints can be downloaded at
# https://github.com/facebookresearch/fairseq/tree/main/examples/hubert

wav2vec = HubertWithKmeans(
    checkpoint_path = './hubert/hubert_base_ls960.pt',
    kmeans_path = './hubert/hubert_base_ls960_L9_km500.bin'
)

semantic_transformer = SemanticTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    dim = 1024,
    depth = 6
).cuda()

fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 512,
    depth = 6
).cuda()

coarse_transformer = CoarseTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 512,
    depth = 6
).cuda()

audiolm = AudioLM(
    wav2vec = wav2vec,
    codec = encodec,
    semantic_transformer = semantic_transformer,
    coarse_transformer = coarse_transformer,
    fine_transformer = fine_transformer
).cuda()

speaker_name = "unknown"
# generated_wav = audiolm(batch_size = 1)

# or with priming

# generated_wav_with_prime = audiolm(prime_wave = )

# or with text condition, if given

generated_wav_with_text_condition = audiolm(batch_size=1, text = ['The North Wind and the Sun were disputing which was the stronger, when a traveler came along wrapped in a warm cloak.'])

# Save it on the disk
# filename = "demo_output_%02d.wav" % num_generated
if not os.path.exists("out_audios"):
    os.mkdir("out_audios")

dir_path = os.path.dirname(os.path.realpath(__file__))  # current dir 
filename = os.path.join(dir_path, f"out_audios/{speaker_name}_syn.wav")
# print(wav.dtype)
sf.write(filename, generated_wav_with_text_condition.cpu().detach().numpy()[0], 24_000)
