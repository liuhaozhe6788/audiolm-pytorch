import torch
from audiolm_pytorch import HubertWithKmeans, EncodecWrapper, CoarseTransformer, CoarseTransformerTrainer

wav2vec = HubertWithKmeans(
    checkpoint_path = './hubert/hubert_base_ls960.pt',
    kmeans_path = './hubert/hubert_base_ls960_L9_km500.bin'
)

encodec = EncodecWrapper()

coarse_transformer = CoarseTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 512,
    depth = 6
)

trainer = CoarseTransformerTrainer(
    transformer = coarse_transformer,
    codec = encodec,
    wav2vec = wav2vec,
    folder = 'audio',
    batch_size = 4,
    data_max_length = 3200 * 32,
    num_train_steps = 1_000
)

trainer.train()