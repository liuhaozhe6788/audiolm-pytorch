import torch
from audiolm_pytorch import EncodecWrapper, FineTransformer, FineTransformerTrainer

encodec = EncodecWrapper()

fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 512,
    depth = 6
)

trainer = FineTransformerTrainer(
    transformer = fine_transformer,
    codec = encodec,
    folder = 'audio',
    batch_size = 4,
    data_max_length = 3200 * 32,
    num_train_steps = 1_000
)

trainer.train()