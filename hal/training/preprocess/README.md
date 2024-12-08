Dimension key:
- B: batch size
- T: trajectory length (i.e. partially preprocessed sequence sampled from dataset or closed loop eval buffer size)
- L: sequence length (i.e. training-ready sequence length)
- D: model dimension (sometimes called d_model or embedding_dim)
- G: preprocessed gamestate or input size, analogous to vocabulary size
- C: controller input size or target size, number of classes