# Integration tests

These tests require:
- A GPU (CUDA-capable device)
- Real API credentials in `.env` (WANDB_API_KEY, HF_TOKEN)
- HuggingFace access to Meta-gated models (Llama-3.1-8B-Instruct, Llama-Guard-3-8B)

Run with:

    pytest tests/integration --integration -v

Every test here carries the `@pytest.mark.integration` marker. CI skips them unless
you pass the `--integration` flag.
