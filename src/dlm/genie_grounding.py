import torch
from transformers.modeling_outputs import BaseModelOutput


def decode_with_backbone(backbone, tokenizer, latent: torch.Tensor, max_new_tokens: int = 180) -> str:
    enc_out = BaseModelOutput(last_hidden_state=latent)
    with torch.no_grad():
        out = backbone.generate(
            encoder_outputs=enc_out,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()
