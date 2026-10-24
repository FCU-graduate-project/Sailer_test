"""Padding-masked variant of the Crab head (MultiModalEmotionClassifierDeep).

Same modules and state dict as the official head; only the forward differs: zero padding is
excluded from the BiGRUs (packed sequences), cross-attention (key_padding_mask), attention
pooling (its existing mask argument) and the frame-level contrastive means, for audio and
text alike. Without padding the output equals the original forward exactly, so a prediction
no longer depends on how a clip is batched.
"""
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .ser import MultiModalEmotionClassifierDeep


def speech_frame_mask(ssl_model, sample_mask, n_frames):
    """Bool [B, n_frames] frame mask of an SSL encoder output from its sample-level mask."""
    lengths = ssl_model._get_feat_extract_output_lengths(sample_mask.sum(1).long())
    if int(lengths.max()) > n_frames:
        raise RuntimeError("frame length mismatch")
    return torch.arange(n_frames, device=sample_mask.device)[None] < lengths.to(sample_mask.device)[:, None]


def _packed_gru(gru, x, mask):
    lengths = mask.sum(1).cpu()
    with torch.autocast("cuda", enabled=False):  # packed cuDNN GRU in fp32
        packed = pack_padded_sequence(x.float(), lengths, batch_first=True, enforce_sorted=False)
        out, _ = gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
    return out


def _masked_mean(x, mask):
    weight = mask.unsqueeze(-1).to(x.dtype)
    return (x * weight).sum(1) / weight.sum(1).clamp(min=1)


class MaskedMultiModalEmotionClassifierDeep(MultiModalEmotionClassifierDeep):
    def forward(self, features1, features2, return_embeddings=False, speech_mask=None, text_mask=None):
        """speech_mask / text_mask: bool [B, T], True = real frame or token (both required)."""
        if speech_mask is None or text_mask is None:
            raise ValueError("MaskedMultiModalEmotionClassifierDeep needs speech_mask and text_mask")
        speech_mask, text_mask = speech_mask.bool(), text_mask.bool()
        speech_proj = self.speech_norm(self.speech_projection(features1))
        text_proj = self.text_norm(self.text_projection(features2))
        speech_hidden = _packed_gru(self.speech_gru, speech_proj, speech_mask)
        text_hidden = _packed_gru(self.text_gru, text_proj, text_mask)
        speech_attended, _ = self.speech_attention(speech_hidden, text_hidden, text_hidden,
                                                   key_padding_mask=~text_mask)
        text_attended, _ = self.text_attention(text_hidden, speech_hidden, speech_hidden,
                                               key_padding_mask=~speech_mask)
        speech_pooled = self.attention_pool(speech_hidden + speech_attended, self.speech_attn, mask=speech_mask)
        text_pooled = self.attention_pool(text_hidden + text_attended, self.text_attn, mask=text_mask)
        normalized = self.layer_norm(torch.cat([speech_pooled, text_pooled], dim=-1))
        hidden = self.classifier_fc1(normalized)
        logits = self.classifier_fc2(self.classifier_dropout(self.classifier_relu(hidden)))
        if not return_embeddings:
            return logits
        return logits, {
            "speech_frame_emb": self.speech_contrastive_mlp(_masked_mean(speech_proj, speech_mask)),
            "text_frame_emb": self.text_contrastive_mlp(_masked_mean(text_proj, text_mask)),
            "speech_pooled_emb": self.speech_pooled_contrastive_mlp(speech_pooled),
            "text_pooled_emb": self.text_pooled_contrastive_mlp(text_pooled),
            "fusion_emb": self.fusion_contrastive_mlp(hidden),
            "normalized": normalized,
        }
