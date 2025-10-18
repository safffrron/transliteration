import os
import re
import warnings
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Suppress warnings
warnings.filterwarnings('ignore')

# ================================
# LSTM MODEL DEFINITIONS
# ================================
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers=1, 
                 dropout=0.1, bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.embed = nn.Embedding(input_dim, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
    def forward(self, src, src_lens):
        embedded = self.embed(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, c_n) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out, (h_n, c_n)

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim, encoder_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(encoder_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        
    def forward(self, decoder_hidden, encoder_outputs, src_lens):
        batch_size, src_len, _ = encoder_outputs.size()
        decoder_hidden = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)
        energy = torch.tanh(self.W1(decoder_hidden) + self.W2(encoder_outputs))
        scores = self.V(energy).squeeze(2)
        mask = torch.arange(src_len, device=scores.device).unsqueeze(0) >= src_lens.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e10)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        return context, attn_weights

class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, encoder_dim,
                 num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(output_dim, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim, encoder_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim + encoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.out = nn.Linear(hidden_dim + encoder_dim + embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, dec_in, hidden, encoder_outputs, src_lens):
        embedded = self.dropout(self.embed(dec_in))
        batch_size, seq_len, _ = embedded.size()
        outputs = []
        h, c = hidden
        
        for t in range(seq_len):
            emb_t = embedded[:, t:t+1, :]
            h_prev = h[-1]
            context, attn_weights = self.attention(h_prev, encoder_outputs, src_lens)
            context = context.unsqueeze(1)
            lstm_input = torch.cat([emb_t, context], dim=2)
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            combined = torch.cat([lstm_out, context, emb_t], dim=2)
            output = self.out(combined)
            outputs.append(output)
        
        outputs = torch.cat(outputs, dim=1)
        return outputs, (h, c)

# ================================
# TRANSFORMER MODEL DEFINITIONS
# ================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class ImprovedTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=2048,
                 dropout=0.1, activation="relu", max_seq_length=128, layer_norm_eps=1e-5):
        super().__init__()
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps,
            batch_first=True, norm_first=False
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model, eps=layer_norm_eps)
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps,
            batch_first=True, norm_first=False
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model, eps=layer_norm_eps)
        )
        
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self.output_proj.weight = self.tgt_embed.weight
        self._init_parameters()
    
    def _init_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'embed' in name:
                    nn.init.normal_(p, mean=0, std=self.d_model ** -0.5)
                else:
                    nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_emb = self.embed_dropout(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.embed_dropout(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        tgt_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        output = self.decoder(
            tgt_emb, memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        logits = self.output_proj(output)
        return logits
    
    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

# ================================
# INFERENCE FUNCTIONS
# ================================
def prepare_decoder_hidden(h_n, c_n, enc, dec, device):
    batch_size = h_n.size(1)
    num_layers = enc.num_layers
    
    if enc.bidirectional:
        h_n = h_n.view(num_layers, 2, batch_size, enc.hidden_dim)
        c_n = c_n.view(num_layers, 2, batch_size, enc.hidden_dim)
        h_n = torch.cat([h_n[:, 0, :, :], h_n[:, 1, :, :]], dim=2)
        c_n = torch.cat([c_n[:, 0, :, :], c_n[:, 1, :, :]], dim=2)
        
        if h_n.size(2) != dec.hidden_dim:
            proj_h = nn.Linear(h_n.size(2), dec.hidden_dim).to(device)
            proj_c = nn.Linear(c_n.size(2), dec.hidden_dim).to(device)
            h_0 = proj_h(h_n)
            c_0 = proj_c(c_n)
        else:
            h_0, c_0 = h_n, c_n
    else:
        h_0, c_0 = h_n, c_n
    
    return h_0.contiguous(), c_0.contiguous()

def beam_search_lstm(enc, dec, src_ids, src_len, inv_tgt, device, beam_size=5, max_len=64, alpha=0.6):
    SOS, EOS = 1, 2
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_len_tensor = torch.tensor([src_len], dtype=torch.long, device=device)
    
    with torch.no_grad():
        enc_out, (h_n, c_n) = enc(src_tensor, src_len_tensor)
        h_0, c_0 = prepare_decoder_hidden(h_n, c_n, enc, dec, device)
        hidden = (h_0, c_0)
        
        beams = [(0.0, [SOS], hidden)]
        completed = []
        
        for step in range(max_len):
            candidates = []
            for score, seq, h in beams:
                if seq[-1] == EOS:
                    completed.append((score, seq))
                    continue
                
                input_t = torch.tensor([[seq[-1]]], dtype=torch.long, device=device)
                out, h_next = dec(input_t, h, enc_out, src_len_tensor)
                logits = out[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                topk_probs, topk_ids = log_probs.topk(beam_size)
                
                for prob, idx in zip(topk_probs, topk_ids):
                    new_score = score + prob.item()
                    new_seq = seq + [idx.item()]
                    candidates.append((new_score, new_seq, h_next))
            
            candidates.sort(key=lambda x: x[0] / (len(x[1]) ** alpha), reverse=True)
            beams = candidates[:beam_size]
            if not beams:
                break
        
        all_hyps = completed + [(score, seq, None) for score, seq, _ in beams]
        if not all_hyps:
            return ""
        
        best_seq = max(all_hyps, key=lambda x: x[0] / (len(x[1]) ** alpha))[1]
        pred_chars = []
        for pid in best_seq[1:]:
            if pid == EOS:
                break
            ch = inv_tgt.get(pid, "")
            if ch not in ("<pad>", "<sos>", "<eos>"):
                pred_chars.append(ch if ch != "<unk>" else "")
        
        return "".join(pred_chars)

def beam_search_transformer(model, src_ids, inv_tgt, device, beam_size=5, max_len=64, alpha=0.6):
    SOS, EOS = 1, 2
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad_mask = (src_tensor == 0)
    
    with torch.no_grad():
        beams = [(0.0, [SOS])]
        completed = []
        
        for step in range(max_len):
            candidates = []
            for score, seq in beams:
                if seq[-1] == EOS:
                    completed.append((score, seq))
                    continue
                
                dec_input = torch.tensor([seq], dtype=torch.long, device=device)
                tgt_pad_mask = (dec_input == 0)
                logits = model(src_tensor, dec_input,
                             src_key_padding_mask=src_pad_mask,
                             tgt_key_padding_mask=tgt_pad_mask)
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
                topk_probs, topk_ids = log_probs.topk(min(beam_size * 2, log_probs.size(0)))
                
                for prob, idx in zip(topk_probs, topk_ids):
                    new_score = score + prob.item()
                    new_seq = seq + [idx.item()]
                    candidates.append((new_score, new_seq))
            
            candidates.sort(key=lambda x: x[0] / (len(x[1]) ** alpha), reverse=True)
            beams = candidates[:beam_size]
            if not beams or (completed and len(completed) >= beam_size):
                break
        
        all_hyps = completed + beams
        if not all_hyps:
            return ""
        
        best_seq = max(all_hyps, key=lambda x: x[0] / (len(x[1]) ** alpha))[1]
        pred_chars = []
        for pid in best_seq[1:]:
            if pid == EOS:
                break
            ch = inv_tgt.get(pid, "")
            if ch not in ("<pad>", "<sos>", "<eos>"):
                pred_chars.append(ch if ch != "<unk>" else "")
        
        return "".join(pred_chars)

def infer_nvidia_nemo(sentence: str, api_key: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
        
        prompt = f"""You are a precise linguistic model for phonetic transliteration between English and Hindi (Devanagari).
Your task: Convert the following English text into its exact Hindi script equivalent, preserving pronunciation (not meaning).

Text: "{sentence}"

Output only the transliterated Hindi text — no explanation, no punctuation, and no extra text."""
        
        completion = client.chat.completions.create(
            model="nv-mistralai/mistral-nemo-12b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, top_p=0.7, max_tokens=128, stream=False
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def infer_nvidia_small(sentence: str, api_key: str) -> str:
    try:
        import requests
        invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        
        prompt = f"""You are a precise linguistic model for phonetic transliteration between English and Hindi (Devanagari). 
Output only the Hindi text — no explanations or extra words.

### Examples:
English: ram | Hindi: राम
English: shakti | Hindi: शक्ति

---
English: "{sentence}"
Hindi:"""
        
        payload = {
            "model": "mistralai/mistral-small-3.1-24b-instruct-2503",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128, "temperature": 0.0, "top_p": 1, "stream": False
        }
        
        resp = requests.post(invoke_url, headers=headers, json=payload)
        data = resp.json()
        raw_output = data["choices"][0]["message"].get("content", "") or \
                     data["choices"][0]["message"].get("reasoning_content", "")
        matches = re.findall(r"[\u0900-\u097F]+", raw_output)
        return "".join(matches) if matches else raw_output.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# ================================
# MODEL LOADING
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load LSTM model
lstm_ckpt = torch.load("lstm_checkpoint/v2.pt", map_location=device)
lstm_src_vocab = lstm_ckpt["src_vocab"]
lstm_tgt_vocab = lstm_ckpt["tgt_vocab"]
lstm_inv_tgt = lstm_ckpt["inv_tgt"]

lstm_enc = Encoder(
    input_dim=len(lstm_src_vocab), embed_dim=256, hidden_dim=512,
    num_layers=2, dropout=0.3, bidirectional=True
).to(device)

lstm_dec = AttentionDecoder(
    output_dim=len(lstm_tgt_vocab), embed_dim=256, hidden_dim=1024,
    encoder_dim=1024, num_layers=2, dropout=0.3
).to(device)

lstm_enc.load_state_dict(lstm_ckpt["enc_state"])
lstm_dec.load_state_dict(lstm_ckpt["dec_state"])
lstm_enc.eval()
lstm_dec.eval()

# Load Transformer model
trans_ckpt = torch.load("transformer_checkpoint/v2.pt", map_location=device)
trans_src_vocab = trans_ckpt["src_vocab"]
trans_tgt_vocab = trans_ckpt["tgt_vocab"]
trans_inv_tgt = trans_ckpt["inv_tgt"]

transformer = ImprovedTransformer(
    src_vocab_size=len(trans_src_vocab), tgt_vocab_size=len(trans_tgt_vocab),
    d_model=512, nhead=8, num_encoder_layers=2, num_decoder_layers=2,
    dim_feedforward=2048, dropout=0.1, activation="relu",
    max_seq_length=128, layer_norm_eps=1e-5
).to(device)

transformer.load_state_dict(trans_ckpt["model_state"])
transformer.eval()

# NVIDIA API Key (You should move this to environment variable in production)
NVIDIA_API_KEY = ""

# ================================
# GRADIO INTERFACE
# ================================
def transliterate(text, model_choice):
    if not text.strip():
        return "Please enter some text"
    
    words = text.strip().split()
    results = []
    
    if model_choice == "LSTM with Attention":
        for word in words:
            ids = [lstm_src_vocab.get(ch, lstm_src_vocab["<unk>"]) for ch in word.lower()]
            pred = beam_search_lstm(lstm_enc, lstm_dec, ids, len(ids), 
                                   lstm_inv_tgt, device)
            results.append(pred)
    
    elif model_choice == "Transformer":
        for word in words:
            ids = [trans_src_vocab.get(ch, trans_src_vocab["<unk>"]) for ch in word.lower()]
            pred = beam_search_transformer(transformer, ids, trans_inv_tgt, device)
            results.append(pred)
    
    elif model_choice == "NVIDIA Mistral Nemo":
        return infer_nvidia_nemo(text, NVIDIA_API_KEY)
    
    elif model_choice == "NVIDIA Mistral Small":
        return infer_nvidia_small(text, NVIDIA_API_KEY)
    
    return " ".join(results)

# Create Gradio interface
with gr.Blocks(title="English to Hindi Transliteration") as demo:
    gr.Markdown("# 🔤 English to Hindi Transliteration")
    gr.Markdown("Convert English text to Hindi (Devanagari) script using various models")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Enter English Text",
                placeholder="e.g., namaste dost",
                lines=3
            )
            model_dropdown = gr.Dropdown(
                choices=[
                    "LSTM with Attention",
                    "Transformer",
                    "NVIDIA Mistral Nemo",
                    "NVIDIA Mistral Small"
                ],
                value="LSTM with Attention",
                label="Select Model"
            )
            submit_btn = gr.Button("Transliterate", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Hindi Output",
                lines=3,
                interactive=False
            )
    
    # gr.Markdown("### Examples:")
    # gr.Examples(
    #     examples=[
    #         ["namaste", "LSTM with Attention"],
    #         ["ram shakti diwali", "Transformer"],
    #         ["krishna birthday", "NVIDIA Mistral Nemo"],
    #         ["janamdivas rakha", "NVIDIA Mistral Small"]
    #     ],
    #     inputs=[input_text, model_dropdown]
    # )
    
    submit_btn.click(
        fn=transliterate,
        inputs=[input_text, model_dropdown],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()